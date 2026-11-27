"""主端控制线程，计算最新指令并用 CAN 进程确认的反馈发布状态。"""

from collections import deque
from dataclasses import dataclass, replace
from math import degrees, isfinite
from threading import Condition, Thread
from time import monotonic, perf_counter

from ArmController import PredictionObservation
from CanProcess import CanProcessLost
from DMMotor import MODE_NAMES, Read
from USBCANFD import OperationInterrupted
from zlgcan import ZlgError

JOINTS = tuple(range(1, 7))
ALL = (*JOINTS, 7)
PRIORITY = {"stop": 1, "disable": 2, "disable_all": 3, "close": 4}


@dataclass(frozen=True)
class JointSample:
    feedback: tuple
    times: tuple

    @property
    def sampled_at(self):
        return max(self.times)

    @property
    def oldest_at(self):
        return min(self.times)


@dataclass(frozen=True)
class Snapshot:
    opened: bool
    busy: bool
    first_read: bool
    verified: frozenset
    owned: frozenset
    uncertain: frozenset
    streaming: bool
    gravity: bool
    force_feedback: bool
    filtered_velocity_error: tuple[float, ...] | None
    local_damping_torque: tuple[float, ...] | None
    force_prediction_enabled: bool
    force_prediction_weight: tuple[float, ...] | None
    prediction: PredictionObservation | None
    joint_sample_at: float
    command_until: float
    commands: tuple
    tracking: bool
    coherent: bool
    faulted: bool
    motors: tuple
    message: str
    tx: int
    rx: int
    failed: int
    cycles: int
    hz: float
    feedback_at: float
    remote_action: str


class CanWorker(Thread):
    def __init__(self, bus, listen_only, speed_limit, pose_period, *, queued_cycles, force_feedback_max_age,
                 force_requires_slave_follow=True, controller=None, slave_link=None):
        super().__init__(name="DMArm-Control")
        self.bus = bus
        self.controller = controller
        if not isfinite(force_feedback_max_age) or force_feedback_max_age <= 0:
            raise ValueError("力反馈六轴采样时限须为正的有限秒数")
        self.force_feedback_max_age = force_feedback_max_age
        self.remote_link = None
        self.remote_action = ""
        self.remote_frame = None
        self.slave_link = slave_link
        self.velocity_stream = None
        self.prediction_stream = None
        self.force_requires_slave_follow = force_requires_slave_follow
        self.clear_force_feedback()
        self.force_output_pending = False
        self.commands = {}
        self.joint_sample = None
        self.listen_only = listen_only
        self.tracking = False
        self.pose_period = pose_period
        self.queued_cycles = queued_cycles
        self.speed_limit = speed_limit
        self.cv = Condition()
        self.pending = None
        self.current = "open"
        self.events = deque(maxlen=250)
        self.first_read = False
        self.verified = set()
        self.owned = set()
        self.uncertain = set()
        self.streaming = False
        self.coherent = True
        self.targets = {}
        self.next_monitor = 0.0
        self.message = "正在打开设备"
        self.done = False
        self.faulted = False
        self.publish()

    def publish(self):
        if not self.streaming:
            self.observe_idle_velocities()
        with self.cv:
            # 滤波器会原地更新数组，发布快照时复制六轴数值供异步记录。
            velocity_error = self.controller.filtered_velocity_error if self.controller is not None else None
            local_damping = self.controller.local_damping_torque if self.controller is not None else None
            self.state = Snapshot(
                self.bus.started, bool(self.current or self.pending), self.first_read,
                frozenset(self.verified), frozenset(self.owned), frozenset(self.uncertain),
                self.streaming, bool(self.controller and self.controller.gravity_on),
                self.force_link is not None,
                tuple(velocity_error) if velocity_error is not None else None,
                tuple(local_damping) if local_damping is not None else None,
                bool(self.controller and self.controller.force_prediction_enabled),
                tuple(self.controller.force_prediction_weight) if self.controller else None,
                self.controller.prediction_observation if self.controller else None,
                self.joint_sample.oldest_at if self.joint_sample else 0.0,
                self.bus.confirmed.expires_at if self.bus.applied else 0.0,
                tuple(self.commands.get(n) for n in JOINTS), self.tracking, self.coherent, self.faulted,
                tuple((m.mode_text, m.feedback) for m in self.bus.motors.values()),
                self.message, self.bus.tx, self.bus.rx, self.bus.failed, self.bus.cycles,
                self.bus.hz if self.streaming else 0.0,
                min(m.feedback_at for m in self.bus.motors.values()), self.remote_action,
            )

    def snapshot(self):
        with self.cv:
            messages = tuple(self.events)
            self.events.clear()
            state = replace(self.state, busy=bool(self.current or self.pending))
        return state, messages

    def latest_state(self):
        state = self.state
        if not state.streaming or not state.command_until:
            return state
        latest = self.bus.latest_motion(state.cycles)
        if latest is not None:
            command, cycles, hz, tx, rx, failed, records = latest
            motors = tuple((MODE_NAMES.get(mode, "—"), fb) for mode, fb, _ in records)
            if all(a[0] == b[0] for a, b in zip(state.motors, motors)):
                commands = list(state.commands)
                for node, mode, (_, data) in command.targets:
                    if node in JOINTS:
                        commands[node - 1] = self.bus.motors[node].joint_command_value(mode, data)
                return replace(state, cycles=cycles, hz=hz, tx=tx, rx=rx, failed=failed, motors=motors,
                               commands=tuple(commands), feedback_at=min(r[2] for r in records),
                               joint_sample_at=min(r[2] for r in records[:6]), command_until=command.expires_at)
        return state if perf_counter() < state.command_until else None

    def log(self, message):
        with self.cv:
            self.events.append(message)

    def report(self, message):
        self.message = message
        self.log(message)

    def submit(self, action, nodes=()):
        with self.cv:
            if self.done or (action == "close" and not self.can_close()):
                return False
            priority = PRIORITY.get(action, 0)
            if self.pending and priority <= PRIORITY.get(self.pending[0], 0):
                return False
            if self.current and priority <= PRIORITY.get(self.current, 0):
                return False
            self.pending = (action, tuple(nodes))
            if priority:
                self.bus.interrupt()
            self.cv.notify()
            return True

    def can_close(self):
        """查询可以中断，周期启停、换模和使失能操作须先完成。"""
        with self.cv:
            return (not self.streaming and self.current in (None, "open", "read", "first", "pose_on")
                    and (self.pending is None or self.pending[0] == "read"))

    def checkpoint(self):
        with self.cv:
            if self.pending and PRIORITY.get(self.pending[0], 0) > PRIORITY.get(self.current, 0):
                raise OperationInterrupted()

    def mode(self, node, check=None):
        value = self.bus.read(node, Read.MODE, check or self.checkpoint, self.log)
        if value not in (1, 2, 3, 4):
            raise ZlgError(f"ID {node} 模式寄存器值 {value} 未识别")
        return value

    def status(self, node, check=None, verbose=True):
        return self.bus.read(node, Read.STATUS, check or self.checkpoint, self.log if verbose else None)

    def special(self, node, code, check=None):
        data = bytes([0xFF] * 7 + [code])
        self.bus.exchange([(self.bus.motors[node].command_id, data)], {node: Read.STATUS},
                          check or self.checkpoint, self.log)

    def require_off(self, node):
        fb = self.status(node)
        if fb.status != 0:
            raise ZlgError(f"ID {node} 当前为 {fb.status_text}，操作要求失能")
        return fb

    def disable(self, nodes):
        """即使某轴失联，也给其余目标逐台发失能。仅状态回读 0 才销账。"""
        self.streaming = False
        self.clear_force_feedback()
        self.invalidate_sample()
        self.targets.clear()
        self.clear_remote()
        self.commands.clear()
        if self.controller:
            self.controller.gravity_on = False
        result = self.bus.request("disable", (set(nodes), self.coherent), self.log)
        self.verified.difference_update(result.nodes)
        self.verified.update(result.verified)
        self.owned.difference_update(result.nodes - result.uncertain)
        self.uncertain.difference_update(result.nodes)
        self.uncertain.update(result.uncertain)
        if self.uncertain:
            self.report(f"失能未确认 {sorted(self.uncertain)}；请支撑并物理停机")
            return False
        self.force_output_pending = False
        if result.queue_error:
            self.report(f"失能已回读；清发送队列失败：{result.queue_error}")
            self.faulted = True
            return False
        self.report(f"失能已确认 {sorted(nodes)}")
        return True

    def clear_motor_command(self, node):
        self.targets[node] = self.bus.motors[node].empty_target()
        self.commands[node] = 0.0

    def enable(self, nodes):
        if self.owned or self.uncertain or not self.coherent or not set(nodes) <= self.verified:
            raise ZlgError("请先完成目标轴失能确认和模式核对")
        for node in nodes:
            self.special(node, 0xFB)
            self.clear_motor_command(node)
            self.owned.add(node)
            self.verified.discard(node)
            self.special(node, 0xFC)
            if self.status(node).status != 1:
                raise ZlgError(f"ID {node} 使能未确认")
        self.report(f"使能已回读 {sorted(self.owned)}")

    def switch_mode(self, value, nodes=JOINTS):
        if self.owned or self.uncertain or not set(nodes) <= self.verified:
            raise ZlgError("切换模式前请先确认所选节点均已失能")
        for node in nodes:
            self.require_off(node)
        self.write_modes(value, nodes)
        self.report(f"节点 {list(nodes)} 已回读 {self.bus.motors[nodes[0]].mode_text}")

    def write_modes(self, value, nodes):
        self.clear_force_feedback()
        self.invalidate_sample()
        # 寄存器写入与回读是额外批次；分频发送的夹钳只在到期的控制轮中加入。
        continuing = {n: target for n, target in self.targets.items()
                      if self.streaming and n not in nodes and (n != 7 or self.bus.tool_update_cycles == 1)}
        self.coherent = False
        self.bus.request("modes", (value, tuple(nodes), continuing), self.log)
        self.coherent = True

    def start_cycles(self):
        if not self.owned or self.uncertain or not self.coherent:
            raise ZlgError("周期发送需要已确认使能的目标轴")
        self.invalidate_sample()
        for node in sorted(self.owned):
            fb = self.status(node)
            if fb.status != 1:
                raise ZlgError(f"ID {node} 状态为 {fb.status_text}")
            if node not in self.targets:
                motor = self.bus.motors[node]
                if motor.mode == 2:
                    self.set_positions({node: fb.position})
                else:
                    self.clear_motor_command(node)
        self.bus.request("start", (self.queued_cycles, tuple(self.targets)))
        self.joint_feedback = {n: Read.STATUS for n in JOINTS if n in self.targets}
        self.streaming = True
        mode = "队列发送" if self.queued_cycles else "直接批发"
        self.report(f"周期发送 {sorted(self.targets)} · {mode}")

    def invalidate_sample(self):
        self.joint_sample = None
        self.reset_velocity_filter()

    def confirm_cycle(self, expected):
        self.check_enabled_feedback(expected)
        for node, mode, (_, data) in self.bus.confirmed.targets:
            if node in self.joint_feedback:
                self.commands[node] = self.bus.motors[node].joint_command_value(mode, data)
        self.update_joint_sample()

    def update_joint_sample(self):
        if len(self.joint_feedback) == len(JOINTS):
            self.joint_sample = JointSample(tuple(self.bus.motors[n].feedback for n in JOINTS),
                                            tuple(self.bus.motors[n].feedback_at for n in JOINTS))

    def cycle(self):
        ctl = self.controller
        motors = self.bus.motors
        if self.bus.refresh_latest():
            self.confirm_cycle(self.joint_feedback)
        sampled_at = min(motors[n].feedback_at for n in (self.joint_feedback or self.targets))
        if self.remote_action:
            self.update_remote()
        link = self.slave_link
        frame = link.fresh() if link is not None else None
        if ctl is not None:
            self.update_velocity_observation(link, frame, self.joint_sample, self.force_feedback_max_age)
        force_torque = self.force_feedback_torque(link, frame)
        if ctl and ctl.gravity_on and any(motors[n].mode == 1 for n in JOINTS):
            feedback = tuple(motors[n].feedback for n in JOINTS)
            torques = ctl.gravity_commands(feedback)
            if force_torque is not None:
                torques += force_torque
            for node in JOINTS:
                motor = motors[node]
                if motor.mode == 1:
                    torque = max(-motor.output_limit, min(motor.output_limit, torques[node - 1]))
                    self.targets[node] = motor.command_id, motor.encode_mit(0, 0, torque, 0, 0)
        expires_at = sampled_at + self.bus.command_max_age
        if force_torque is not None:
            expires_at = min(expires_at, sampled_at + self.force_feedback_max_age)
        remote = frame if force_torque is not None else self.remote_frame
        if remote is not None and (force_torque is not None or self.remote_action == "follow"):
            remote_link = self.force_link if force_torque is not None else self.remote_link
            expires_at = min(expires_at, perf_counter() + remote_link.timeout - (monotonic() - remote.received))
        self.checkpoint()
        wait_applied = bool(self.current or (self.force_output_pending and force_torque is None))
        if force_torque is not None:
            self.force_output_pending = True
        has_tool = self.bus.cycle(self.targets, sampled_at, expires_at, wait_applied)
        expected = self.targets if has_tool else self.joint_feedback
        self.confirm_cycle(expected)
        self.force_output_pending = force_torque is not None

    def check_enabled_feedback(self, nodes):
        for node in nodes:
            fb = self.bus.motors[node].feedback
            if fb.status != 1:
                raise ZlgError(f"ID {node} 反馈退出使能：{fb.status_text}")

    def stop(self):
        self.clear_force_feedback()
        if self.force_output_pending:
            self.cycle()
        if self.remote_action:
            self.hold_remote("停止发送")
            self.cycle()
        self.bus.request("stop")
        self.streaming = False
        self.invalidate_sample()
        self.report("周期发送已停止，发送队列已清空")

    def change_running_mode(self, nodes, value):
        """在使能状态下交接，PV 锁存原始电机角，MIT 首帧沿用当前补偿。"""
        if not nodes:
            return
        self.clear_force_feedback()
        if self.streaming:
            self.cycle()
        else:
            for n in nodes:
                self.status(n)
            self.check_enabled_feedback(nodes)
        feedback = {n: self.bus.motors[n].feedback for n in nodes}
        self.write_modes(value, nodes)
        for n, fb in feedback.items():
            motor = self.bus.motors[n]
            if value == 2:
                self.targets[n] = motor.command_id, motor.encode_pv(fb.position, self.speed_limit)
                self.commands[n] = fb.position
            else:
                self.clear_motor_command(n)
        if not self.streaming:
            return
        # 新模式的首个控制周期按最新反馈填入已开启的 MIT 重力矩。
        self.cycle()
        angle_change, node = max((abs(self.bus.motors[n].feedback.position - fb.position), n)
                                 for n, fb in feedback.items())
        self.log(f"换模前末帧至新模式首帧，最大原始角差 J{node} {degrees(angle_change):.3f}°")

    def local_control(self, action, args):
        ctl = self.controller
        motors = self.bus.motors
        if self.uncertain or self.faulted or not self.coherent:
            raise ZlgError("当前通道状态禁止本地控制")
        if action in ("lock", "unlock"):
            node, = args
            previous = 1 if action == "lock" else 2
            if node not in JOINTS or node not in self.owned or motors[node].mode != previous:
                raise ZlgError("当前关节状态不允许此锁定操作")
            self.change_running_mode((node,), 2 if action == "lock" else 1)
            self.report(f"J{node} 锁定目标已设为实测位置" if action == "lock" else f"J{node} 已释放到 MIT")
            return
        if not self.streaming or not set(JOINTS) <= self.owned:
            raise ZlgError("需要六关节均已使能并周期反馈")
        if action in ("gravity_on", "gravity_off"):
            ctl.gravity_on = action == "gravity_on"
            if not ctl.gravity_on:
                self.clear_force_feedback()
                for n in JOINTS:
                    if motors[n].mode == 1:
                        self.clear_motor_command(n)
            self.report("MIT 重力补偿已开启" if ctl.gravity_on else "MIT 重力补偿已关闭")
            return
        positions = ctl.motor_home(motors)
        ctl.gravity_on = True
        switching = tuple(n for n in JOINTS if motors[n].mode != 2)
        if switching:
            self.write_modes(2, switching)
        self.set_positions(positions)
        self.report(f"已设置六轴 PV 预设目标，限速 {self.speed_limit:g} rad/s")

    def set_positions(self, positions):
        for n, pos in positions.items():
            motor = self.bus.motors[n]
            self.targets[n] = motor.command_id, motor.encode_pv(pos, self.speed_limit)
            self.commands[n] = pos

    def clear_remote(self):
        self.remote_link = self.remote_frame = None
        self.remote_action = ""

    def clear_force_feedback(self):
        self.force_link = self.force_generation = None

    def reset_velocity_filter(self):
        self.velocity_stream = None
        self.prediction_stream = None
        if self.controller is not None:
            self.controller.reset_velocity_filter()

    def update_velocity_observation(self, link, frame, sample, max_dt):
        if frame is None or sample is None or perf_counter() - sample.oldest_at > max_dt:
            self.reset_velocity_filter()
            return
        # 不同 UDP 连接的段编号会重复。换连接或断流恢复时，首拍重新取实测速度差。
        stream = (link, frame.generation)
        if stream != self.velocity_stream:
            self.controller.reset_velocity_filter()
            self.velocity_stream = stream
        self.controller.update_velocity_error(sample.feedback, frame.dq, sample.sampled_at, max_dt)
        if self.controller.force_prediction_enabled:
            session, commands = link.history
            if not self.slave_allows_force_feedback(frame) or not commands or frame.received < commands[0].sent_at:
                self.controller.reset_prediction()
                self.prediction_stream = None
                return
            prediction_stream = (link, frame.generation, session)
            if prediction_stream != self.prediction_stream:
                self.controller.reset_prediction()
                self.prediction_stream = prediction_stream
            self.controller.update_prediction(sample.feedback, frame, commands, sample.sampled_at, max_dt)

    def observe_idle_velocities(self):
        """只听、读取状态及失能姿态采样时，用已有回包更新速度差，不额外发送查询。"""
        if self.controller is None:
            return
        link = self.slave_link
        frame = link.fresh() if link is not None else None
        sample = None
        if frame is not None and self.bus.started and self.coherent and not self.faulted:
            motors = self.bus.motors
            feedback = tuple(motors[n].feedback for n in JOINTS)
            if all(fb is not None for fb in feedback):
                sample = JointSample(feedback, tuple(motors[n].feedback_at for n in JOINTS))
        self.update_velocity_observation(link, frame, sample, self.bus.timeout)

    def slave_allows_force_feedback(self, frame):
        return not self.force_requires_slave_follow or (frame is not None and frame.teleop_state != 0)

    def can_reflect_force(self, state):
        return (state.opened and state.streaming and state.gravity and state.coherent
                and not state.uncertain and not state.faulted
                and state.joint_sample_at > 0 and perf_counter() - state.joint_sample_at <= self.force_feedback_max_age
                and any(self.controller.force_limit)
                and (not self.controller.force_prediction_enabled or state.prediction is not None)
                and all(n in state.owned and mode == "MIT" for n, (mode, _) in zip(JOINTS, state.motors)))

    def start_force_feedback(self, link):
        frame = link.fresh() if link else None
        if frame is None or not self.can_reflect_force(self.state):
            self.report("力反馈需要六轴 MIT 使能、重力补偿及新鲜主从反馈；预测补偿还需要主端持续发布并完成初始化")
            return
        if not self.slave_allows_force_feedback(frame):
            self.report("从端未开启跟随，无法开启阻抗力反馈")
            return
        self.force_link = link
        self.force_generation = frame.generation
        self.report("自由空间预测补偿力反馈已开启" if self.controller.force_prediction_enabled else "关节阻抗力反馈已开启")

    def force_feedback_torque(self, link, frame):
        """力反馈开启时使用本轮观测值输出阻抗力矩。"""
        if self.force_link is None:
            return None
        if link is not self.force_link or frame is None or frame.generation != self.force_generation:
            self.clear_force_feedback()
            self.report("从端关节位置断流，阻抗力反馈已关闭")
            return None
        if not self.slave_allows_force_feedback(frame):
            self.clear_force_feedback()
            self.report("从端已退出跟随，阻抗力反馈已关闭")
            return None
        sample = self.joint_sample
        if sample is None or perf_counter() - sample.oldest_at > self.force_feedback_max_age:
            self.clear_force_feedback()
            self.report("主端六轴反馈缺失或已过期，阻抗力反馈已关闭")
            return None
        if self.controller.force_prediction_enabled and self.controller.prediction_observation is None:
            self.clear_force_feedback()
            self.report("主端发布已结束或预测会话未就绪，力反馈已关闭")
            return None
        return self.controller.impedance_torque(sample.feedback, frame.q)

    def hold_remote(self, reason):
        self.clear_remote()
        self.set_positions({n: self.bus.motors[n].feedback.position for n in JOINTS})
        self.report(f"{reason}，六轴按当前反馈位置持位")

    def start_remote(self, action, link):
        frame = link.fresh()
        if (not frame or not self.streaming or self.uncertain or self.faulted or not self.coherent
                or not set(JOINTS) <= self.owned):
            self.report("遥操作需要新鲜从端状态及已使能六轴的周期反馈")
            return
        self.controller.gravity_on = True
        switching = tuple(n for n in JOINTS if self.bus.motors[n].mode != 2)
        if switching:
            self.change_running_mode(switching, 2)
            latest = link.fresh()
            if not latest or latest.generation != frame.generation:
                self.hold_remote("切换 PV 期间从端断流")
                return
            frame = latest
        self.remote_link, self.remote_frame, self.remote_action = link, frame, action
        self.set_positions(self.controller.motor_targets(frame.q))
        self.report("主端持续跟随从端" if action == "follow" else "主端已锁存一次同步目标")

    def update_remote(self):
        frame = self.remote_link.fresh()
        if not frame or frame.generation != self.remote_frame.generation:
            self.hold_remote("从端断流或已断开")
        elif self.remote_action == "follow" and frame is not self.remote_frame:
            self.set_positions(self.controller.motor_targets(frame.q))
            self.remote_frame = frame

    def set_zero(self, node):
        if (self.streaming or self.tracking or self.faulted or self.uncertain or not self.coherent
                or node not in JOINTS or node not in (self.owned | self.verified)):
            self.report("请停止周期收发，并选择已读取状态的关节设零")
            return
        self.invalidate_sample()
        motor = self.bus.motors[node]
        target = motor.empty_target()
        # 先替换旧坐标目标，设零之后再向电机发送空指令。
        if node in self.owned:
            self.targets[node] = target
            self.commands[node] = 0.0
        self.special(node, 0xFE)
        self.bus.exchange([target], {node: Read.STATUS}, self.checkpoint, self.log)
        fb = motor.feedback
        if fb.status != int(node in self.owned):
            raise ZlgError(f"J{node} 设零后回读 {fb.status_text}")
        self.report(f"J{node} 设零已应答，回读位置 {fb.position:.4f} rad，{fb.status_text}")

    def dispatch(self, action, nodes):
        if action in (*PRIORITY, "pose_off"):
            self.tracking = False
        if action == "close":
            try:
                self.bus.close()
            except (ZlgError, OSError) as error:
                self.faulted = True
                self.report(f"关闭设备失败：{error}")
                self.done = self.bus.process is None
                return
            self.invalidate_sample()
            self.targets.clear()
            self.commands.clear()
            self.clear_remote()
            self.clear_force_feedback()
            if self.controller:
                self.controller.gravity_on = False
            self.done = True
            self.report("CAN FD 设备已释放")
            return
        if action == "force_on":
            self.start_force_feedback(nodes[0])
            return
        if action == "force_off":
            self.clear_force_feedback()
            self.report("力反馈已关闭")
            return
        if self.listen_only or not self.bus.started:
            raise ZlgError("控制操作需要已启动且可发送的通道")
        if action in ("disable", "disable_all"):
            if action == "disable_all":
                nodes = ALL
            self.disable(set(nodes))
            return
        if action == "stop":
            self.stop()
            return
        if action in ("sync", "follow"):
            self.start_remote(action, nodes[0])
            return
        if action == "remote_off":
            if self.remote_action:
                self.hold_remote("人工停止遥操作")
            return
        if self.remote_action and action in ("home", "lock", "unlock", "pv", "mit"):
            self.hold_remote("切换本地控制")
        if action in ("lock", "unlock", "gravity_on", "gravity_off", "home"):
            self.local_control(action, nodes)
            return
        if action in ("mit", "pv") and self.owned:
            if self.faulted or self.uncertain or not self.coherent or not set(JOINTS) <= self.owned:
                raise ZlgError("当前状态禁止整臂换模")
            value = 1 if action == "mit" else 2
            self.change_running_mode(tuple(n for n in JOINTS if self.bus.motors[n].mode != value), value)
            self.report(f"六轴已切换到 {self.bus.motors[1].mode_text}")
            return
        if action == "zero":
            node, = nodes
            self.set_zero(node)
            return
        if self.owned and action != "start":
            raise ZlgError("操作前请先失能")
        if self.faulted or self.uncertain or self.streaming:
            raise ZlgError("当前通道状态禁止此操作")
        if action == "pose_off":
            self.report("连续姿态读取已停止")
            return
        if self.tracking:
            raise ZlgError("请先结束连续姿态读取")
        if action == "pose_on":
            if not self.first_read:
                raise ZlgError("请先读取 J1 模式")
            self.sample_pose()
            self.tracking = True
            self.report("正在连续读取七节点失能姿态")
        elif action == "first":
            self.first_read = False
            self.mode(1)
            self.first_read = True
            self.report("J1 模式已回读")
        elif action == "read":
            self.verified.clear()
            self.first_read = False
            self.mode(1)
            self.first_read = True
            missing = []
            enabled = set()
            for node in ALL:
                try:
                    if node != 1:
                        self.mode(node)
                    status = self.status(node).status
                    if status == 0:
                        self.verified.add(node)
                    elif status == 1:
                        enabled.add(node)
                except (ZlgError, OSError) as error:
                    missing.append(str(node))
                    self.log(str(error))
            if enabled == set(ALL):
                unsupported = [n for n, m in self.bus.motors.items()
                               if m.mode not in (1, 2) and not (n == 7 and m.mode == 4)]
                if unsupported:
                    self.report(f"节点 {unsupported} 的当前模式不支持周期发送")
                else:
                    self.bus.request("adopt", (enabled,))
                    self.owned.update(enabled)
                    self.report("六轴和夹钳均已使能，可直接开始发送")
            else:
                self.report("六轴和夹钳已读取" if not missing else f"待读取节点：{', '.join(missing)}")
        elif action == "enable":
            self.enable(nodes)
        elif action in ("mit", "pv"):
            self.switch_mode(1 if action == "mit" else 2)
        elif action in ("tool_mit", "tool_pvt"):
            self.switch_mode(1 if action == "tool_mit" else 4, (7,))
        elif action == "start":
            self.start_cycles()
        elif action == "clear":
            for node in nodes:
                self.mode(node)
                if self.status(node).status == 1:
                    raise ZlgError(f"ID {node} 清错前须失能")
                self.special(node, 0xFB)
                fb = self.require_off(node)
                self.log(f"ID {node} 清错后回读 {fb.status_text}，位置 {fb.position:.4f} rad")
            self.report(f"清错已应答 {list(nodes)}")

    def sample_pose(self):
        for node in ALL:
            fb = self.status(node, verbose=False)
            if fb.status != 0:
                raise ZlgError(f"ID {node} 回读 {fb.status_text}，结束姿态读取")
        self.next_monitor = monotonic() + self.pose_period

    def connection_lost(self, error):
        self.uncertain.update(self.owned)
        self.streaming = self.tracking = False
        self.faulted = True
        self.report(str(error))

    def run(self):
        try:
            self.report(self.bus.open(self.listen_only))
            with self.cv:
                self.current = None
                if not self.listen_only and not self.pending:
                    self.pending = ("read", ())
                self.publish()
            self.loop()
        except CanProcessLost as error:
            self.connection_lost(error)
        except (ZlgError, OSError, AttributeError) as error:
            self.report(f"设备错误：{error}")
        finally:
            try:
                if self.bus.process is not None and self.owned and not self.done:
                    self.disable(self.owned.copy())
            except CanProcessLost as error:
                self.connection_lost(error)
            finally:
                if self.bus.process is not None:
                    try:
                        self.bus.close()
                    except CanProcessLost as error:
                        self.connection_lost(error)
                    except (ZlgError, OSError) as error:
                        self.report(str(error))
            self.done = True
            self.current = None
            self.publish()

    def loop(self):
        while not self.done:
            with self.cv:
                command = self.pending
                self.pending = None
                self.current = command[0] if command else None
                self.bus.generation = self.bus.epoch.value
            try:
                if command:
                    self.dispatch(*command)
                elif self.tracking and monotonic() >= self.next_monitor:
                    self.sample_pose()
                elif self.streaming:
                    self.cycle()
                elif not self.faulted:
                    self.bus.request("receive", log=self.log)
            except OperationInterrupted:
                self.log("普通操作已中断，处理停止请求")
                # 部分使能或换模未核对时先逐轴失能；模式已知时由待处理的停止操作接管。
                switching = self.current in ("lock", "unlock", "home", "mit", "pv", "sync", "follow")
                if self.owned and (self.current == "enable" or (switching and not self.coherent)):
                    self.disable(self.owned.copy())
                    with self.cv:
                        if self.pending and self.pending[0] == "stop":
                            # 已清队列并逐轴执行失能，保留回读结果供操作员确认。
                            self.pending = None
            except (ZlgError, OSError, ValueError, OverflowError) as error:
                self.tracking = False
                message = str(error)
                self.log(message)
                if self.owned:
                    self.disable(self.owned.copy() | self.uncertain)
                    self.report(f"{message}；{self.message}")
                else:
                    self.report(message)
                if not command and not self.streaming:
                    self.faulted = True
            with self.cv:
                self.current = None
                self.publish()
                if not self.streaming and not self.pending and not self.done:
                    self.cv.wait(0.005)
