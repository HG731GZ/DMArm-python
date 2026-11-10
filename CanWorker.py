"""CAN FD 独占线程，串行执行操作，向 Qt 发布完整状态快照。"""

from collections import deque
from dataclasses import dataclass, replace
from math import degrees
from threading import Condition, Thread
from time import monotonic, perf_counter

from DMMotor import MODE_REGISTER, PARAM_ID, Read
from USBCANFD import OperationInterrupted
from zlgcan import ZlgError

JOINTS = tuple(range(1, 7))
ALL = (*JOINTS, 7)
PRIORITY = {"stop": 1, "disable": 2, "disable_all": 3, "close": 4, "release": 5}


def uninterrupted():
    """单轴失能及其回读必须完成后才处理下一项操作。"""


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
    homing: bool
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


class CanWorker(Thread):
    def __init__(self, bus, listen_only, period, speed_limit, stop_speed, stop_timeout, pose_period,
                 stop_sample_period=0.010, *, queued_cycles, controller=None):
        super().__init__(name="DMArm-CAN-FD")
        self.bus = bus
        self.controller = controller
        self.commands = {}
        self.listen_only = listen_only
        self.tracking = False
        self.pose_period = pose_period
        self.period = period
        self.queued_cycles = queued_cycles
        self.speed_limit = speed_limit
        self.stop_speed = stop_speed
        self.stop_timeout = stop_timeout
        self.stop_sample_period = stop_sample_period
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
        self.cycles = 0
        self.run_cycles = 0
        self.hz = 0.0
        self.next_cycle = self.next_monitor = 0.0
        self.message = "正在打开设备"
        self.physically_stopped = False
        self.done = False
        self.faulted = False
        self.publish()

    def publish(self):
        with self.cv:
            self.state = Snapshot(
                self.bus.started, bool(self.current or self.pending), self.first_read,
                frozenset(self.verified), frozenset(self.owned), frozenset(self.uncertain),
                self.streaming, bool(self.controller and self.controller.gravity_on),
                bool(self.controller and self.controller.home_targets),
                tuple(self.commands.get(n) for n in JOINTS), self.tracking, self.coherent, self.faulted,
                tuple((m.mode_text, m.feedback) for m in self.bus.motors.values()),
                self.message, self.bus.tx, self.bus.rx, self.bus.failed, self.cycles, self.hz,
            )

    def snapshot(self):
        with self.cv:
            messages = tuple(self.events)
            self.events.clear()
            return replace(self.state, busy=bool(self.current or self.pending)), messages

    def log(self, message):
        with self.cv:
            self.events.append(message)

    def report(self, message):
        self.message = message
        self.log(message)

    def submit(self, action, nodes=()):
        with self.cv:
            if self.done:
                return False
            priority = PRIORITY.get(action, 0)
            if self.pending and priority <= PRIORITY.get(self.pending[0], 0):
                return False
            if self.current and priority <= PRIORITY.get(self.current, 0):
                return False
            self.pending = (action, tuple(nodes))
            self.cv.notify()
            return True

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
                          check or self.checkpoint, self.log, disabling=(code == 0xFD))

    def require_off(self, node):
        fb = self.status(node)
        if fb.status != 0:
            raise ZlgError(f"ID {node} 当前为 {fb.status_text}，操作要求失能")
        return fb

    def disable(self, nodes):
        """即使某轴失联，也给其余目标逐台发失能。仅状态回读 0 才销账。"""
        self.streaming = False
        self.clear_targets()
        clearing_error = None
        try:
            self.bus.backend.clear_transmit_queue()
        except (ZlgError, OSError) as error:
            clearing_error = error
            self.log(str(error))
        for node in sorted(nodes):
            self.verified.discard(node)
            try:
                if self.bus.motors[node].mode not in (1, 2, 3, 4) or not self.coherent:
                    self.mode(node, uninterrupted)
                acknowledged = True
                try:
                    self.special(node, 0xFD, uninterrupted)
                except (ZlgError, OSError) as error:
                    acknowledged = False
                    self.log(f"ID {node} 失能帧：{error}")
                fb = self.status(node, uninterrupted)
                if fb.status != 0:
                    raise ZlgError(f"回读 {fb.status_text}")
            except (ZlgError, OSError) as error:
                self.uncertain.add(node)
                self.log(f"ID {node} 失能未确认：{error}")
            else:
                self.owned.discard(node)
                self.uncertain.discard(node)
                if acknowledged:
                    self.verified.add(node)
                self.log(f"ID {node} 失能已回读 0")
        if self.uncertain:
            self.report(f"失能未确认 {sorted(self.uncertain)}；请支撑并物理停机")
            return False
        if clearing_error:
            self.report(f"失能已回读；清发送队列失败：{clearing_error}")
            self.faulted = True
            return False
        self.report(f"失能已确认 {sorted(nodes)}")
        return True

    def clear_targets(self):
        self.targets.clear()
        self.commands.clear()
        if self.controller:
            self.controller.reset()

    def target(self, node, fb):
        motor = self.bus.motors[node]
        return motor.command_id, motor.resting_target(fb.position, self.speed_limit)

    def enable(self, nodes):
        if self.owned or self.uncertain or not self.coherent or not set(nodes) <= self.verified:
            raise ZlgError("请先完成目标轴失能确认和模式核对")
        # 在首个使能帧之前，核对所有目标的模式、状态及可用的静止载荷。
        for node in nodes:
            self.mode(node)
            self.target(node, self.require_off(node))
        for node in nodes:
            motor = self.bus.motors[node]
            fb = self.require_off(node)
            if motor.mode != 1 and abs(fb.velocity) > self.stop_speed:
                raise ZlgError(f"ID {node} 仍在运动，无法建立当前位置目标")
            target = self.target(node, fb)
            self.bus.exchange([target], {node: Read.STATUS}, self.checkpoint, self.log)
            if motor.feedback.status != 0:
                raise ZlgError(f"ID {node} 静止目标预置后未回读失能")
            self.owned.add(node)
            self.verified.discard(node)
            self.special(node, 0xFC)
            if self.status(node).status != 1:
                raise ZlgError(f"ID {node} 使能未确认")
        self.next_monitor = monotonic()
        self.report(f"使能已回读 {sorted(self.owned)}")

    def switch_mode(self, value, nodes=JOINTS):
        if self.owned or self.uncertain or not set(nodes) <= self.verified:
            raise ZlgError("切换模式前请先确认所选节点均已失能")
        for node in nodes:
            self.require_off(node)
        self.write_modes(value, nodes)
        self.report(f"节点 {list(nodes)} 已回读 {self.bus.motors[nodes[0]].mode_text}")

    def exchange_mode_register(self, payloads, kind, continuing):
        """换模轴收寄存器应答，同批的其余轴继续控制并收使能反馈。"""
        frames = [(PARAM_ID, data) for data in payloads.values()]
        frames.extend(continuing.values())
        expected = dict.fromkeys(payloads, kind)
        expected.update(dict.fromkeys(continuing, Read.STATUS))
        replies = self.bus.exchange(frames, expected, self.checkpoint, self.log)
        self.check_enabled_feedback(continuing)
        return replies

    def write_modes(self, value, nodes):
        continuing = {n: target for n, target in self.targets.items() if n not in nodes}
        writes = {n: bytes((n, 0, 0x55, MODE_REGISTER, value, 0, 0, 0)) for n in nodes}
        self.coherent = False
        replies = self.exchange_mode_register(writes, Read.MODE_WRITE, continuing)
        for node, data in writes.items():
            if replies[node, Read.MODE_WRITE] != data:
                raise ZlgError(f"ID {node} 模式写入应答不符")
        self.exchange_mode_register({n: Read.MODE.payload(n) for n in nodes}, Read.MODE, continuing)
        for node in nodes:
            actual = self.bus.motors[node].mode
            if actual != value:
                raise ZlgError(f"ID {node} 模式回读为 {actual}，要求 {value}")
        self.coherent = True

    def start_cycles(self):
        if not self.owned or self.uncertain or not self.coherent:
            raise ZlgError("周期发送需要已确认使能的目标轴")
        self.targets.clear()
        for node in sorted(self.owned):
            fb = self.status(node)
            if fb.status != 1:
                raise ZlgError(f"ID {node} 状态为 {fb.status_text}")
            self.targets[node] = self.target(node, fb)
            self.commands[node] = 0.0 if self.bus.motors[node].mode == 1 else fb.position
        if self.queued_cycles:
            self.bus.backend.enable_transmit_queue()
        self.expected_feedback = dict.fromkeys(self.targets, Read.STATUS)
        # 毫秒级发送调度和频率统计共用高分辨率单调时钟。
        self.next_cycle = perf_counter()
        self.run_started_at = self.next_cycle
        self.hz = 0.0
        self.run_cycles = 0
        self.streaming = True
        mode = "队列发送" if self.queued_cycles else "直接批发"
        self.report(f"周期发送 {sorted(self.targets)} · {mode}")

    def cycle(self):
        ctl = self.controller
        motors = self.bus.motors
        if ctl and ctl.gravity_on and any(motors[n].mode == 1 for n in JOINTS):
            torques = ctl.gravity_commands(motors, monotonic())
            for node in JOINTS:
                motor = motors[node]
                if motor.mode == 1:
                    torque = max(-motor.max_torque, min(motor.max_torque, torques[node - 1]))
                    self.targets[node] = motor.command_id, motor.encode_mit(0, 0, torque, 0, 0)
                    self.commands[node] = torque
        self.bus.exchange(self.targets.values(), self.expected_feedback,
                          self.checkpoint, queued=self.queued_cycles)
        self.check_enabled_feedback(self.targets)
        if ctl and ctl.home_targets:
            if all(abs(self.bus.motors[n].feedback.position - pos) <= ctl.arrival_rad
                   and abs(self.bus.motors[n].feedback.velocity) <= self.stop_speed
                   for n, pos in ctl.home_targets.items()):
                ctl.home_targets.clear()
                self.report("六轴已到预设位，继续 PV 持位")
            elif monotonic() >= ctl.home_deadline:
                self.stop()
                self.report("回预设位超时，已按实测位置停住")
        self.cycles += 1
        self.run_cycles += 1
        now = perf_counter()
        elapsed = now - self.run_started_at
        if elapsed > 0:
            self.hz = self.run_cycles / elapsed
        self.next_cycle = max(now, self.next_cycle + self.period)

    def check_enabled_feedback(self, nodes):
        for node in nodes:
            fb = self.bus.motors[node].feedback
            if fb.status != 1:
                raise ZlgError(f"ID {node} 反馈退出使能：{fb.status_text}")

    def stop(self):
        self.streaming = False
        self.clear_targets()
        self.bus.backend.clear_transmit_queue()
        neutral = {}
        holding = {}
        for node in sorted(self.owned):
            if self.bus.motors[node].mode in (2, 4):
                fb = self.status(node)
                if fb.status != 1:
                    raise ZlgError(f"ID {node} 持位前状态为 {fb.status_text}")
                holding[node] = self.target(node, fb)
            elif self.bus.motors[node].mode == 1:
                motor = self.bus.motors[node]
                neutral[node] = motor.command_id, motor.encode_mit(0, 0, 0, 0, 0)
        if neutral:
            self.bus.exchange(list(neutral.values()), dict.fromkeys(neutral, Read.STATUS), self.checkpoint)
            if any(self.bus.motors[n].feedback.status != 1 for n in neutral):
                raise ZlgError("清零 MIT 力矩时反馈退出使能")
        if holding:
            deadline = monotonic() + self.stop_timeout
            settled = 0
            while settled < 3:
                if monotonic() >= deadline:
                    raise ZlgError("限速持位后速度未进入停止范围")
                self.bus.exchange(list(holding.values()), dict.fromkeys(holding, Read.STATUS),
                                  self.checkpoint)
                feedback = [self.bus.motors[n].feedback for n in holding]
                if any(f.status != 1 for f in feedback):
                    raise ZlgError("持位反馈退出使能")
                settled = settled + 1 if all(abs(f.velocity) <= self.stop_speed for f in feedback) else 0
                with self.cv:
                    self.cv.wait(self.stop_sample_period)
            self.log(f"持位轴 {sorted(holding)} 连续三次速度达标，末次 rad/s：" +
                     "、".join(f"{f.velocity:.4f}" for f in feedback))
        self.report("周期已停止，发送队列已清空")

    def change_running_mode(self, nodes, value):
        """在使能状态下交接，PV 锁存原始电机角，MIT 首帧沿用当前补偿。"""
        if not nodes:
            return
        if self.streaming:
            self.cycle()
        else:
            for n in nodes:
                self.status(n)
            self.check_enabled_feedback(nodes)
        feedback = {n: self.bus.motors[n].feedback for n in nodes}
        self.write_modes(value, nodes)
        targets = {n: self.target(n, fb) for n, fb in feedback.items()}
        if self.streaming:
            self.targets.update(targets)
            for n, fb in feedback.items():
                self.commands[n] = fb.position if value == 2 else 0.0
            # 立即发送完整的新模式首轮；cycle 在投递前填入已开启的 MIT 重力矩。
            self.next_cycle = perf_counter()
            self.cycle()
        else:
            self.bus.exchange(targets.values(), dict.fromkeys(nodes, Read.STATUS), self.checkpoint, self.log)
            self.check_enabled_feedback(nodes)
        angle_change, node = max((abs(self.bus.motors[n].feedback.position - fb.position), n)
                                 for n, fb in feedback.items())
        self.log(f"换模前末帧至新模式首帧，最大原始角差 J{node} {degrees(angle_change):.3f}°")

    def set_pv_target(self, node, position, speed):
        motor = self.bus.motors[node]
        self.targets[node] = motor.command_id, motor.encode_pv(position, speed)
        self.commands[node] = position

    def local_control(self, action, args):
        ctl = self.controller
        motors = self.bus.motors
        if self.uncertain or self.faulted or not self.coherent:
            raise ZlgError("当前通道状态禁止本地控制")
        if action in ("lock", "unlock"):
            node, = args
            previous = 1 if action == "lock" else 2
            if node not in JOINTS or node not in self.owned or motors[node].mode != previous or ctl.home_targets:
                raise ZlgError("当前关节状态不允许此锁定操作")
            self.change_running_mode((node,), 2 if action == "lock" else 1)
            self.report(f"J{node} 已按实测位置锁定" if action == "lock" else f"J{node} 已释放到 MIT")
            return
        if not self.streaming or not set(JOINTS) <= self.owned:
            raise ZlgError("需要六关节均已使能并周期反馈")
        if action in ("gravity_on", "gravity_off"):
            ctl.gravity_on = action == "gravity_on"
            if ctl.gravity_on:
                ctl.gravity_start = monotonic()
            else:
                for n in JOINTS:
                    if motors[n].mode == 1:
                        self.targets[n] = self.target(n, motors[n].feedback)
                        self.commands[n] = 0.0
            self.report("MIT 重力补偿已开启" if ctl.gravity_on else "MIT 重力补偿已关闭")
            return
        # 六轴全部进入 PV 并回读后，才发布同一批预设目标。
        self.change_running_mode(tuple(n for n in JOINTS if motors[n].mode != 2), 2)
        positions = ctl.motor_home(motors)
        travel = max(abs(motors[n].feedback.position - pos) for n, pos in positions.items())
        ctl.home_deadline = monotonic() + travel / ctl.home_speed + self.stop_timeout
        for n, pos in positions.items():
            self.set_pv_target(n, pos, ctl.home_speed)
        ctl.home_targets = positions
        self.report(f"六轴回预设位，PV 限速 {ctl.home_speed:g} rad/s")

    def set_zero(self, node):
        if (self.owned != set(ALL) or self.streaming or self.faulted or self.uncertain
                or not self.coherent or node not in JOINTS):
            self.report("零位设定需要整臂已使能、周期发送已停止，并选择 J1～J6")
            return
        self.mode(node)
        fb = self.status(node)
        if fb.status != 1:
            raise ZlgError(f"J{node} 设零前回读 {fb.status_text}")
        if abs(fb.velocity) > self.stop_speed:
            self.report(f"J{node} 仍在运动，请停在机械参考位后设零")
            return
        self.special(node, 0xFE)
        fb = self.status(node)
        if fb.status != 1:
            raise ZlgError(f"J{node} 设零后回读 {fb.status_text}")
        if self.bus.motors[node].mode == 2:
            # 零点已改变，PV 改用新坐标下的当前位置持位。
            self.bus.exchange([self.target(node, fb)], {node: Read.STATUS}, self.checkpoint, self.log)
            fb = self.bus.motors[node].feedback
            if fb.status != 1:
                raise ZlgError(f"J{node} 设零后持位回读 {fb.status_text}")
        self.report(f"J{node} 设零已应答，回读位置 {fb.position:.4f} rad，{fb.status_text}")

    def dispatch(self, action, nodes):
        if action in (*PRIORITY, "pose_off"):
            self.tracking = False
        if action in ("close", "release"):
            if action == "close" and (self.owned or self.uncertain):
                try:
                    self.stop()
                except (ZlgError, OSError) as error:
                    self.log(f"停发确认失败：{error}")
                if not self.disable(self.owned | self.uncertain):
                    return
            self.physically_stopped = action == "release"
            try:
                self.bus.close()
            except (ZlgError, OSError) as error:
                self.faulted = True
                self.report(f"关闭设备失败：{error}")
                self.done = self.bus.backend is None
                return
            self.done = True
            self.report("现场已物理停机，设备已释放" if self.physically_stopped else "CAN FD 设备已释放")
            return
        if self.listen_only or not self.bus.started:
            raise ZlgError("控制操作需要已启动且可发送的通道")
        if action in ("disable", "disable_all"):
            if action == "disable_all":
                nodes = ALL
            if self.owned.difference(nodes):
                self.stop()
            self.disable(set(nodes))
            return
        if action == "stop":
            self.stop()
            return
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
            for node in ALL:
                try:
                    if node != 1:
                        self.mode(node)
                    if self.status(node).status == 0:
                        self.verified.add(node)
                except (ZlgError, OSError) as error:
                    missing.append(str(node))
                    self.log(str(error))
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

    def run(self):
        try:
            self.report(self.bus.open(self.listen_only))
            with self.cv:
                self.current = None
                if not self.listen_only and not self.pending:
                    self.pending = ("read", ())
                self.publish()
            self.loop()
        except (ZlgError, OSError, AttributeError) as error:
            self.report(f"设备错误：{error}")
        finally:
            if self.bus.backend is not None:
                try:
                    if self.owned and not self.physically_stopped:
                        self.disable(self.owned.copy())
                finally:
                    try:
                        self.bus.close()
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
            try:
                if command:
                    self.dispatch(*command)
                elif self.tracking and monotonic() >= self.next_monitor:
                    self.sample_pose()
                elif self.streaming and perf_counter() >= self.next_cycle:
                    self.cycle()
                elif self.owned and not self.uncertain and monotonic() >= self.next_monitor and not self.streaming:
                    for node in sorted(self.owned):
                        if self.status(node, verbose=False).status != 1:
                            raise ZlgError(f"ID {node} 主动状态回读退出使能")
                    self.next_monitor = monotonic() + 0.1
                elif not self.faulted:
                    self.bus.receive(log=None if self.streaming else self.log)
            except OperationInterrupted:
                self.log("普通操作已中断，处理停止请求")
                # 部分使能或换模未核对时先逐轴失能；模式已知时由待处理的停止操作接管。
                switching = self.current in ("lock", "unlock", "home", "mit", "pv")
                if self.owned and (self.current == "enable" or (switching and not self.coherent)):
                    self.disable(self.owned.copy())
            except (ZlgError, OSError, ValueError) as error:
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
                delay = max(0, self.next_cycle - perf_counter()) if self.streaming else 0.005
                if delay > 0 and not self.pending and not self.done:
                    self.cv.wait(delay)
