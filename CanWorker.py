"""CAN FD 独占线程，串行执行操作，向 Qt 发布完整状态快照。"""

from collections import deque
from contextlib import ExitStack
from dataclasses import dataclass, replace
from math import ceil, degrees, isfinite, radians
from statistics import median
import sys
from threading import Condition, Thread
from time import monotonic, perf_counter

from ArmController import HOLD_FIELDS
from DMMotor import MODE_REGISTER, PARAM_ID, Read
from PidTuneSample import TunePoint
from USBCANFD import OperationInterrupted
from zlgcan import ZlgError

JOINTS = tuple(range(1, 7))
ALL = (*JOINTS, 7)
PRIORITY = {"stop": 1, "disable": 2, "disable_all": 3, "close": 4}


def uninterrupted():
    """单轴失能及其回读必须完成后才处理下一项操作。"""


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
class TuneLimits:
    step_deg: tuple
    travel_deg: tuple
    lock_deg: float

    def __post_init__(self):
        if (len(self.step_deg) != 6 or len(self.travel_deg) != 6
                or any(not isfinite(v) or v < 0 for v in self.step_deg)
                or any(not (isfinite(a) and isfinite(b) and a < b) for a, b in self.travel_deg)
                or not isfinite(self.lock_deg) or self.lock_deg <= 0):
            raise ValueError("请检查六轴调参幅度、角度区间及 PV 锁定容差（°）")


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
    hold_state: str
    hold_details: tuple
    joint_sample_at: float
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
    tune_node: int | None
    tune_phase: str
    tune_origin: float | None
    tune_locks: tuple
    tune_parameters: tuple
    tune_revision: int

    @property
    def can_reflect_force(self):
        return (self.opened and self.streaming and self.gravity and self.coherent
                and not self.uncertain and not self.faulted and not self.tune_node
                and all(n in self.owned and mode == "MIT" for n, (mode, _) in zip(JOINTS, self.motors)))


class CanWorker(Thread):
    def __init__(self, bus, listen_only, cycle_delay, speed_limit, pose_period, *, queued_cycles,
                 tool_update_cycles, controller=None, tune_limits=None):
        super().__init__(name="DMArm-CAN-FD")
        if type(tool_update_cycles) is not int or tool_update_cycles < 1:
            raise ValueError("夹钳发送间隔必须为正整数（轮）")
        self.tool_update_cycles = tool_update_cycles
        self.bus = bus
        self.controller = controller
        self.tune_limits = tune_limits
        self.tune_points = deque(maxlen=16384)
        self.tune_point = None
        self.tune_parameters = self.parameter_columns()
        self.tune_revision = 0
        self.clear_tuning()
        self.periods = deque(maxlen=16384)
        self.previous_batch = None
        self.remote_link = None
        self.remote_action = ""
        self.remote_frame = None
        self.clear_force_feedback()
        self.force_output_pending = False
        self.commands = {}
        self.joint_sample = None
        self.last_hold_sample_at = None
        self.last_torque = {}
        self.listen_only = listen_only
        self.tracking = False
        self.pose_period = pose_period
        if not isfinite(cycle_delay) or cycle_delay < 0:
            raise ValueError("CAN FD 轮间等待时间必须为非负有限秒数")
        self.cycle_delay = cycle_delay
        self.queued_cycles = queued_cycles
        self.speed_limit = speed_limit
        self.cv = Condition()
        self.waiting_timer = None
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
        self.done = False
        self.faulted = False
        self.publish()

    def publish(self):
        with self.cv:
            if self.tune_point is not None:
                self.tune_points.append(self.tune_point)
                self.tune_point = None
            self.state = Snapshot(
                self.bus.started, bool(self.current or self.pending), self.first_read,
                frozenset(self.verified), frozenset(self.owned), frozenset(self.uncertain),
                self.streaming, bool(self.controller and self.controller.gravity_on),
                self.force_link is not None,
                self.controller.hold_state if self.controller else "空闲",
                self.controller.details if self.controller else (),
                self.joint_sample.oldest_at if self.joint_sample else 0.0,
                tuple(self.commands.get(n) for n in JOINTS), self.tracking, self.coherent, self.faulted,
                tuple((m.mode_text, m.feedback) for m in self.bus.motors.values()),
                self.message, self.bus.tx, self.bus.rx, self.bus.failed, self.cycles,
                self.hz if self.streaming else 0.0,
                min(m.feedback_at for m in self.bus.motors.values()), self.remote_action,
                self.tune_node, self.tune_phase, self.tune_origin, self.tune_locks,
                self.tune_parameters, self.tune_revision,
            )

    def snapshot(self):
        with self.cv:
            messages = tuple(self.events)
            self.events.clear()
            points = self.tune_points
            if points:
                # 从生产者摘下整批，界面消化积压样本时工作线程可立即追加下一拍。
                self.tune_points = deque(maxlen=points.maxlen)
            else:
                points = ()
            state = replace(self.state, busy=bool(self.current or self.pending))
        return state, messages, tuple(points)

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
            self.pending = (action, (self.joint_sample,) if action == "hold_on" else tuple(nodes))
            if self.waiting_timer is None:
                self.cv.notify()
            else:
                self.waiting_timer.wake()
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
                          check or self.checkpoint, self.log, disabling=(code == 0xFD))

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
        self.finish_hold()
        self.clear_tuning()
        if self.uncertain:
            self.report(f"失能未确认 {sorted(self.uncertain)}；请支撑并物理停机")
            return False
        self.force_output_pending = False
        if clearing_error:
            self.report(f"失能已回读；清发送队列失败：{clearing_error}")
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
        self.clear_force_feedback()
        self.invalidate_sample()
        # 寄存器写入与回读是额外批次；分频发送的夹钳只在到期的控制轮中加入。
        continuing = {n: target for n, target in self.targets.items()
                      if self.streaming and n not in nodes and (n != 7 or self.tool_update_cycles == 1)}
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
        if self.queued_cycles:
            self.bus.backend.enable_transmit_queue()
        self.expected_feedback = dict.fromkeys(sorted(self.targets), Read.STATUS)
        self.joint_feedback = {n: Read.STATUS for n in JOINTS if n in self.targets}
        # 毫秒级发送调度和频率统计共用高分辨率单调时钟。
        self.next_cycle = perf_counter()
        self.rate_started_at = self.next_cycle
        self.rate_start_cycles = self.cycles
        self.hz = 0.0
        self.run_cycles = 0
        self.streaming = True
        mode = "队列发送" if self.queued_cycles else "直接批发"
        self.report(f"周期发送 {sorted(self.targets)} · {mode}")

    def invalidate_sample(self):
        with self.cv:
            self.joint_sample = None
        self.last_torque.clear()

    def parameter_columns(self):
        return tuple(tuple(getattr(self.controller.hold, name)) for name in HOLD_FIELDS) if self.controller else ()

    def clear_tuning(self):
        self.tune_node = None
        self.tune_phase = ""
        self.tune_origin = None
        self.tune_locks = ()
        self.tune_draft = None
        self.previous_parameters = None
        self.tune_event = ""
        self.confirmed_target = None
        self.exit_reason = ""

    def start_tuning(self, node):
        sample = self.joint_sample
        if node not in JOINTS or not self.tune_limits:
            self.report("未配置此轴的调参范围")
            return
        if not (self.streaming and self.queued_cycles and self.bus.started and not self.listen_only
                and self.coherent and not self.faulted and not self.uncertain and set(JOINTS) <= self.owned
                and all(self.bus.motors[n].mode in (1, 2) for n in JOINTS)):
            self.report("调参需要已核对 MIT/PV 模式的六轴使能反馈及 CAN FD 队列发送")
            return
        if self.remote_action or self.controller.hold_state != "空闲" or self.tune_node:
            self.report("请先结束当前调参、整臂持位或遥操作")
            return
        if (not sample or perf_counter() - sample.oldest_at > self.controller.hold.max_dt
                or any(f.status != 1 for f in sample.feedback)):
            self.report("六轴完整使能反馈已过期或缺失")
            return
        self.tune_node = node
        self.gravity_before_tuning = self.controller.gravity_on
        self.tune_phase = "等待 PV 锁定"
        self.tune_locks = tuple(f.position for f in sample.feedback)
        switching = tuple(n for n in JOINTS if self.bus.motors[n].mode != 2)
        self.set_positions({n: self.tune_locks[n-1] for n in JOINTS if n not in switching})
        if switching:
            self.write_modes(2, switching)
            self.set_positions({n: self.tune_locks[n-1] for n in switching})
        self.report(f"准备 J{node} 调参，等待六轴 PV 首批应答")

    def end_tuning(self, reason):
        """另五轴沿用原锁存目标，调参轴在新鲜实测角就地锁定。"""
        node = self.tune_node
        self.hold_checkpoint()
        position = self.joint_sample.feedback[node-1].position
        self.finish_hold()
        self.tune_phase = "锁回 PV"
        self.exit_reason = reason
        if self.bus.motors[node].mode != 2:
            self.write_modes(2, (node,))
        locks = list(self.tune_locks)
        locks[node-1] = position
        self.tune_locks = tuple(locks)
        self.set_positions({node: position})
        self.report(f"{reason}，等待六轴 PV 控制应答")

    def tune_command(self, action, args):
        node, *values = args
        if action == "tune_start":
            self.start_tuning(node)
            return
        if node != self.tune_node:
            self.report(f"J{node} 当前未在调参")
            return
        if action == "tune_end":
            if self.tune_phase != "锁回 PV":
                if self.joint_sample is not None:
                    self.end_tuning("结束本轴调参")
                else:
                    self.exit_reason = "结束本轴调参"
            return
        if self.tune_phase != "调参中":
            self.report("请等候调参轴就绪")
            return
        if action == "tune_apply":
            self.tune_draft, = values
            return
        delta, = values
        idx = node - 1
        target = self.tune_origin + radians(delta)
        lower, upper = self.tune_limits.travel_deg[idx]
        span = degrees(self.bus.motors[node].max_position)
        lower, upper = max(-span, lower), min(span, upper)
        step = self.tune_limits.step_deg[idx]
        if not (abs(delta) <= step and radians(lower) <= target <= radians(upper)):
            self.report(f"J{node} 目标 {degrees(target):.3f}° 越界，允许 [{lower:g}, {upper:g}]°，"
                        f"阶跃幅度 ±{step:g}°")
            return
        self.controller.set_axis_reference(target)

    def advance_tuning(self, sent_at, holding):
        """仅在当前六轴或七节点控制批次完整确认后推进交接及记录。"""
        node = self.tune_node
        idx = node - 1
        sample = self.joint_sample
        ctl = self.controller
        if self.tune_phase == "锁回 PV":
            ctl.gravity_on = self.gravity_before_tuning
            self.report(f"{self.exit_reason}，六轴 PV 已确认")
            self.clear_tuning()
            return
        errors = [(abs(sample.feedback[n-1].position - self.tune_locks[n-1]), n) for n in JOINTS if n != node]
        worst, lock_node = max(errors)
        if holding:
            fb = sample.feedback[idx]
            a = ctl.details[idx]
            event = self.tune_event
            if a.target != self.confirmed_target:
                event = " / ".join(filter(None, ("阶跃" if a.target != self.tune_origin else "回原位", event)))
            if self.tune_event == "应用参数":
                self.previous_parameters = None
                self.tune_parameters = self.parameter_columns()
                self.tune_revision += 1
                self.report(f"J{node} 本轴参数已应用")
            self.tune_point = TunePoint(node, sent_at, sample.times[idx], a.target, fb.position, a.velocity_ref,
                                        fb.velocity, a.gravity, a.pid_raw, a.pid, self.last_torque[node], fb.torque,
                                        a.pid_limited, a.output_limited, a.rate_limited, lock_node, worst,
                                        event, self.confirmed_target)
            self.confirmed_target = a.target
            self.tune_event = ""
        if self.exit_reason:
            self.end_tuning(self.exit_reason)
            return
        if degrees(worst) > self.tune_limits.lock_deg:
            if self.tune_phase == "等待 PV 锁定":
                waiting = "、".join(f"J{n}" for e, n in errors if degrees(e) > self.tune_limits.lock_deg)
                self.message = f"等待 {waiting} 就位，最大偏差 {degrees(worst):.3f}°"
            else:
                self.end_tuning(f"J{lock_node} 锁定偏差 {degrees(worst):.3f}° 超限")
            return
        if self.tune_phase == "等待 PV 锁定":
            self.write_modes(1, (node,))
            motor = self.bus.motors[node]
            gravity = ctl.gravity_commands(sample.feedback)[idx]
            torque = max(-motor.output_limit, min(motor.output_limit, gravity))
            self.targets[node] = motor.command_id, motor.encode_mit(0, 0, torque, 0, 0)
            ctl.gravity_on = True
            self.tune_phase = "等待 MIT 首批"
        elif self.tune_phase == "等待 MIT 首批":
            self.tune_origin = self.confirmed_target = sample.feedback[idx].position
            ctl.begin_hold([self.tune_origin], sample.feedback, [self.last_torque[node]], slice(idx, node))
            self.reset_hold_stats()
            self.tune_phase = "调参中"
            self.tune_event = "原位"
            self.report(f"J{node} 已锁存原位 {degrees(self.tune_origin):.3f}° · 队列发送")

    def begin_hold(self, clicked):
        ctl = self.controller
        if (not self.bus.started or self.listen_only or not self.streaming or not self.coherent
                or self.uncertain or self.faulted or self.remote_action or not set(JOINTS) <= self.owned
                or any(self.bus.motors[n].mode != 1 for n in JOINTS)):
            self.report("持位需要可发送通道及六轴 MIT 使能周期，且已结束遥操作")
            return
        sample = self.joint_sample
        if (clicked is None or sample is None or perf_counter() - clicked.oldest_at > ctl.hold.max_dt
                or any(f.status != 1 for f in clicked.feedback + sample.feedback)):
            self.report("点击时的六轴完整使能反馈缺失或已过期，未开启持位")
            return
        ctl.begin_hold([f.position for f in clicked.feedback], sample.feedback,
                       [self.last_torque[n] for n in JOINTS])
        self.reset_hold_stats()
        self.report("已锁存六轴原始电机角，双环持位已开启")

    def reset_hold_stats(self):
        self.last_hold_sample_at = None
        self.pid_peaks = [0.0] * 6
        self.command_peaks = [0.0] * 6
        self.max_hold_dt = 0.0
        self.periods.clear()
        self.previous_batch = None
        self.reported_batches = set()

    def finish_hold(self):
        if self.controller and self.controller.hold_state != "空闲":
            if self.previous_parameters is not None:
                ctl = self.controller
                ctl.hold = self.previous_parameters
                i = self.tune_node - 1
                ctl.speed_limit[i] = min(ctl.hold.speed_limit[i], ctl.motor_speeds[i])
                self.previous_parameters = None
            def peaks(values):
                return "、".join(f"{v:.3f}" for v in values)
            self.log(f"持位汇总 J1～J6：PID 纠偏绝对峰值 [{peaks(self.pid_peaks)}] N·m，"
                     f"下发总力矩绝对峰值 [{peaks(self.command_peaks)}] N·m，"
                     f"最大采样间隔 {self.max_hold_dt * 1000:.2f} ms")
            self.controller.end_hold()
            self.last_hold_sample_at = None
            for tool, label in ((False, "六帧"), (True, "含夹钳")):
                times = sorted(dt * 1000 for dt, _, has_tool in self.periods if has_tool == tool)
                if times:
                    exchange_mean = sum(t for _, t, has_tool in self.periods if has_tool == tool) * 1000 / len(times)
                    gap_mean = sum(times) / len(times) - exchange_mean
                    self.log(f"{label}周期 {len(times)} 轮，中位 {median(times):.3f} / "
                             f"P95 {times[ceil(len(times)*0.95)-1]:.3f} / 最大 {times[-1]:.3f} ms，"
                             f"平均整批收发 {exchange_mean:.3f} / 收齐至下批开始 {gap_mean:.3f} ms")
            self.previous_batch = None

    def hold_checkpoint(self):
        self.checkpoint()
        if perf_counter() - self.joint_sample.oldest_at > self.controller.hold.max_dt:
            raise ZlgError("双环持位六轴反馈超时")

    def prepare_hold(self):
        self.hold_checkpoint()
        ctl = self.controller
        sample = self.joint_sample
        if self.tune_draft is not None:
            node = self.tune_node
            previous = ctl.hold
            try:
                ctl.apply_axis_parameters(node-1, self.tune_draft, sample.feedback, self.last_torque[node])
            except ValueError as error:
                self.report(f"J{node} 参数未应用：{error}")
            else:
                self.previous_parameters = previous
                self.last_hold_sample_at = None
                self.tune_event = "应用参数"
            self.tune_draft = None
        dt = 0.0
        if self.last_hold_sample_at is not None:
            dt = sample.sampled_at - self.last_hold_sample_at
            if not 0 < dt <= ctl.hold.max_dt:
                raise ZlgError(f"持位采样间隔异常：{dt * 1000:.2f} ms")
        commands, goals = ctl.step_hold(sample.feedback, dt, self.force_torque)
        self.last_hold_sample_at = sample.sampled_at
        matched = True
        for n, torque, goal in zip(JOINTS[ctl.axes], commands, goals):
            motor = self.bus.motors[n]
            data = motor.encode_mit(0, 0, torque, 0, 0)
            self.targets[n] = motor.command_id, data
            if ctl.hold_state == "退出中" and motor.payload_torque_code(data) != motor.torque_code(goal):
                matched = False
        return ctl.hold_state == "退出中" and matched

    def confirm_batch(self, expected):
        self.check_enabled_feedback(expected)
        for n in self.joint_feedback:
            motor = self.bus.motors[n]
            if motor.mode == 1:
                code = motor.payload_torque_code(self.targets[n][1])
                self.last_torque[n] = self.commands[n] = motor.torque_value(code)
        if len(self.joint_feedback) == len(JOINTS):
            sample = JointSample(tuple(self.bus.motors[n].feedback for n in JOINTS),
                                 tuple(self.bus.motors[n].feedback_at for n in JOINTS))
            with self.cv:
                self.joint_sample = sample

    def cycle(self):
        ctl = self.controller
        motors = self.bus.motors
        if self.remote_action:
            self.update_remote()
        if self.force_link is not None:
            self.update_force_feedback()
        holding = ctl and ctl.hold_state != "空闲"
        released = False
        if holding:
            released = self.prepare_hold()
        elif ctl and ctl.gravity_on and not self.tune_node and any(motors[n].mode == 1 for n in JOINTS):
            feedback = tuple(motors[n].feedback for n in JOINTS)
            torques = ctl.gravity_commands(feedback)
            if self.force_torque is not None:
                torques += self.force_torque
            for node in JOINTS:
                motor = motors[node]
                if motor.mode == 1:
                    torque = max(-motor.output_limit, min(motor.output_limit, torques[node - 1]))
                    self.targets[node] = motor.command_id, motor.encode_mit(0, 0, torque, 0, 0)
        expected = self.joint_feedback
        if (self.run_cycles + 1) % self.tool_update_cycles == 0:
            expected = self.expected_feedback
        if expected:
            frames = [self.targets[n] for n in expected]
            check = self.hold_checkpoint if holding else self.checkpoint
            sent_at = perf_counter()
            # 收齐无反馈分量的新指令应答前，停发仍需撤去可能已经下发的反馈力矩。
            if self.force_torque is not None:
                self.force_output_pending = True
            self.bus.exchange(frames, expected, check, queued=self.queued_cycles)
            if holding:
                exchanged_at = perf_counter()
                check()
            self.confirm_batch(expected)
            self.force_output_pending = self.force_torque is not None
            if holding:
                has_tool = 7 in expected
                if has_tool not in self.reported_batches:
                    ids = "、".join(f"0x{can_id:03X}" for can_id, _ in frames)
                    method = "队列发送" if self.queued_cycles else "直接批发"
                    self.log(f"{method}控制批次 [{ids}]，已确认节点 {list(expected)}")
                    self.reported_batches.add(has_tool)
                if ctl.hold_state == "持位" and not self.tune_event:
                    if self.previous_batch is not None:
                        previous_at, previous_exchange, had_tool = self.previous_batch
                        self.periods.append((sent_at - previous_at, previous_exchange, had_tool))
                    self.previous_batch = sent_at, exchanged_at - sent_at, has_tool
                else:
                    self.previous_batch = None
                self.max_hold_dt = max(self.max_hold_dt, self.joint_sample.sampled_at - self.last_hold_sample_at)
                for n in JOINTS[ctl.axes]:
                    i = n - 1
                    axis = ctl.details[i]
                    self.pid_peaks[i] = max(self.pid_peaks[i], abs(axis.pid))
                    self.command_peaks[i] = max(self.command_peaks[i], abs(self.last_torque[i + 1]))
                if released:
                    self.finish_hold()
                    self.report("已退出持位，继续重力补偿" + ("及力反馈" if self.force_link else ""))
            if self.tune_node:
                self.advance_tuning(sent_at, holding)
        self.cycles += 1
        self.run_cycles += 1
        now = perf_counter()
        # 用最近一个完整统计窗口的收发轮数显示 Hz，切换补偿前的轮数只影响所在窗口。
        elapsed = now - self.rate_started_at
        if elapsed >= 1.0:
            self.hz = (self.cycles - self.rate_start_cycles) / elapsed
            self.rate_started_at = now
            self.rate_start_cycles = self.cycles
        self.next_cycle = now + self.cycle_delay

    def check_enabled_feedback(self, nodes):
        for node in nodes:
            fb = self.bus.motors[node].feedback
            if fb.status != 1:
                raise ZlgError(f"ID {node} 反馈退出使能：{fb.status_text}")

    def stop(self):
        self.clear_force_feedback()
        if self.force_output_pending and self.controller.hold_state == "空闲":
            self.cycle()
        if self.remote_action:
            self.hold_remote("停止发送")
            self.cycle()
        self.bus.backend.clear_transmit_queue()
        if self.tune_node or (self.controller and self.controller.hold_state != "空闲"):
            for node in JOINTS:
                if self.bus.motors[node].mode == 1:
                    self.clear_motor_command(node)
            self.bus.exchange([self.targets[n] for n in JOINTS], self.joint_feedback, self.checkpoint)
            self.confirm_batch(self.joint_feedback)
            self.force_output_pending = False
            self.finish_hold()
            self.clear_tuning()
            self.controller.gravity_on = False
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
        self.force_link = self.force_frame = self.force_torque = None

    def start_force_feedback(self, link):
        frame = link.fresh() if link else None
        if not self.state.can_reflect_force or not frame or not frame.has_net_torque:
            self.report("力反馈需要六轴 MIT 使能周期、重力补偿及有效的从端净驱动力矩")
            return
        self.force_link = link
        self.force_frame = frame
        self.force_torque = self.controller.reflected_torque(frame.net_torque)
        self.report("力反馈已开启")

    def update_force_feedback(self):
        frame = self.force_link.fresh()
        if frame is self.force_frame:
            return
        if not frame or frame.generation != self.force_frame.generation or not frame.has_net_torque:
            self.clear_force_feedback()
            self.report("从端断流或净驱动力矩全部无效，力反馈已关闭")
            return
        self.force_frame = frame
        self.force_torque = self.controller.reflected_torque(frame.net_torque)

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
        if action in ("tune_start", "tune_end", "tune_step", "tune_apply"):
            self.tune_command(action, nodes)
            return
        if self.tune_node and action not in PRIORITY:
            self.report(f"J{self.tune_node} 正在调参，请先结束并锁回 PV")
            return
        if action == "close":
            try:
                self.bus.close()
            except (ZlgError, OSError) as error:
                self.faulted = True
                self.report(f"关闭设备失败：{error}")
                self.done = self.bus.backend is None
                return
            self.finish_hold()
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
        if action == "hold_on":
            if self.controller.hold_state == "空闲":
                self.begin_hold(*nodes)
            else:
                self.report("双环持位已开启，请先退出持位")
            return
        if action == "hold_off":
            if self.controller.hold_state == "持位":
                self.controller.begin_release()
                self.report("正在退出持位，向重力补偿" + ("及力反馈" if self.force_link else "") + "过渡")
            return
        if (self.controller and self.controller.hold_state != "空闲" and
                action in ("gravity_on", "gravity_off", "mit", "pv", "lock", "unlock", "home", "sync", "follow")):
            self.report("请先退出双环持位，再切换控制方式")
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

    def run(self):
        try:
            self.report(self.bus.open(self.listen_only))
            with self.cv:
                self.current = None
                if not self.listen_only and not self.pending:
                    self.pending = ("read", ())
                self.publish()
            with ExitStack() as resources:
                timer = None
                if sys.platform == "win32" and not self.listen_only and self.cycle_delay > 0:
                    from WinCycleTimer import WinCycleTimer
                    timer = WinCycleTimer()
                    resources.callback(timer.close)
                self.loop(timer)
        except (ZlgError, OSError, AttributeError) as error:
            self.report(f"设备错误：{error}")
        finally:
            if self.bus.backend is not None:
                try:
                    if self.owned and not self.done:
                        self.disable(self.owned.copy())
                finally:
                    try:
                        self.bus.close()
                    except (ZlgError, OSError) as error:
                        self.report(str(error))
            self.done = True
            self.current = None
            self.publish()

    def loop(self, timer):
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
                elif not self.faulted:
                    self.bus.receive(log=None if self.streaming else self.log)
            except OperationInterrupted:
                self.log("普通操作已中断，处理停止请求")
                # 部分使能或换模未核对时先逐轴失能；模式已知时由待处理的停止操作接管。
                switching = self.current in ("lock", "unlock", "home", "mit", "pv", "sync", "follow")
                if self.owned and (self.current == "enable" or ((switching or self.tune_node) and not self.coherent)):
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
                delay = self.next_cycle - perf_counter() if self.streaming else 0.005
                if delay <= 0 or self.pending or self.done:
                    continue
                if not self.streaming or timer is None:
                    self.cv.wait(delay)
                    continue
                # 在锁内登记等待者，释放锁后即使尚未进入内核等待也能收到命令事件。
                self.waiting_timer = timer
            try:
                timer.wait_until(self.next_cycle)
            finally:
                with self.cv:
                    self.waiting_timer = None
