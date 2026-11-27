"""CAN FD 收发进程。控制端最多投递一份待确认指令，收发端自行重复最新有效批次。"""

from contextlib import ExitStack
from math import isfinite
from multiprocessing import get_context
from multiprocessing.connection import wait
from struct import Struct
import sys
from time import perf_counter, sleep
from typing import NamedTuple

from DMMotor import Feedback, Read
from USBCANFD import DisableResult, OperationInterrupted
from zlgcan import ZlgError


class CycleCommand(NamedTuple):
    sequence: int
    sampled_at: float
    expires_at: float
    submitted_at: float
    targets: tuple  # 节点、已确认模式和已编码的 CAN FD 帧


class CanProcessLost(RuntimeError):
    """收发进程的通信管道已经关闭。"""


# 一份快照同时携带指令序号、真实整轮计数、收发计数以及七个节点的模式和反馈。
HEADER = Struct("<QQdQQQ?")
MOTOR = Struct("<B?4d3B")
IMAGE_SIZE = HEADER.size + 7 * MOTOR.size


def motor_records(image):
    for i in range(7):
        mode, valid, stamp, *values = MOTOR.unpack_from(image, HEADER.size + i * MOTOR.size)
        yield mode or None, Feedback(*values) if valid else None, stamp


class LatestFeedback:
    def __init__(self, context):
        self.data = context.RawArray("c", IMAGE_SIZE)
        self.lock = context.Lock()

    def publish(self, image):
        # 读端复制上一批时跳过此次覆盖，CAN 立即继续下一轮。
        if self.lock.acquire(False):
            try:
                self.data.raw = image
            finally:
                self.lock.release()

    def read(self):
        if not self.lock.acquire(False):
            return None
        try:
            return self.data.raw
        finally:
            self.lock.release()


class CanSession:
    def __init__(self, bus, connection, epoch, latest, delay, tool_interval, max_age):
        self.bus = bus
        self.connection = connection
        self.epoch = epoch
        self.latest = latest
        self.delay = delay
        self.tool_interval = tool_interval
        self.max_age = max_age
        self.guard_until = None
        self.sequence = self.cycles = self.rounds = 0
        self.hz = self.next_send = 0.0
        self.active = self.trip = None
        self.owned = set()
        self.streaming = False
        self.coherent = True

    def image(self):
        bus = self.bus
        parts = [HEADER.pack(self.sequence, self.cycles, self.hz, bus.tx, bus.rx, bus.failed, bus.started)]
        for motor in bus.motors.values():
            fb = motor.feedback
            parts.append(MOTOR.pack(motor.mode or 0, fb is not None, motor.feedback_at,
                                    *(fb if fb is not None else (0., 0., 0., 0, 0, 0))))
        return b"".join(parts)

    def reply(self, status, value, logs=(), image=None):
        self.connection.send((status, value, image if image is not None else self.image(), logs))

    def check_epoch(self):
        if self.request_epoch != self.epoch.value:
            raise OperationInterrupted()

    def check_cycle(self, phase="总线收发期间"):
        self.check_epoch()
        if self.guard_until is None:
            return
        now = perf_counter()
        if now >= self.guard_until:
            command = self.active
            if command is None:
                raise ZlgError(f"控制指令交接超时，停止周期发送（{phase}）")
            raise ZlgError(
                f"控制指令已过期，停止周期发送（{phase}，指令 {command.sequence}，"
                f"反馈至提交 {(command.submitted_at - command.sampled_at) * 1000:.1f} ms，"
                f"提交至接管 {(self.accepted_at - command.submitted_at) * 1000:.1f} ms，"
                f"接管至检查 {(now - self.accepted_at) * 1000:.1f} ms，"
                f"有效期 {(self.guard_until - command.sampled_at) * 1000:.1f} ms，"
                f"接管后完成 {self.rounds - self.accepted_round} 轮）")

    def accept(self, command):
        self.check_epoch()
        if not self.streaming or command.sequence <= self.sequence:
            raise ZlgError("周期未启动或指令序号已失效")
        if tuple(n for n, _, _ in command.targets) != self.nodes:
            raise ZlgError("周期指令未覆盖启动时确认的整批节点")
        for node, mode, (can_id, _) in command.targets:
            motor = self.bus.motors[node]
            if motor.mode != mode or motor.command_id != can_id:
                raise ZlgError(f"ID {node} 指令与已回读模式不符")
        self.active = command
        self.accepted_at = perf_counter()
        self.accepted_round = self.rounds
        self.guard_until = min(command.sampled_at + self.max_age, command.expires_at)
        self.sequence = command.sequence
        self.joint_frames = tuple(frame for node, _, frame in command.targets if node != 7)
        self.all_frames = tuple(frame for _, _, frame in command.targets)
        self.check_cycle("接管新指令时")

    def exchange_cycle(self):
        self.check_cycle("开始本轮收发前")
        with_tool = 7 in self.expected and (not self.joint_frames or (self.rounds + 1) % self.tool_interval == 0)
        frames = self.all_frames if with_tool else self.joint_frames
        expected = self.expected if with_tool else self.joint_expected
        sent = perf_counter()
        self.bus.exchange(frames, expected, self.check_cycle, queued=self.queued)
        finished = perf_counter()
        try:
            self.check_cycle("本轮回包已收齐")
        except ZlgError as error:
            raise ZlgError(f"{error}，本轮 {len(expected)} 个节点收发用时 "
                           f"{(finished - sent) * 1000:.1f} ms") from error
        for node in expected:
            fb = self.bus.motors[node].feedback
            if fb.status != 1:
                raise ZlgError(f"ID {node} 反馈退出使能：{fb.status_text}")
        self.rounds += 1
        self.cycles += 1
        elapsed = finished - self.rate_start
        if elapsed >= 1:
            self.hz = (self.cycles - self.rate_cycles) / elapsed
            self.rate_start, self.rate_cycles = finished, self.cycles
        self.next_send = finished + self.delay
        return with_tool

    def disarm(self, nodes, coherent, log):
        result = self.bus.disable_nodes(nodes, coherent, log)
        self.owned.difference_update(result.nodes - result.uncertain)
        return result

    def stop_on_error(self, error):
        self.active = None
        self.guard_until = None
        self.streaming = False
        logs = []
        result = self.disarm(self.owned.copy(), self.coherent, logs.append)
        self.trip = str(error), result, logs

    def manage(self, op, args, logs):
        bus = self.bus
        if op == "open":
            return bus.open(*args)
        if op == "close":
            bus.close()
            return
        if op == "disable":
            nodes, coherent = args
            self.streaming = False
            self.guard_until = None
            if self.trip is None:
                return self.disarm(nodes, coherent, logs.append)
            # 独立进程已在超龄或总线错误当拍执行失能，直接交回该次回读。
            reason, result, messages = self.trip
            self.trip = None
            logs.extend((reason, *messages))
            remaining = nodes - result.nodes
            if remaining:
                extra = self.disarm(remaining, coherent, logs.append)
                result = DisableResult(result.nodes | extra.nodes, result.verified | extra.verified,
                                       result.uncertain | extra.uncertain, result.queue_error or extra.queue_error)
            return result
        if self.trip is not None:
            raise ZlgError(self.trip[0])
        self.check_epoch()
        if op == "adopt":
            self.owned.update(args[0])
        elif op == "start":
            self.queued, nodes = args
            self.nodes = tuple(sorted(nodes))
            self.expected = dict.fromkeys(self.nodes, Read.STATUS)
            self.joint_expected = {n: Read.STATUS for n in self.nodes if n != 7}
            self.owned.update(nodes)
            if self.queued:
                bus.backend.enable_transmit_queue()
            self.rounds = 0
            self.hz = self.next_send = 0.0
            self.rate_cycles, self.rate_start = self.cycles, perf_counter()
            self.streaming = True
            self.guard_until = self.rate_start + self.max_age
        elif op == "modes":
            self.coherent = False
            bus.write_modes(*args, self.check_cycle, logs.append)
            self.coherent = True
        elif op == "stop":
            bus.backend.clear_transmit_queue()
            self.streaming = False
            self.guard_until = None
        elif op == "receive":
            bus.receive(log=logs.append)
        elif op == "read":
            node, kind, verbose = args
            return bus.read(node, kind, self.check_epoch, logs.append if verbose else None)
        elif op == "exchange":
            frames, expected, verbose = args
            for can_id, payload in frames:
                if payload == b"\xff" * 7 + b"\xfc":
                    self.owned.update(n for n in expected if bus.motors[n].command_id == can_id)
            return bus.exchange(frames, expected, self.check_epoch, logs.append if verbose else None)
        else:
            raise ValueError(f"未知 CAN 操作 {op}")

    def loop(self, timer):
        awaiting = False
        needs_reply = False
        completed = None
        while True:
            if self.active is None and self.guard_until is not None:
                # 首批、换模及中断交接期间也受控，控制端停顿不能让旧输出无限保留。
                remaining = self.guard_until - perf_counter()
                if remaining <= 0:
                    self.stop_on_error("控制指令交接超时，停止周期发送")
                    continue
                if not self.connection.poll(remaining):
                    continue
            if not awaiting and (self.active is None or self.connection.poll()):
                op, epoch, args = self.connection.recv()
                self.request_epoch = epoch
                if op != "cycle":
                    # 模式寄存器、停发及读写操作独占总线，旧模式指令到此作废。
                    self.active = completed = None
                    logs = []
                    try:
                        value = self.manage(op, args, logs)
                    except OperationInterrupted:
                        self.reply("interrupted", None, logs)
                    except (ZlgError, OSError, AttributeError) as error:
                        self.reply("error", str(error), logs)
                    else:
                        self.reply("ok", value, logs)
                    if op == "close":
                        return
                    continue
                awaiting = True
                needs_reply = True
                command, wait_applied = args
            else:
                command = None
            try:
                if self.trip is not None:
                    self.reply("error", self.trip[0])
                    awaiting = False
                    continue
                if command is not None:
                    self.accept(command)
                    if not wait_applied and completed is not None:
                        # 回送上一轮的实际反馈，控制端可在本批收发期间并行准备下批。
                        with_tool, image = completed
                        self.reply("ok", with_tool, image=image)
                        needs_reply = False
                self.check_cycle("周期调度时")
                now = perf_counter()
                if now < self.next_send:
                    # 正轮间隔也能在 1 ms 内响应停发代次变化。
                    until = min(self.next_send, self.guard_until, now + .001)
                    if timer is not None:
                        timer.wait_until(until)
                    else:
                        sleep(max(0., until - now))
                    continue
                with_tool = self.exchange_cycle()
                image = self.image()
                completed = with_tool, image
                self.latest.publish(image)
                if awaiting and needs_reply:
                    self.reply("ok", with_tool, image=image)
            except OperationInterrupted:
                self.active = completed = None
                if awaiting and needs_reply:
                    self.reply("interrupted", None)
            except (ZlgError, OSError) as error:
                completed = None
                self.stop_on_error(error)
                if awaiting and needs_reply:
                    self.reply("error", self.trip[0])
            awaiting = False


def run_can(bus, connection, epoch, latest, delay, tool_interval, max_age):
    """在子进程内加载驱动并持有设备，控制端退出时完成失能回读。"""
    session = CanSession(bus, connection, epoch, latest, delay, tool_interval, max_age)
    try:
        with ExitStack() as resources:
            timer = None
            if sys.platform == "win32" and delay > 0:
                from WinCycleTimer import WinCycleTimer
                timer = WinCycleTimer()
                resources.callback(timer.close)
            session.loop(timer)
    except (EOFError, BrokenPipeError):
        if session.trip is not None:
            reason, _, logs = session.trip
            print(reason, *logs, sep="\n", flush=True)
    finally:
        try:
            if bus.backend is not None:
                try:
                    if session.owned and session.trip is None:
                        session.disarm(session.owned.copy(), session.coherent, lambda message: print(message, flush=True))
                finally:
                    bus.close()
        finally:
            connection.close()


class CanProcessBus:
    def __init__(self, device, cycle_delay, tool_update_cycles, command_max_age):
        if not isfinite(cycle_delay) or cycle_delay < 0 or not isfinite(command_max_age) or command_max_age <= 0:
            raise ValueError("CAN 轮间等待须非负，指令有效期须为正的有限秒数")
        if type(tool_update_cycles) is not int or tool_update_cycles < 1:
            raise ValueError("夹钳发送间隔必须为正整数（轮）")
        self.device = device
        self.motors = device.motors
        self.timeout = device.timeout
        self.cycle_delay, self.tool_update_cycles = cycle_delay, tool_update_cycles
        self.command_max_age = command_max_age
        self.context = get_context("spawn")
        self.latest = LatestFeedback(self.context)
        # 仅界面在命令锁内递增；两个 x64 后端以对齐的 32 位读写传递中断代次。
        self.epoch = self.context.RawValue("I", 0)
        self.generation = self.sequence = self.applied = 0
        self.submitted = self.confirmed = None
        self.started = False
        self.process = None
        self.cycles = self.tx = self.rx = self.failed = 0
        self.hz = 0.0

    def interrupt(self):
        self.epoch.value += 1

    def apply(self, image):
        seq, self.cycles, self.hz, self.tx, self.rx, self.failed, self.started = HEADER.unpack_from(image)
        for motor, record in zip(self.motors.values(), motor_records(image)):
            motor.mode, motor.feedback, motor.feedback_at = record
        return seq

    def latest_motion(self, after_cycle):
        """UDP 发布线程直接读取已确认的新批次，不修改控制线程的电机镜像。"""
        image = self.latest.read()
        if image is None:
            return None
        seq, cycles, hz, tx, rx, failed, _ = HEADER.unpack_from(image)
        if cycles <= after_cycle:
            return None
        for command in (self.submitted, self.confirmed):
            if command is not None and command.sequence == seq and perf_counter() < command.expires_at:
                return command, cycles, hz, tx, rx, failed, tuple(motor_records(image))
        return None

    def refresh_latest(self):
        if not self.applied:
            return False
        image = self.latest.read()
        if image is None:
            return False
        seq, cycles, *_ = HEADER.unpack_from(image)
        if seq not in (self.applied, self.submitted.sequence) or cycles <= self.cycles:
            return False
        self.apply(image)
        self.applied = seq
        if seq == self.submitted.sequence:
            self.confirmed = self.submitted
        return True

    def finish(self):
        self.connection.close()
        self.process.join()
        self.process.close()
        self.process = None
        self.started = False

    def request(self, op, args=(), log=None):
        self.applied = 0
        try:
            self.connection.send((op, self.generation, args))
            # Windows 子进程导入失败时可能尚未接管管道，启动应答同时等进程退出信号。
            if op == "open" and self.connection not in wait((self.connection, self.process.sentinel)):
                raise EOFError("CAN 收发进程在打开设备前退出")
            status, value, image, logs = self.connection.recv()
        except (EOFError, BrokenPipeError, OSError) as error:
            self.finish()
            raise CanProcessLost("CAN 收发进程连接中断，请核对设备状态") from error
        sequence = self.apply(image)
        if log is not None:
            for message in logs:
                log(message)
        if status == "interrupted":
            raise OperationInterrupted()
        if status == "error":
            raise ZlgError(value)
        if op == "cycle":
            self.applied = sequence
        return value

    def open(self, listen_only):
        self.connection, child = self.context.Pipe()
        self.process = self.context.Process(name="DMArm-CAN-FD", target=run_can,
                                            args=(self.device, child, self.epoch, self.latest,
                                                  self.cycle_delay, self.tool_update_cycles, self.command_max_age))
        try:
            self.process.start()
        except (OSError, RuntimeError) as error:
            self.connection.close()
            self.process = None
            raise ZlgError(f"CAN 收发进程启动失败：{error}") from error
        finally:
            child.close()
        try:
            return self.request("open", (listen_only,))
        except (ZlgError, OSError):
            if self.process is not None:
                self.finish()
            raise

    def close(self):
        try:
            self.request("close")
        finally:
            if self.process is not None:
                self.finish()

    def read(self, node, kind, checkpoint, log=None):
        checkpoint()
        return self.request("read", (node, kind, log is not None), log)

    def exchange(self, frames, expected, checkpoint, log=None):
        checkpoint()
        return self.request("exchange", (frames, expected, log is not None), log)

    def cycle(self, targets, sampled_at, expires_at, wait_applied):
        self.sequence += 1
        frames = tuple((n, self.motors[n].mode, frame) for n, frame in sorted(targets.items()))
        command = CycleCommand(self.sequence, sampled_at, expires_at, perf_counter(), frames)
        previous, self.submitted = self.submitted, command
        with_tool = self.request("cycle", (command, wait_applied))
        self.confirmed = command if self.applied == command.sequence else previous
        if self.confirmed is None or self.confirmed.sequence != self.applied:
            raise ZlgError("周期反馈与已投递的命令序号不符")
        return with_tool
