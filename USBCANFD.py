"""由收发进程独占的 CAN FD 会话、整批应答与失能确认。"""

from contextlib import ExitStack
from time import monotonic, perf_counter
from typing import NamedTuple

from DMMotor import DMMotor, MODE_REGISTER, MOTOR_MODELS, PARAM_ID, Read
from zlgcan import RX_BATCH, ZlgError, load_backend


class OperationInterrupted(Exception):
    """普通操作让出总线，以便处理操作员的停止请求。"""


def uninterrupted():
    """失能和随后的状态回读须一起完成。"""


class DisableResult(NamedTuple):
    nodes: frozenset
    verified: frozenset
    uncertain: frozenset
    queue_error: str


class USBCANFD:
    def __init__(self, windows_dir, linux_dir, device_index, channel_index, query_timeout=0.5,
                 *, joint_torque_limits):
        self.connection = (windows_dir, linux_dir, device_index, channel_index)
        self.timeout = query_timeout
        self.resources = ExitStack()
        self.backend = None
        self.started = False
        if len(joint_torque_limits) != 6:
            raise ValueError("电机软件输出力矩上限须按 J1～J6 填写六项")
        self.motors = {i: DMMotor(i, model, limit) for i, (model, limit) in
                       enumerate(zip(MOTOR_MODELS, (*joint_torque_limits, None)), 1)}
        self.tx = self.rx = self.failed = 0

    def open(self, listen_only=True):
        with ExitStack() as resources:
            backend = load_backend(*self.connection, resources)
            backend.open_device()
            self.backend = backend
            self.resources = resources.pop_all()
        try:
            description = backend.configure(listen_only)
            backend.start()
            self.started = True
            if not listen_only:
                backend.clear_transmit_queue()
        except Exception:
            self.close()
            raise
        return description

    def close(self):
        try:
            if self.started:
                self.backend.reset()
                self.started = False
        finally:
            self.backend.close_device()
            self.started = False
            self.backend = None
            self.resources.close()

    def send(self, frames, *, queued=False):
        count = self.backend.transmit_batch(frames, queued=queued)
        accepted = count if count <= len(frames) else 0
        self.tx += accepted
        if count != len(frames):
            self.failed += len(frames) - accepted
            raise ZlgError(f"FD 发送请求 {len(frames)} 帧，SDK 返回 {count}")
        return count

    def receive(self, expected=None, log=None, *, wait_ms=0, limit=RX_BATCH):
        replies = {}
        try:
            frames = self.backend.receive(wait_ms, limit)
        except ZlgError:
            self.failed += 1
            raise
        self.rx += len(frames)
        for frame in frames:
            if log is not None:
                log(f"RX {frame}")
            motor = self.motors.get(frame.can_id - 0x10)
            if motor is None or frame.extended or not frame.is_data or len(frame.data) != 8:
                continue
            kind = motor.accept(frame.data, expected.get(motor.node) if expected else None)
            if kind is not None:
                replies[motor.node, kind] = frame.data
        return replies

    def exchange(self, frames, expected, checkpoint, log=None, *, disabling=False, queued=False):
        """清理已到达的接收帧，发送一批，分别匹配这一批各节点的应答。"""
        try:
            self.receive(log=log)
        except (ZlgError, OSError):
            # 接收侧故障时仍投递失能帧，由单独的状态查询确认结果。
            if disabling:
                self.send(frames)
            raise
        checkpoint()
        send_started = perf_counter()
        self.send(frames, queued=queued)
        sent_at = perf_counter()
        if log is not None:
            for can_id, data in frames:
                log(f"TX CAN FD 标准 ID=0x{can_id:X} BRS=1 长度={len(data)} {data.hex(' ').upper()}")
        missing = set(expected.items())
        replies = {}
        deadline = monotonic() + self.timeout
        while monotonic() < deadline:
            try:
                checkpoint()
            except ZlgError as error:
                pending = sorted(node for node, _ in missing)
                raise ZlgError(f"{error}，发送调用 {(sent_at - send_started) * 1000:.1f} ms，"
                               f"发送返回后 {(perf_counter() - sent_at) * 1000:.1f} ms，"
                               f"尚待节点 {pending} 回包") from error
            # 由 SDK 在帧到达时唤醒。短等待使停发请求也能及时接管总线。
            arrived = self.receive(expected, log, wait_ms=1, limit=len(missing))
            replies.update(arrived)
            missing.difference_update(arrived)
            if not missing:
                return replies
        self.failed += len(missing)
        detail = "、".join(f"ID {node} {kind.value}" for node, kind in sorted(missing))
        raise ZlgError(f"{detail}应答超时")

    def read(self, node, kind, checkpoint, log=None):
        motor = self.motors[node]
        if kind is Read.MODE:
            motor.mode = None
        elif kind is Read.STATUS:
            motor.feedback = None
        self.exchange([(PARAM_ID, kind.payload(node))], {node: kind}, checkpoint, log)
        return motor.mode if kind is Read.MODE else motor.feedback

    def write_modes(self, value, nodes, continuing, checkpoint, log):
        """连续完成寄存器写入、应答核对和回读，未换模轴在两批中继续控制。"""
        writes = {n: bytes((n, 0, 0x55, MODE_REGISTER, value, 0, 0, 0)) for n in nodes}
        for kind, payloads in ((Read.MODE_WRITE, writes), (Read.MODE, {n: Read.MODE.payload(n) for n in nodes})):
            frames = [(PARAM_ID, data) for data in payloads.values()]
            frames.extend(continuing.values())
            expected = dict.fromkeys(nodes, kind)
            expected.update(dict.fromkeys(continuing, Read.STATUS))
            replies = self.exchange(frames, expected, checkpoint, log)
            for n in continuing:
                fb = self.motors[n].feedback
                if fb.status != 1:
                    raise ZlgError(f"ID {n} 反馈退出使能：{fb.status_text}")
            for n in nodes:
                if kind is Read.MODE_WRITE and replies[n, kind] != writes[n]:
                    raise ZlgError(f"ID {n} 模式写入应答不符")
                if kind is Read.MODE and self.motors[n].mode != value:
                    raise ZlgError(f"ID {n} 模式回读为 {self.motors[n].mode}，要求 {value}")

    def disable_nodes(self, nodes, coherent, log):
        """给每个目标投递失能，并用单独的状态回读确认。"""
        verified, uncertain = set(), set()
        queue_error = ""
        try:
            self.backend.clear_transmit_queue()
        except (ZlgError, OSError) as error:
            queue_error = str(error)
            log(queue_error)
        for node in sorted(nodes):
            motor = self.motors[node]
            try:
                if not coherent or motor.mode not in (1, 2, 3, 4):
                    mode = self.read(node, Read.MODE, uninterrupted, log)
                    if mode not in (1, 2, 3, 4):
                        raise ZlgError(f"ID {node} 模式寄存器值 {mode} 未识别")
                acknowledged = True
                try:
                    self.exchange([(motor.command_id, b"\xff" * 7 + b"\xfd")], {node: Read.STATUS},
                                  uninterrupted, log, disabling=True)
                except (ZlgError, OSError) as error:
                    acknowledged = False
                    log(f"ID {node} 失能帧：{error}")
                fb = self.read(node, Read.STATUS, uninterrupted, log)
                if fb.status != 0:
                    raise ZlgError(f"回读 {fb.status_text}")
            except (ZlgError, OSError) as error:
                uncertain.add(node)
                log(f"ID {node} 失能未确认：{error}")
            else:
                if acknowledged:
                    verified.add(node)
                log(f"ID {node} 失能已回读 0")
        return DisableResult(frozenset(nodes), frozenset(verified), frozenset(uncertain), queue_error)
