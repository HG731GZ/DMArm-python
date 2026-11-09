"""由单一工作线程持有的 CAN FD 会话和有界请求应答。"""

from contextlib import ExitStack
from time import monotonic

from DMMotor import DMMotor, MOTOR_MODELS, PARAM_ID, Read
from zlgcan import RX_BATCH, ZlgError, load_backend


class OperationInterrupted(Exception):
    """普通操作让出总线，以便处理操作员的停止请求。"""


class USBCANFD:
    def __init__(self, windows_dir, linux_dir, device_index, channel_index, query_timeout=0.5):
        self.connection = (windows_dir, linux_dir, device_index, channel_index)
        self.timeout = query_timeout
        self.resources = ExitStack()
        self.backend = None
        self.started = False
        self.motors = {i: DMMotor(i, model) for i, model in enumerate(MOTOR_MODELS, 1)}
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
        self.send(frames, queued=queued)
        if log is not None:
            for can_id, data in frames:
                log(f"TX CAN FD 标准 ID=0x{can_id:X} BRS=1 长度={len(data)} {data.hex(' ').upper()}")
        missing = set(expected.items())
        replies = {}
        deadline = monotonic() + self.timeout
        while monotonic() < deadline:
            checkpoint()
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
