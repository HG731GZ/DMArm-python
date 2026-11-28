"""主端 UDP 套接字、独立接收线程和正常发布的预测器历史。"""

from collections import deque
from dataclasses import dataclass
from secrets import randbelow
import socket
from threading import Event, Lock, Thread
from time import monotonic, perf_counter, time

from UdpEventLog import SocketEvent, UdpEventLog, now_us
from UdpProtocol import BUSINESS, NETWORK_ONLY, FeedbackTiming, decode_slave, encode_master


def next_deadline(deadline, period):
    """沿 UDP 发布的原定时轴前进，跳过已经错过的时隙。"""
    return deadline + (int((perf_counter() - deadline) / period) + 1) * period


@dataclass(frozen=True)
class MasterCommand:
    sent_at: float  # 与反馈接收时刻同一单调时间轴，单位 s
    q: tuple


@dataclass(frozen=True)
class SlaveFrame:
    q: tuple
    mode: int
    grip: tuple
    dq: tuple
    torque: tuple
    teleop_state: int
    ddq: tuple
    received: float
    wall_time: float
    generation: int
    timing: FeedbackTiming | None = None  # 旧日志离线预览没有 v2 字段
    t4_us: int | None = None


class FeedbackOrder:
    def __init__(self):
        self.session = 0
        self.retired = set()
        self.seen = set()
        self.highest = 0

    def classify(self, session, seq):
        if session != self.session:
            if session in self.retired:
                return "old_session"
            self.retired.add(self.session)
            self.session = session
            self.seen.clear()
            self.highest = 0
        if seq in self.seen:
            return "duplicate"
        self.seen.add(seq)
        if seq < self.highest:
            return "reordered"
        self.highest = seq
        return "new"


class UdpClass(Thread):
    def __init__(self, local, peer, timeout, *, command_history_s, kind=BUSINESS,
                 directory=None, session_changed=None):
        super().__init__(name="DMArm-UDP")
        self.peer, self.timeout, self.kind = peer, timeout, kind
        self.latest = None
        self.error = ""
        self.rx = self.rejected = self.tx = 0
        self.order = FeedbackOrder()
        self.session_changed = session_changed
        self.command_history_s = command_history_s
        self.send_lock = Lock()
        self.session_lock = Lock()
        self.history = (0, ())
        self.cmd_seq = 0
        self.commands = deque()
        self.stopping = Event()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self.sock.bind(local)
            self.sock.settimeout(0.05)
            self.log = UdpEventLog(directory, kind, self.sock.getsockname(), peer) if kind == NETWORK_ONLY else None
        except OSError:
            self.sock.close()
            raise

    def start(self):
        if self.log is not None:
            self.log.start()
        super().start()

    def run(self):
        generation = 0
        while not self.stopping.is_set() and not self.error:
            try:
                data, source = self.sock.recvfrom(65535)
                t4 = now_us()
                # 旧控制接口以 monotonic 秒计算断流和预测，与新增的 QPC 整数列分别保存。
                received, wall_time = monotonic(), time()
            except socket.timeout:
                continue
            except OSError as error:
                # Windows 将对端尚未绑定端口产生的 ICMP 通知交给下一次 UDP 接收。
                if getattr(error, "winerror", None) == 10054:
                    continue
                self.error = f"UDP 接收失败：{error}"
                return
            timing = None
            if source != self.peer:
                status = "端点不符"
            else:
                try:
                    values, timing = decode_slave(data, self.kind)
                except ValueError as error:
                    status = str(error)
                else:
                    # 切换会话与反馈分发互斥；套接字发送期间接收线程仍可继续取包。
                    with self.session_lock:
                        session = self.history[0]
                        if session and timing.echo_master_session not in (0, session):
                            status = "old_echo_session"
                        else:
                            previous = self.order.session
                            status = self.order.classify(timing.slave_session, timing.fb_seq)
                            if self.order.session != previous and self.session_changed is not None:
                                self.session_changed("从端", session, self.order.session)
                        if status == "new" and self.kind == BUSINESS:
                            last = self.latest
                            if (last is None or last.timing.slave_session != timing.slave_session
                                    or received - last.received > self.timeout):
                                generation += 1
                            self.latest = SlaveFrame(*values, received, wall_time, generation, timing, t4)
            if status == "new":
                self.rx += 1
            else:
                self.rejected += 1
            if self.log is not None:
                self.log.record(SocketEvent("rx", t4, source, data, status, feedback=timing))

    def fresh(self):
        frame = self.latest
        if (frame and not self.stopping.is_set() and not self.error
                and monotonic() - frame.received <= self.timeout):
            return frame
        return None

    def start_publishing(self):
        with self.send_lock, self.session_lock:
            self.cmd_seq = 0
            session = randbelow(2**64 - 1) + 1
            self.history = session, ()
            if self.session_changed is not None:
                self.session_changed("主端", session, self.order.session)
            return session

    def send(self, payload, session, q=None):
        """单一发布者提交原载荷；与停止操作互斥，成功发送才记录。"""
        with self.send_lock:
            if not session or session != self.history[0] or self.stopping.is_set():
                return None
            seq = self.cmd_seq + 1
            t1 = now_us()
            data = encode_master(payload, self.kind, session, seq, t1)
            try:
                self.sock.sendto(data, self.peer)
            except OSError as error:
                if self.log is not None:
                    self.log.record(SocketEvent("tx_error", t1, self.peer, data, str(error), session, seq))
                raise
            sent_at = monotonic()
            self.cmd_seq = seq
            self.tx += 1
            if self.log is not None:
                self.log.record(SocketEvent("tx", t1, self.peer, data, "sent", session, seq))
            if self.kind == BUSINESS:
                self.commands.append(MasterCommand(sent_at, tuple(q)))
                cutoff = sent_at - self.command_history_s
                # 保留截止时刻之前的最后一条指令，覆盖零阶保持区间。
                while len(self.commands) > 1 and self.commands[1].sent_at < cutoff:
                    self.commands.popleft()
                self.history = session, tuple(self.commands)
            return data

    def stop_publishing(self):
        with self.send_lock, self.session_lock:
            self.commands.clear()
            self.history = 0, ()

    def close(self):
        self.stopping.set()
        self.join()
        self.sock.close()
        if self.log is not None:
            self.log.close()
