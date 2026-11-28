"""后台逐包记录套接字事件，并在连接结束时汇总测时结果。"""

from bisect import bisect_right
from collections import Counter
import csv
from datetime import datetime
import json
from queue import SimpleQueue
from threading import Thread
from time import perf_counter_ns
from typing import NamedTuple

from UdpProtocol import BUSINESS, FeedbackTiming


def now_us():
    # dmarm 的 Windows Python 3.10 使用 QPC 提供高分辨率单调测时。
    return perf_counter_ns() // 1000


class SocketEvent(NamedTuple):
    event: str
    at_us: int
    peer: tuple
    data: bytes
    status: str
    session: int = 0
    seq: int = 0
    feedback: FeedbackTiming | None = None


def distribution(samples):
    """最近秩分位数，整数微秒；空样本的统计量留空。"""
    ordered = sorted(samples)
    values = [ordered[(len(ordered) * p + 99) // 100 - 1] if ordered else None
              for p in (50, 95, 99, 100)]
    return dict(zip(("n", "p50", "p95", "p99", "max"), [len(ordered), *values]))


class NetworkStats:
    def __init__(self):
        self.sent = {}
        self.tx_errors = 0
        self.echoes = {}
        self.received = set()
        self.status = Counter()
        self.arrivals = []
        self.business = []
        self.torque_age = []

    def observe(self, e, kind):
        if e.event == "tx_error":
            self.tx_errors += 1
            return
        if e.event == "tx":
            self.sent[e.session, e.seq] = e.at_us
            return
        self.status[e.status] += 1
        f = e.feedback
        if f is None:
            return
        self.arrivals.append(e.at_us)
        self.received.add((f.slave_session, f.fb_seq))
        if e.status == "new" and kind == BUSINESS:
            self.business.append(e.at_us)
            if f.t_tau_us:
                self.torque_age.append(f.t3_us - f.t_tau_us)
        if f.echo_cmd_seq:
            # 两个线程入队的先后不保证等于收发的先后，收齐实际 TX 后再匹配。
            key = f.echo_master_session, f.echo_cmd_seq, f.echo_t1_us
            self.echoes.setdefault(key, (e.at_us, f.t3_us - f.t2_us))

    def summary(self, start, end):
        rtt, transport, ages = [], [], []
        for key, t1 in self.sent.items():
            match = self.echoes.get((*key, t1))
            if match:
                t4, dwell = match
                rtt.append(t4 - t1)
                transport.append(t4 - t1 - dwell)
            i = bisect_right(self.business, t1)
            if i:
                ages.append(t1 - self.business[i - 1])
        gaps = [b - a for a, b in zip(self.arrivals, self.arrivals[1:])]
        overlap = 0
        if self.sent and self.arrivals:
            overlap = max(0, min(max(self.sent.values()), self.arrivals[-1])
                          - max(min(self.sent.values()), self.arrivals[0]))
        return dict(duration_us=end - start, bidirectional_span_us=overlap,
                    tx=len(self.sent), tx_errors=self.tx_errors, rx_datagrams=sum(self.status.values()), rx_valid=len(self.arrivals),
                    rx_unique=len(self.received), rx_status=dict(self.status),
                    rx_duplicate=self.status["duplicate"], rx_reordered=self.status["reordered"],
                    rx_old_session=self.status["old_session"], echo_covered_commands=len(rtt),
                    echo_rtt_us=distribution(rtt), round_trip_minus_dwell_us=distribution(transport),
                    feedback_interval_us=distribution(gaps), intervals_over_50ms=sum(g > 50_000 for g in gaps),
                    torque_sample_age_us=distribution(self.torque_age), business_age_at_tx_us=distribution(ages),
                    last_business_age_us=end - self.business[-1] if self.business else None)


class UdpEventLog(Thread):
    def __init__(self, directory, kind, local, peer):
        super().__init__(name="DMArm-UDP-Log")
        directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.path = directory / f"udp_{stamp}.csv"
        self.summary_path = self.path.with_suffix(".json")
        self.output = self.path.open("x", encoding="utf-8-sig", newline="")
        self.kind, self.local, self.peer = kind, local, peer
        self.queue = SimpleQueue()
        self.error = ""
        self.start_us = now_us()

    def record(self, event):
        if not self.error:
            self.queue.put(event)

    def run(self):
        stats = NetworkStats()
        try:
            with self.output as output:
                writer = csv.writer(output)
                writer.writerow(("event", "kind", "peer_ip", "peer_port", "bytes", "status",
                                 "master_session", "cmd_seq", "t1_us", "t4_us",
                                 *FeedbackTiming._fields, "datagram_hex"))
                for e in iter(self.queue.get, None):
                    tx = e.event in ("tx", "tx_error")
                    writer.writerow((e.event, self.kind, *e.peer, len(e.data), e.status,
                                     e.session if tx else None, e.seq if tx else None,
                                     e.at_us if tx else None, None if tx else e.at_us,
                                     *(e.feedback or (None,) * len(FeedbackTiming._fields)), e.data.hex()))
                    stats.observe(e, self.kind)
            report = dict(local=self.local, peer=self.peer, kind=self.kind,
                          **stats.summary(self.start_us, self.end_us))
            self.summary_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        except OSError as error:
            self.error = f"UDP 逐包日志写入失败：{error}"

    def close(self):
        self.end_us = now_us()
        self.queue.put(None)
        self.join()
