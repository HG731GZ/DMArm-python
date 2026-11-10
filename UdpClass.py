"""UR 主从 UDP 报文与独立的从端接收线程。"""

from dataclasses import dataclass
from math import isfinite
import socket
import struct
from threading import Event, Thread
from time import monotonic, time
from zlib import crc32


HEADER = bytes.fromhex("55 AA 55 AA")
FOOTER = b"\r\n\r\n"
SLAVE = struct.Struct(">6dB3d")
MASTER = struct.Struct(">6dB3d6d")


@dataclass(frozen=True)
class SlaveFrame:
    q: tuple
    mode: int
    grip: tuple
    received: float
    wall_time: float
    generation: int


def decode_slave(data):
    if len(data) != 85 or data[:4] != HEADER or data[-4:] != FOOTER:
        raise ValueError("从端帧长度或边界错误")
    payload = data[4:77]
    if crc32(payload) != int.from_bytes(data[77:81], "big"):
        raise ValueError("从端帧 CRC32 错误")
    values = SLAVE.unpack(payload)
    if not all(map(isfinite, values)):
        raise ValueError("从端字段含非有限数值")
    return values[:6], values[6], values[7:]


def encode_master(q, dq, grip):
    payload = MASTER.pack(*q, 1, min(0.99, max(0, grip)), 0, 0, *dq)
    return HEADER + payload + crc32(payload).to_bytes(4, "big") + FOOTER


class UdpClass(Thread):
    def __init__(self, local, peer, timeout):
        super().__init__(name="DMArm-UDP")
        self.peer = peer
        self.timeout = timeout
        self.latest = None
        self.error = ""
        self.rx = self.rejected = self.tx = 0
        self.stopping = Event()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self.sock.bind(local)
            self.sock.settimeout(0.05)
        except OSError:
            self.sock.close()
            raise

    def run(self):
        generation = 0
        while not self.stopping.is_set():
            try:
                data, source = self.sock.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError as error:
                self.error = str(error)
                return
            if source != self.peer:
                self.rejected += 1
                continue
            try:
                q, mode, grip = decode_slave(data)
            except ValueError:
                self.rejected += 1
                continue
            now = monotonic()
            if self.latest is None or now - self.latest.received > self.timeout:
                generation += 1
            self.latest = SlaveFrame(q, mode, grip, now, time(), generation)
            self.rx += 1

    def fresh(self):
        frame = self.latest
        if (frame and not self.stopping.is_set() and not self.error
                and monotonic() - frame.received <= self.timeout):
            return frame
        return None

    def send(self, data):
        self.sock.sendto(data, self.peer)
        self.tx += 1

    def close(self):
        self.stopping.set()
        self.join()
        self.sock.close()
