"""主从 UDP v2 的大端线格式与反馈校验。"""

from math import isfinite, isinf
from struct import Struct
from typing import NamedTuple
from zlib import crc32


HEADER = bytes.fromhex("55 AA 55 AA")
FOOTER = b"\r\n\r\n"
VERSION = 2
BUSINESS, NETWORK_ONLY = 0, 1
MASTER = Struct(">6dB3d6d")
SLAVE = Struct(">6dB3d6d6dB6d")
MASTER_EXT = Struct(">BB3Q")
SLAVE_EXT = Struct(">BB13Q")
SLAVE_FRAME_SIZE = 12 + SLAVE.size + SLAVE_EXT.size


class FeedbackTiming(NamedTuple):
    slave_session: int
    fb_seq: int
    echo_master_session: int
    echo_cmd_seq: int
    echo_t1_us: int
    t2_us: int
    t3_us: int
    applied_master_session: int
    applied_cmd_seq: int
    t_apply_us: int
    t_qdq_us: int
    t_tau_us: int
    t_ddq_us: int


def master_payload(q, dq, grip):
    return MASTER.pack(*q, 1, min(0.99, max(0, grip)), 0, 0, *dq)


def encode_master(payload, kind, session, seq, t1_us):
    body = payload + MASTER_EXT.pack(VERSION, kind, session, seq, t1_us)
    return HEADER + body + crc32(body).to_bytes(4, "big") + FOOTER


def decode_slave(data, kind):
    if len(data) != SLAVE_FRAME_SIZE or not data.startswith(HEADER) or not data.endswith(FOOTER):
        raise ValueError("帧长度或边界错误")
    body = data[4:-8]
    if crc32(body) != int.from_bytes(data[-8:-4], "big"):
        raise ValueError("CRC32 错误")
    version, wire_kind, *fields = SLAVE_EXT.unpack_from(body, SLAVE.size)
    if version != VERSION or wire_kind != kind:
        raise ValueError("版本或模式不符")
    timing = FeedbackTiming(*fields)
    if not timing.slave_session or not timing.fb_seq:
        raise ValueError("从端会话或序号为零")
    if kind == NETWORK_ONLY and (any(body[:SLAVE.size]) or any(fields[7:])):
        raise ValueError("诊断载荷或采用、采样字段非零")
    if ((any(fields[2:6]) and (not all(fields[2:6]) or timing.t2_us > timing.t3_us))
            or (any(fields[7:10]) and not all(fields[7:10]))):
        raise ValueError("回显或采用事件字段不一致")
    values = SLAVE.unpack_from(body)
    q, grip, dq, torque, ddq = values[:6], values[7:10], values[10:16], values[16:22], values[23:29]
    if not all(map(isfinite, (*q, *dq))) or any(map(isinf, (*grip, *torque, *ddq))):
        raise ValueError("业务字段含无效数值")
    return (q, values[6], grip, dq, torque, values[22], ddq), timing
