"""达妙电机的只读请求、反馈解码及原 UR 主端的控制载荷编码。"""

from enum import Enum
from math import ceil, floor, isfinite
import struct
from time import perf_counter
from typing import NamedTuple

PARAM_ID = 0x7FF
MODE_REGISTER = 10
MODE_NAMES = {1: "MIT", 2: "PV", 3: "速度", 4: "PVT"}
STATUS_NAMES = {
    0: "失能", 1: "使能", 8: "超压", 9: "欠压", 10: "过流", 11: "MOS过热",
    12: "线圈过热", 13: "通讯丢失", 14: "过载",
}

# 现场从寄存器 21～23 逐台核对，单位依次为 rad、rad/s、N·m。
LIMITS = {4340: (12.5, 10, 28), 4310: (12.5, 30, 10), 3507: (12.566, 50, 5)}
MOTOR_MODELS = (4340, 4340, 4340, 4310, 4310, 4310, 3507)


class Read(Enum):
    MODE = "模式"
    STATUS = "状态"
    MODE_WRITE = "模式写入应答"

    def payload(self, node):
        if self is Read.MODE:
            return bytes((node, 0, 0x33, MODE_REGISTER, 0, 0, 0, 0))
        if self is Read.STATUS:
            return bytes((node, 0, 0xCC, 0))
        raise ValueError("模式写入需要目标值")


class Feedback(NamedTuple):
    position: float
    velocity: float
    torque: float
    mos: int
    rotor: int
    status: int

    @property
    def status_text(self):
        return f"0x{self.status:X} {STATUS_NAMES.get(self.status, '未知')}"

    @property
    def grip_force(self):
        return max(0.0, -self.torque / 0.01)


class DMMotor:
    def __init__(self, node, model, output_limit=None):
        self.node = node
        self.max_position, self.max_velocity, self.max_torque = LIMITS[model]
        if node == 7 and output_limit is None:
            output_limit = self.max_torque
        if output_limit is None or not isfinite(output_limit) or output_limit <= 0:
            raise ValueError(f"ID {node} 软件输出力矩上限须为正的有限值（N·m）")
        self.output_limit = min(output_limit, self.max_torque)
        self.min_torque_code = ceil(self.torque_scale(-self.output_limit))
        self.max_torque_code = floor(self.torque_scale(self.output_limit))
        if self.min_torque_code > self.max_torque_code:
            raise ValueError(f"ID {node} 输出上限内没有可发送的 MIT 力矩码")
        self.mode = None
        self.feedback = None
        self.feedback_at = 0.0

    @property
    def mode_text(self):
        if self.mode is None:
            return "—"
        return MODE_NAMES.get(self.mode, f"未知({self.mode})")

    def accept(self, data, expected=None):
        """由总线按反馈 ID 和帧格式分发；参数应答与运动反馈分别解析。"""
        if (data[0] & 0x0F) != self.node:
            return None
        if data[0] == self.node and data[1] == 0 and data[2] in (0x33, 0x55, 0xAA):
            if expected is Read.MODE_WRITE and data[2:4] == bytes((0x55, MODE_REGISTER)):
                return Read.MODE_WRITE
            if expected is Read.MODE and data[:4] == bytes((self.node, 0, 0x33, MODE_REGISTER)):
                self.mode = int.from_bytes(data[4:], "little")
                return Read.MODE
            return None
        position = (data[1] << 8) | data[2]
        velocity = (data[3] << 4) | (data[4] >> 4)
        torque = ((data[4] & 15) << 8) | data[5]
        self.feedback = Feedback(
            (position - 32768) * self.max_position / 32768,
            (velocity - 2048) * self.max_velocity / 2048,
            (torque - 2048) * self.max_torque / 2048,
            data[6], data[7], data[0] >> 4,
        )
        # 与 CAN 周期调度共用高分辨率时钟，区分同一系统时钟节拍内的连续回包。
        self.feedback_at = perf_counter()
        return Read.STATUS

    @property
    def command_id(self):
        return self.node + {1: 0, 2: 0x100, 3: 0x200, 4: 0x300}[self.mode]

    def empty_target(self):
        """按原主端生成当前模式的空指令，位置、限速、力矩和增益均为零。"""
        if self.mode == 1:
            data = self.encode_mit(0, 0, 0, 0, 0)
        elif self.mode == 2:
            data = self.encode_pv(0, 0)
        elif self.mode == 4 and self.node == 7:
            data = self.encode_pvt(0, 0, 0)
        else:
            raise ValueError(f"ID {self.node} 不支持以 {self.mode_text} 模式发送控制指令")
        return self.command_id, data

    def torque_scale(self, torque):
        return (torque + self.max_torque) * 4095 / (2 * self.max_torque)

    def torque_code(self, torque):
        code = int(self.torque_scale(torque))
        # 已发送力矩作为首帧起点时，避免浮点往返把同一力矩降到相邻码。
        if self.torque_value(code + 1) == torque:
            code += 1
        return max(self.min_torque_code, min(self.max_torque_code, code))

    @staticmethod
    def payload_torque_code(payload):
        return ((payload[6] & 15) << 8) | payload[7]

    def torque_value(self, code):
        """按 MIT 指令量程还原实际下发的 12 位力矩字段。"""
        return (2 * code - 4095) * self.max_torque / 4095

    def encode_mit(self, position, velocity, torque, kp, kd):
        """按电机协议量程编码 MIT 目标，返回八字节载荷。"""
        p = int((position + self.max_position) * 65535 / (2 * self.max_position))
        v = int((velocity + self.max_velocity) * 4095 / (2 * self.max_velocity))
        t = self.torque_code(torque)
        k = int(kp * 4095 / 500)
        d = int(kd * 4095 / 5)
        return p.to_bytes(2, "big") + ((v << 12) | k).to_bytes(3, "big") + ((d << 12) | t).to_bytes(3, "big")

    @staticmethod
    def encode_pv(position, velocity_limit):
        return struct.pack("<ff", position, velocity_limit)

    @staticmethod
    def encode_pvt(position, velocity_limit, torque_limit):
        """按原夹钳的位置上限和速度、电流比例编码。"""
        velocity = min(5000, int(velocity_limit * 100))
        current = min(5000, int(abs(torque_limit) * 1800))
        return struct.pack("<fHH", min(position, 2.35), velocity, current)
