"""单轴控制拍记录及绘图进程的批量传输格式。"""

from struct import Struct
from typing import NamedTuple


# 节点、时间和各环数值、三个限幅标记、锁定轴及偏差、上一目标。事件文字随批头传递。
SAMPLE_RECORD = Struct("<B11d3?B2d")


class TunePoint(NamedTuple):
    node: int
    sent_at: float
    feedback_at: float
    target: float
    position: float
    velocity_ref: float
    velocity: float
    gravity: float
    pid_raw: float
    pid: float
    command: float
    torque: float
    pid_limited: bool
    output_limited: bool
    rate_limited: bool
    lock_node: int
    lock_error: float
    event: str
    previous_target: float

    def pack(self):
        return SAMPLE_RECORD.pack(*self[:-2], self.previous_target)
