"""两个平台共用的 CAN FD 接收记录与 SDK 返回值约定。"""

import ctypes as C
import sys
from typing import NamedTuple

DEVICE_TYPE = 43  # USBCANFD-100U-mini
TYPE_CANFD = 1
RX_BATCH = 128


class ZlgError(RuntimeError):
    """保留失败的 SDK 操作及其原始返回值。"""


def bind(lib, name, result, *parameters):
    function = getattr(lib, name)
    function.restype = result
    function.argtypes = parameters


def check(step, result):
    if result != 1:
        raise ZlgError(f"{step} 失败，返回值 {result}")


def frame_count(step, result):
    if result == 0xFFFFFFFF:
        raise ZlgError(f"{step} 失败，返回值 0xFFFFFFFF")
    return result


class ReceivedFD(NamedTuple):
    can_id: int
    extended: bool
    brs: bool
    data: bytes
    timestamp: int
    is_data: bool = True

    def __str__(self):
        kind = "扩展" if self.extended else "标准"
        return (f"时间戳={self.timestamp}  CAN FD {kind}  ID=0x{self.can_id:X}  "
                f"BRS={int(self.brs)}  长度={len(self.data)}  {self.data.hex(' ').upper()}")


# 板卡信息的字段顺序相同，Linux 按 1 字节紧排，Windows 使用原生对齐。
BOARD_FIELDS = [
    ("hardware", C.c_uint16), ("firmware", C.c_uint16), ("driver", C.c_uint16),
    ("api", C.c_uint16), ("irq", C.c_uint16), ("channels", C.c_uint8),
    ("serial", C.c_char * 20), ("model", C.c_char * 40), ("reserved", C.c_uint16 * 4),
]


def board_description(info):
    return (f"{info.model.decode('ascii')} · SN {info.serial.decode('ascii')} · "
            f"固件 0x{info.firmware:04X} · 接口 0x{info.api:04X}")


def load_backend(windows_dir, linux_dir, device, channel, resources):
    if sys.platform == "win32":
        from zlgcan_win import WindowsCANFD
        return WindowsCANFD(windows_dir, device, channel, resources)
    if sys.platform == "linux":
        from zlgcan_linux import LinuxCANFD
        return LinuxCANFD(linux_dir, device, channel)
    raise ZlgError(f"此平台未配置 ZLG SDK：{sys.platform}")
