"""按 Windows x64 厂商 zlgcan.h 和 canframe.h 绑定句柄接口。"""

import ctypes as C
import os

from zlgcan import (
    BOARD_FIELDS, DEVICE_TYPE, RX_BATCH, TYPE_CANFD, ReceivedFD, ZlgError,
    bind, board_description, check, frame_count,
)

U8, U16, U32 = C.c_uint8, C.c_uint16, C.c_uint32
HANDLE = C.c_void_p
CUSTOM_TIMING = b"1.0Mbps(75%),5.0Mbps(75%),(60,00000E2B,00800001)"


class BoardInfo(C.Structure):
    _fields_ = BOARD_FIELDS


class FDInit(C.Structure):
    _fields_ = [
        ("acc_code", U32), ("acc_mask", U32), ("abit_timing", U32), ("dbit_timing", U32),
        ("brp", U32), ("filter", U8), ("mode", U8), ("pad", U16), ("reserved", U32),
    ]


class Init(C.Structure):
    # 厂商联合体由最大的 CAN FD 成员决定尺寸和对齐，此处只绑定该成员。
    _fields_ = [("can_type", U32), ("canfd", FDInit)]


class Frame(C.Structure):
    _fields_ = [("can_id", U32), ("length", U8), ("flags", U8), ("reserved", U8 * 2),
                ("data", U8 * 64)]


class RxFD(C.Structure):
    _fields_ = [("frame", Frame), ("timestamp", C.c_uint64)]


class TxFD(C.Structure):
    _fields_ = [("frame", Frame), ("transmit_type", U32)]


class WindowsCANFD:
    def __init__(self, directory, device, channel, resources):
        for path in (directory, directory / "kerneldlls"):
            resources.enter_context(os.add_dll_directory(str(path)))
        self.lib = lib = C.WinDLL(str(directory / "zlgcan.dll"))
        self.device_index = device
        self.channel_index = channel
        bind(lib, "ZCAN_OpenDevice", HANDLE, U32, U32, U32)
        bind(lib, "ZCAN_CloseDevice", U32, HANDLE)
        bind(lib, "ZCAN_GetDeviceInf", U32, HANDLE, C.POINTER(BoardInfo))
        bind(lib, "ZCAN_SetValue", U32, HANDLE, C.c_char_p, C.c_void_p)
        bind(lib, "ZCAN_InitCAN", HANDLE, HANDLE, U32, C.POINTER(Init))
        for name in ("ZCAN_ClearBuffer", "ZCAN_StartCAN", "ZCAN_ResetCAN"):
            bind(lib, name, U32, HANDLE)
        bind(lib, "ZCAN_GetReceiveNum", U32, HANDLE, U8)
        bind(lib, "ZCAN_ReceiveFD", U32, HANDLE, C.POINTER(RxFD), U32, C.c_int32)
        bind(lib, "ZCAN_TransmitFD", U32, HANDLE, C.POINTER(TxFD), U32)
        self.tx_buffer = (TxFD * 7)()
        self.buffer = (RxFD * RX_BATCH)()

    def open_device(self):
        self.device = self.lib.ZCAN_OpenDevice(DEVICE_TYPE, self.device_index, 0)
        if not self.device:
            raise ZlgError("ZCAN_OpenDevice 失败，返回句柄 0")

    def set_value(self, name, value):
        path = f"{self.channel_index}/{name}".encode("ascii")
        check(f"ZCAN_SetValue({name})", self.lib.ZCAN_SetValue(self.device, path, value))

    def configure(self, listen_only):
        info = BoardInfo()
        check("ZCAN_GetDeviceInf", self.lib.ZCAN_GetDeviceInf(self.device, C.byref(info)))
        self.set_value("canfd_standard", b"0")
        self.set_value("baud_rate_custom", CUSTOM_TIMING)
        config = Init()
        config.can_type = TYPE_CANFD
        config.canfd.mode = int(listen_only)
        self.channel = self.lib.ZCAN_InitCAN(self.device, self.channel_index, C.byref(config))
        if not self.channel:
            raise ZlgError("ZCAN_InitCAN 失败，返回句柄 0")
        self.set_value("initenal_resistance", b"1")
        self.set_value("tx_timeout", b"100")
        self.set_value("filter_clear", b"0")
        check("ZCAN_ClearBuffer", self.lib.ZCAN_ClearBuffer(self.channel))
        self.set_value("set_device_tx_echo", b"0")
        return board_description(info)

    def start(self):
        check("ZCAN_StartCAN", self.lib.ZCAN_StartCAN(self.channel))

    def clear_transmit_queue(self):
        self.set_value("clear_delay_send_queue", b"0")

    def enable_transmit_queue(self):
        self.set_value("set_send_mode", b"1")

    def transmit_batch(self, frames, *, queued=False):
        for index, (can_id, data) in enumerate(frames):
            tx = self.tx_buffer[index]
            tx.transmit_type = 0  # 正常发送，PV 帧仲裁让出后仍由控制器发出
            frame = tx.frame
            frame.can_id = can_id
            frame.flags = (int(queued) << 4) | 1  # 队列标志、BRS；保留字节为零表示无附加延时
            frame.length = len(data)
            frame.data[:len(data)] = data
        return self.lib.ZCAN_TransmitFD(self.channel, self.tx_buffer, len(frames))

    def transmit_fd(self, can_id, data):
        count = self.transmit_batch(((can_id, data),))
        check("ZCAN_TransmitFD", count)
        return count

    def receive(self, wait_ms=0, limit=RX_BATCH):
        if not wait_ms:
            waiting = self.lib.ZCAN_GetReceiveNum(self.channel, TYPE_CANFD)
            limit = min(limit, frame_count("ZCAN_GetReceiveNum(FD)", waiting))
            if not limit:
                return []
        count = frame_count("ZCAN_ReceiveFD", self.lib.ZCAN_ReceiveFD(
            self.channel, self.buffer, limit, wait_ms))
        frames = []
        for entry in self.buffer[:count]:
            frame = entry.frame
            frames.append(ReceivedFD(frame.can_id & 0x1FFFFFFF, bool(frame.can_id & 0x80000000),
                                     bool(frame.flags & 1), bytes(frame.data[:frame.length]),
                                     entry.timestamp, not (frame.can_id & 0x60000000 or frame.flags & 0x20)))
        return frames

    def reset(self):
        check("ZCAN_ResetCAN", self.lib.ZCAN_ResetCAN(self.channel))

    def close_device(self):
        check("ZCAN_CloseDevice", self.lib.ZCAN_CloseDevice(self.device))
