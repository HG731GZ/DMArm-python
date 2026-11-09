"""按 Linux x86_64 厂商 zcan.h 绑定 VCI 索引接口。"""

import ctypes as C

from zlgcan import BOARD_FIELDS, DEVICE_TYPE, RX_BATCH, ReceivedFD, bind, board_description, check, frame_count

U8, U16, U32 = C.c_uint8, C.c_uint16, C.c_uint32


class Timing(C.Structure):
    _pack_ = 1
    _fields_ = [("tseg1", U8), ("tseg2", U8), ("sjw", U8), ("smp", U8), ("brp", U16)]


class Init(C.Structure):
    _pack_ = 1
    _fields_ = [("clk", U32), ("mode", U32), ("aset", Timing), ("dset", Timing)]


class BoardInfo(C.Structure):
    _pack_ = 1
    _fields_ = BOARD_FIELDS


class Frame(C.Structure):
    _pack_ = 1
    _fields_ = [
        ("timestamp", U32), ("can_id", U32), ("info", U32), ("pad", U16),
        ("channel", U8), ("length", U8), ("data", U8 * 64),
    ]


class LinuxCANFD:
    def __init__(self, directory, device, channel):
        self.card = (DEVICE_TYPE, device)
        self.port = (*self.card, channel)
        self.fd_queue = (*self.card, channel | 0x80000000)
        self.lib = lib = C.CDLL(str(directory / "libusbcanfd.so"))
        bind(lib, "VCI_OpenDevice", U32, U32, U32, U32)
        bind(lib, "VCI_CloseDevice", U32, U32, U32)
        bind(lib, "VCI_ReadBoardInfo", U32, U32, U32, C.POINTER(BoardInfo))
        bind(lib, "VCI_InitCAN", U32, U32, U32, U32, C.POINTER(Init))
        bind(lib, "VCI_SetReference", U32, U32, U32, U32, U32, C.c_void_p)
        for name in ("VCI_ClearBuffer", "VCI_StartCAN", "VCI_ResetCAN", "VCI_GetReceiveNum"):
            bind(lib, name, U32, U32, U32, U32)
        bind(lib, "VCI_ReceiveFD", U32, U32, U32, U32, C.POINTER(Frame), U32, U32)
        bind(lib, "VCI_TransmitFD", U32, U32, U32, U32, C.POINTER(Frame), U32)
        self.tx_buffer = (Frame * 7)()
        self.buffer = (Frame * RX_BATCH)()

    def open_device(self):
        check("VCI_OpenDevice", self.lib.VCI_OpenDevice(*self.card, 0))

    def configure(self, listen_only):
        info = BoardInfo()
        check("VCI_ReadBoardInfo", self.lib.VCI_ReadBoardInfo(*self.card, C.byref(info)))
        # 寄存器以段长减一编码。60 MHz 下仲裁段 60 个时间份额，数据段分频 3、4 个份额。
        # 两段采样点分别为 45/60 和 3/4。模式位 0 置一启用只听，位 1 为零选择 ISO。
        config = Init(60_000_000, int(listen_only), Timing(43, 14, 0, 0, 0), Timing(1, 0, 0, 0, 2))
        check("VCI_InitCAN", self.lib.VCI_InitCAN(*self.port, C.byref(config)))
        check("VCI_SetReference(终端电阻)", self.lib.VCI_SetReference(
            *self.port, 0x18, C.byref(U32(1))))
        check("VCI_SetReference(发送超时)", self.lib.VCI_SetReference(
            *self.port, 0x44, C.byref(U32(100))))
        check("VCI_ClearBuffer(FD)", self.lib.VCI_ClearBuffer(*self.fd_queue))
        return board_description(info)

    def start(self):
        check("VCI_StartCAN", self.lib.VCI_StartCAN(*self.port))

    def clear_transmit_queue(self):
        check("VCI_SetReference(清发送队列)", self.lib.VCI_SetReference(
            *self.port, 0x102, C.byref(U32(1))))

    def enable_transmit_queue(self):
        check("VCI_SetReference(启用发送队列)", self.lib.VCI_SetReference(
            *self.port, 0x103, C.byref(U8(1))))

    def transmit_batch(self, frames, *, queued=False):
        for index, (can_id, data) in enumerate(frames):
            frame = self.tx_buffer[index]
            frame.can_id = can_id
            frame.channel = self.port[2]
            # 低四位取正常发送，PV 帧遇到高优先级反馈后仍可参与下一次仲裁。
            frame.info = (int(queued) << 16) | (1 << 11) | (1 << 4)
            frame.length = len(data)
            frame.data[:len(data)] = data
        return self.lib.VCI_TransmitFD(*self.port, self.tx_buffer, len(frames))

    def transmit_fd(self, can_id, data):
        count = self.transmit_batch(((can_id, data),))
        check("VCI_TransmitFD", count)
        return count

    def receive(self, wait_ms=0, limit=RX_BATCH):
        # SDK 的零超时空读取仍会进入一次定时等待，排空时先查数量。
        if not wait_ms:
            waiting = self.lib.VCI_GetReceiveNum(*self.fd_queue)
            limit = min(limit, frame_count("VCI_GetReceiveNum(FD)", waiting))
            if not limit:
                return []
        count = frame_count("VCI_ReceiveFD", self.lib.VCI_ReceiveFD(
            *self.port, self.buffer, limit, wait_ms))
        return [
            ReceivedFD(frame.can_id, bool(frame.info & (1 << 9)), bool(frame.info & (1 << 11)),
                       bytes(frame.data[:frame.length]), frame.timestamp,
                       not (frame.info & ((1 << 8) | (1 << 10) | (1 << 13))))
            for frame in self.buffer[:count]
        ]

    def reset(self):
        check("VCI_ResetCAN", self.lib.VCI_ResetCAN(*self.port))

    def close_device(self):
        check("VCI_CloseDevice", self.lib.VCI_CloseDevice(*self.card))
