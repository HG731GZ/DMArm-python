"""Windows CAN FD 周期线程使用的高分辨率定时器和命令唤醒事件。"""

from contextlib import ExitStack
import ctypes as C
from ctypes.wintypes import BOOL, DWORD, HANDLE, LONG, LPCWSTR
from math import ceil
from time import perf_counter


kernel = C.WinDLL("kernel32", use_last_error=True)


def bind(name, result, *arguments):
    function = getattr(kernel, name)
    function.restype = result
    function.argtypes = arguments
    return function


create_timer = bind("CreateWaitableTimerExW", HANDLE, C.c_void_p, LPCWSTR, DWORD, DWORD)
create_event = bind("CreateEventW", HANDLE, C.c_void_p, BOOL, BOOL, LPCWSTR)
set_timer = bind("SetWaitableTimer", BOOL, HANDLE, C.POINTER(C.c_longlong), LONG,
                 C.c_void_p, C.c_void_p, BOOL)
set_event = bind("SetEvent", BOOL, HANDLE)
wait_multiple = bind("WaitForMultipleObjects", DWORD, DWORD, C.POINTER(HANDLE), BOOL, DWORD)
close_handle = bind("CloseHandle", BOOL, HANDLE)

CREATE_WAITABLE_TIMER_HIGH_RESOLUTION = 2
TIMER_MODIFY_STATE = 2
SYNCHRONIZE = 0x100000
INFINITE = 0xFFFFFFFF


class WinCycleTimer:
    def __init__(self):
        with ExitStack() as resources:
            self.timer = create_timer(None, None, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION,
                                      TIMER_MODIFY_STATE | SYNCHRONIZE)
            if not self.timer:
                raise C.WinError(C.get_last_error())
            resources.callback(close_handle, self.timer)
            self.event = create_event(None, False, False, None)
            if not self.event:
                raise C.WinError(C.get_last_error())
            resources.callback(close_handle, self.event)
            self.handles = (HANDLE * 2)(self.event, self.timer)
            self.resources = resources.pop_all()

    def wait_until(self, deadline):
        remaining = deadline - perf_counter()
        if remaining <= 0:
            return
        # 负值为相对到期时间，单位 100 ns；每次按本轮剩余时间设置一次。
        due = C.c_longlong(-ceil(remaining * 10_000_000))
        if not set_timer(self.timer, C.byref(due), 0, None, None, False):
            raise C.WinError(C.get_last_error())
        if wait_multiple(2, self.handles, False, INFINITE) == 0xFFFFFFFF:
            raise C.WinError(C.get_last_error())

    def wake(self):
        if not set_event(self.event):
            raise C.WinError(C.get_last_error())

    def close(self):
        self.resources.close()
