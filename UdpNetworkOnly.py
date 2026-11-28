"""主端静止网络测时。在 PyCharm 中直接运行本文件即可采集一轮。"""

import argparse
from contextlib import ExitStack
from math import isfinite
from pathlib import Path
import sys
from time import perf_counter, sleep

from UdpClass import UdpClass, next_deadline
from UdpProtocol import MASTER, NETWORK_ONLY


LOCAL_IP = "192.168.1.5"  # Windows 主端有线网卡的 IPv4 地址
LOCAL_PORT = 6005  # 主端绑定的 UDP 端口
PEER_IP = "192.168.1.14"  # Ubuntu 从端有线网卡的 IPv4 地址
PEER_PORT = 5005  # 从端绑定的 UDP 端口
SEND_PERIOD_MS = 5  # 诊断帧发送间隔，单位 ms
CAPTURE_SECONDS = 120  # 单轮采集时长，单位 s
OUTPUT_DIR = Path(__file__).resolve().parent / "logs"  # 逐包 CSV 和汇总 JSON 的目录
THREAD_SWITCH_S = 0.00025  # 收发与日志线程的切换间隔，单位 s，与正常发布设置一致


def main():
    parser = argparse.ArgumentParser(description="仅测 UDP 网络，到时或按 Ctrl+C 保存逐包日志及汇总。")
    parser.add_argument("--local-ip", default=LOCAL_IP, help="主端网卡 IPv4 地址")
    parser.add_argument("--local-port", type=int, default=LOCAL_PORT, help="主端绑定端口")
    parser.add_argument("--peer-ip", default=PEER_IP, help="从端网卡 IPv4 地址")
    parser.add_argument("--peer-port", type=int, default=PEER_PORT, help="从端绑定端口")
    parser.add_argument("--seconds", type=float, default=CAPTURE_SECONDS, help="采集时长，单位 s")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR, help="逐包 CSV 和汇总 JSON 的目录")
    args = parser.parse_args()
    if not isfinite(args.seconds) or args.seconds <= 0:
        parser.error("--seconds 须为有限正数")
    period = SEND_PERIOD_MS / 1000
    sys.setswitchinterval(THREAD_SWITCH_S)
    payload = bytes(MASTER.size)
    try:
        with ExitStack() as resources:
            timer = None
            if sys.platform == "win32":
                from WinCycleTimer import WinCycleTimer
                timer = WinCycleTimer()
                resources.callback(timer.close)
            link = UdpClass((args.local_ip, args.local_port), (args.peer_ip, args.peer_port), timeout=0,
                            command_history_s=0, directory=args.output, kind=NETWORK_ONLY)
            link.start()
            resources.callback(link.close)
            session = link.start_publishing()
            print(f"仅测网络 {link.log.local} → {link.peer}，记录到 {link.log.path}", flush=True)
            deadline = perf_counter()
            end = deadline + args.seconds
            try:
                while deadline < end and not (link.error or link.log.error):
                    if timer:
                        timer.wait_until(deadline)
                    else:
                        sleep(max(0, deadline - perf_counter()))
                    if perf_counter() >= end:
                        break
                    link.send(payload, session)
                    deadline = next_deadline(deadline, period)
            except KeyboardInterrupt:
                print("已结束采集，正在保存。")
    except OSError as error:
        print(f"网络测量失败：{error}", file=sys.stderr)
        return 1
    if link.error or link.log.error:
        print(link.error or link.log.error, file=sys.stderr)
        return 1
    print(f"测量汇总 {link.log.summary_path}")
    print(link.log.summary_path.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
