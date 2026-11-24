"""按独立节拍发布主端 UDP 指令并采样日志，界面只切换操作状态。"""

from contextlib import ExitStack
import sys
from threading import Condition, Thread
from time import perf_counter

from Robot_UR import Robot_UR
from UdpClass import encode_master


class TeleopPublisher(Thread):
    def __init__(self, period, timeout, auto_log, get_state, logger, report):
        super().__init__(name="DMArm-UDP-TX")
        self.period, self.timeout = period, timeout
        self.auto_log = auto_log
        self.get_state, self.logger, self.report = get_state, logger, report
        self.cv = Condition()
        self.link = self.timer = None
        self.publishing = self.closing = False
        self.log_mode = self.log_generation = None
        self.armed = True
        self.revision = 0

    def master_ready(self, state):
        return bool(state and state.opened and not state.faulted and not state.uncertain
                    and state.coherent and perf_counter() - state.feedback_at <= self.timeout
                    and all(fb and fb.status in (0, 1) for _, fb in state.motors))

    def changed(self):
        self.revision += 1
        self.cv.notify_all()
        if self.timer is not None:
            self.timer.wake()

    def set_mode(self, mode, frame=None):
        self.log_mode = mode
        self.log_generation = frame.generation if frame is not None else None
        if mode is None:
            self.logger.end_session()

    def stop_log(self):
        self.armed = not self.publishing
        self.set_mode(None)

    def set_link(self, link):
        with self.cv:
            if self.link is not None:
                if self.log_generation is not None:
                    self.stop_log()
                    self.report("UDP 已断开，日志记录已停止")
                self.set_publishing(False)
            self.link = link
            self.changed()

    def set_publishing(self, enabled):
        with self.cv:
            if enabled == self.publishing or (enabled and self.link is None):
                return
            self.publishing = enabled
            self.armed = True
            if not enabled:
                self.link.stop_publishing()
                if self.log_mode == "auto":
                    self.set_mode(None)
            self.changed()

    def set_recording(self, enabled):
        with self.cv:
            if enabled:
                self.set_mode("manual", self.link.fresh() if self.link else None)
            else:
                self.stop_log()
            self.changed()

    def log_failed(self, first_sample):
        with self.cv:
            if self.logger.first_sample is first_sample:
                self.stop_log()
                self.changed()

    def tick(self):
        with self.cv:
            link, publishing, revision = self.link, self.publishing, self.revision
            session = link.history[0] if link is not None else None
        state = self.get_state()
        packet = None
        if publishing:
            if not link.error and self.master_ready(state):
                q, dq = Robot_UR.joint_state(tuple(fb for _, fb in state.motors[:6]))
                data = encode_master(q, dq, state.motors[6][1].position)
                try:
                    if link.send(data, session):
                        packet = data
                except OSError as error:
                    with self.cv:
                        if self.link is link and self.publishing and session == link.history[0]:
                            link.error = f"UDP 发送失败：{error}"
                            self.set_publishing(False)
                    return
            else:
                with self.cv:
                    if revision != self.revision:
                        return
                    self.set_publishing(False)
                    revision = self.revision
        frame = link.fresh() if link else None
        with self.cv:
            # 停录、切换连接或发布会话后，正在发送的旧采样不能写进新文件。
            if revision != self.revision or self.closing:
                return
            if self.log_generation is not None and (frame is None or frame.generation != self.log_generation):
                self.stop_log()
                self.report("从端反馈断流，日志记录已停止")
                return
            teleoperating = packet is not None and state.streaming and state.gravity
            following = frame is not None and frame.teleop_state != 0
            if not teleoperating or (self.log_mode == "auto" and not following):
                self.armed = True
                if self.log_mode == "auto":
                    self.set_mode(None)
            elif self.auto_log and self.armed and following:
                self.armed = False
                if self.log_mode is None:
                    self.set_mode("auto", frame)
            if self.log_mode is not None:
                self.log_generation = frame.generation if frame is not None else None
                self.logger.record(state, packet, frame)

    def run(self):
        with ExitStack() as resources:
            with self.cv:
                if sys.platform == "win32":
                    from WinCycleTimer import WinCycleTimer
                    self.timer = WinCycleTimer()
                    resources.callback(self.timer.close)
            try:
                deadline = perf_counter()
                while True:
                    with self.cv:
                        while not (self.closing or self.publishing or self.log_mode):
                            self.cv.wait()
                            deadline = perf_counter()
                        if self.closing:
                            return
                        remaining = deadline - perf_counter()
                        if remaining > 0 and self.timer is None:
                            self.cv.wait(remaining)
                            continue
                    if remaining > 0:
                        self.timer.wait_until(deadline)
                        continue
                    self.tick()
                    # 按原定时轴进入下一个未来节拍，迟到时跳过错过的时隙。
                    deadline += (int((perf_counter() - deadline) / self.period) + 1) * self.period
            finally:
                with self.cv:
                    self.timer = None

    def close(self):
        with self.cv:
            self.closing = True
            self.set_publishing(False)
            self.changed()
        self.join()
        self.logger.close()
