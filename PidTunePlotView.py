"""主窗口侧的绘图进程连接与已完成图像显示。"""

from collections import deque
import json
from math import ceil
from pathlib import Path
import sys
from uuid import uuid4

from PyQt5 import QtCore, QtGui, QtWidgets


class TunePlotView(QtWidgets.QWidget):
    summaryChanged = QtCore.pyqtSignal(str)

    def __init__(self, parent):
        super().__init__(parent)
        self.process = QtCore.QProcess(self)
        self.process.started.connect(self.send_config)
        self.process.readyReadStandardOutput.connect(self.receive_frame)
        self.process.errorOccurred.connect(self.process_error)
        self.process.finished.connect(self.process_finished)
        QtWidgets.QApplication.instance().aboutToQuit.connect(self.shutdown)
        self.pending = deque(maxlen=16384)
        self.generation = 0
        self.missed = 0
        self.inflight = None
        self.frame_key = None
        self.image = QtGui.QImage()
        self.memory = None
        self.failed = self.closing = False

    def configure(self, band, dwell, tail_time):
        legends = [self.property(name).split("|") for name in ("positionLegend", "velocityLegend", "torqueLegend")]
        self.config = [legends, self.property("limitLegend").split("|"), (band, dwell, tail_time)]

    def showEvent(self, event):
        super().showEvent(event)
        if self.process.state() == QtCore.QProcess.NotRunning and not self.closing:
            self.failed = False
            self.process.start(sys.executable, ["-u", str(Path(__file__).with_name("PidTuneRenderer.py"))])

    def send_config(self):
        self.process.write(json.dumps(self.config).encode("ascii") + b"\n")
        self.frame_key = None

    def clear(self):
        self.generation += 1
        self.pending.clear()
        self.missed = 0
        self.frame_key = None
        self.image = QtGui.QImage()
        self.update()

    def append(self, points):
        self.missed += max(0, len(self.pending) + len(points) - self.pending.maxlen)
        self.pending.extend(points)

    def request_frame(self, node):
        if self.failed or self.inflight or self.process.state() != QtCore.QProcess.Running:
            return
        w, h, ratio = self.width(), self.height(), self.devicePixelRatioF()
        pw, ph = ceil(w*ratio), ceil(h*ratio)
        size = (w, h, pw, ph, ratio)
        font = self.font()
        if font.pointSizeF() > 0:
            font.setPixelSize(round(font.pointSizeF() * self.logicalDpiY() / 72))
        key = (self.generation, size, font.toString())
        if self.frame_key == key and not self.pending:
            return
        required = pw * ph * 4
        if self.memory is None or required > self.memory.size():
            self.memory = QtCore.QSharedMemory("DMArmPlot" + uuid4().hex)
            if not self.memory.create(required):
                self.fail(self.memory.errorString())
                return
        # 每次只投递一帧。绘图落后时限制本次打包量，停止和失能按钮仍由主事件循环处理。
        points = [self.pending.popleft() for _ in range(min(len(self.pending), 1024))]
        data = b"".join(p.pack() for p in points)
        events = [(i, p.event) for i, p in enumerate(points) if p.event]
        header = dict(generation=self.generation, node=node, count=len(points), events=events,
                      missed=self.missed, key=self.memory.key(), size=size, font=key[2])
        self.inflight = key
        self.process.write(json.dumps(header).encode("ascii") + b"\n" + data)
        self.missed = 0

    def receive_frame(self):
        if not self.process.canReadLine():
            return
        generation, summary = json.loads(bytes(self.process.readLine()))
        key, self.inflight = self.inflight, None
        if generation != self.generation or self.closing:
            return
        _, (_, _, pw, ph, ratio), _ = key
        frame = QtGui.QImage(self.memory.constData(), pw, ph, 4*pw, QtGui.QImage.Format_ARGB32_Premultiplied)
        self.image = frame.copy()
        self.image.setDevicePixelRatio(ratio)
        self.frame_key = key
        self.summaryChanged.emit(summary)
        self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.fillRect(event.rect(), QtGui.QColor("#fafbfd"))
        painter.drawImage(0, 0, self.image)

    def fail(self, reason):
        self.failed = True
        self.inflight = None
        self.summaryChanged.emit(f"曲线显示异常：{reason}")

    def process_error(self, error):
        if not self.closing:
            self.fail(self.process.errorString())

    def process_finished(self, code, status):
        if not self.closing:
            diagnostic = bytes(self.process.readAllStandardError()).decode("utf-8", errors="replace").strip()
            self.fail(diagnostic.splitlines()[-1] if diagnostic else f"绘图进程已退出（{code}）")

    def shutdown(self):
        self.closing = True
        if self.process.state() != QtCore.QProcess.NotRunning:
            self.process.closeWriteChannel()
            if not self.process.waitForFinished(1500):
                self.process.kill()
                self.process.waitForFinished(1000)
        self.memory = None
