"""在独立进程中计算调参响应指标，将三幅曲线绘制到共享图像。"""

from collections import deque
from ctypes import memmove
import json
from math import degrees
import sys

import numpy as np
from PyQt5 import QtCore, QtGui

from PidTuneSample import SAMPLE_RECORD, TunePoint


class ResponseMetrics:
    def __init__(self, band, dwell, tail_time):
        self.band, self.dwell, self.tail_time = band, dwell, tail_time
        self.last = None

    def add(self, point):
        now = point.feedback_at
        error = degrees(point.target - point.position)
        if self.last is None or point.event:
            self.start = point.sent_at
            self.delta = degrees(point.target - point.previous_target)
            self.drift = self.peak = self.overshoot = 0.0
            self.ten = self.ninety = self.inside = None
            self.tail = deque()
            self.baseline = point.previous_target
            # 阶跃前的末次反馈用作首个阈值跨越的插值端点。
            self.prev_time = self.start
            prev_pos = self.last.position if self.last is not None and point.event != "原位" else self.baseline
            self.prev_progress = degrees(prev_pos - self.baseline) / self.delta if self.delta else 0.0
        self.last = point
        self.drift = max(self.drift, abs(error))
        self.peak = max(self.peak, abs(point.command))
        self.tail.append((now, error))
        while len(self.tail) > 1 and self.tail[1][0] <= now - self.tail_time:
            self.tail.popleft()
        if self.delta:
            progress = degrees(point.position - self.baseline) / self.delta
            for level, name in ((0.1, "ten"), (0.9, "ninety")):
                if getattr(self, name) is None and progress >= level:
                    crossing = self.prev_time
                    if self.prev_progress < level:
                        crossing += (now - self.prev_time) * (level - self.prev_progress) / (progress - self.prev_progress)
                    setattr(self, name, crossing)
            self.overshoot = max(self.overshoot, (progress - 1.0) * 100)
            self.prev_time, self.prev_progress = now, progress
            if abs(error) > self.band:
                self.inside = None
            elif self.inside is None:
                self.inside = now

    def tail_error(self):
        """在末段时间窗口上对位置误差按采样时间积分。"""
        samples = iter(self.tail)
        t, e = next(samples)
        start = max(t, self.last.feedback_at - self.tail_time)
        area = 0.0
        for end, value in samples:
            if t < start:
                e += (value - e) * (start - t) / (end - t)
                t = start
            area += (end - t) * (value + e) / 2
            t, e = end, value
        return area / (t - start) if t > start else e

    def summary(self):
        p = self.last
        flags = [label for label, yes in (("PID", p.pid_limited), ("总力矩", p.output_limited),
                                          ("变化率", p.rate_limited)) if yes]
        extra = (f"最大锁定偏差 J{p.lock_node} {degrees(p.lock_error):.3f}°　"
                 f"PID 限幅前 / 后 {p.pid_raw:.3f} / {p.pid:.3f} N·m　"
                 f"限幅：{'、'.join(flags) if flags else '无'}")
        if self.delta:
            rise = f"{self.ninety-self.ten:.3f} s" if self.ninety is not None else "—"
            settled = (f"{self.inside-self.start:.3f} s" if self.inside is not None
                       and p.feedback_at - self.inside >= self.dwell else "未回稳")
            return (f"上升时间 {rise}　超调 {self.overshoot:.2f}%　回稳 {settled}　"
                    f"末段平均误差 {self.tail_error():+.3f}°\n{extra}")
        return (f"位置误差 {degrees(p.target-p.position):+.3f}°　最大漂移 {self.drift:.3f}°　"
                f"下发力矩峰值 {self.peak:.3f} N·m\n{extra}")


class ResponsePlot:
    colors = ("#155fad", "#cd5a20", "#168056", "#87449e")
    limit_colors = ("#b02736", "#9c5900", "#68449d")

    def __init__(self, legends, limit_labels):
        self.legends = legends
        self.limit_labels = limit_labels
        self.history = deque(maxlen=16384)
        # 镜像环形区使跨过末尾的时间窗也能直接切片，两份内容都只在新样本到达时写入。
        self.samples = np.empty((2 * self.history.maxlen, 13), dtype=np.float64)
        self.cursor = 0
        self.events = deque()
        self.zero = 0.0

    def clear(self):
        self.history.clear()
        self.events.clear()
        self.cursor = 0

    def add(self, point):
        if point.event == "原位" or not self.history:
            self.clear()
            self.zero = point.sent_at
        elif point.target != point.previous_target:
            self.zero = point.sent_at
        self.history.append(point)
        index = self.cursor
        self.samples[index] = (point.sent_at, point.feedback_at, degrees(point.target), degrees(point.position),
                               point.velocity_ref, point.velocity, point.gravity, point.pid, point.command,
                               point.torque, point.pid_limited, point.output_limited, point.rate_limited)
        capacity = self.history.maxlen
        self.samples[index + capacity] = self.samples[index]
        self.cursor = (index + 1) % capacity
        if point.event:
            self.events.append((point.sent_at, point.event))
        while self.events and self.events[0][0] < self.history[0].sent_at:
            self.events.popleft()

    @staticmethod
    def polygon(x, y):
        """直接填写 Qt 5 的连续 QPointF 坐标，每个点为两个双精度数。"""
        polygon = QtGui.QPolygonF(len(x))
        buffer = polygon.data().asarray(len(x) * 2 * np.dtype(np.float64).itemsize)
        coords = np.frombuffer(buffer, dtype=np.float64).reshape(-1, 2)
        coords[:, 0] = x
        coords[:, 1] = y
        return polygon

    def draw(self, painter, width, height):
        painter.fillRect(0, 0, width, height, QtGui.QColor("#fafbfd"))
        left, right = 63, width - 12
        row_height = (height - 19) / 3
        for row, (title, *labels) in enumerate(self.legends):
            y = round(row * row_height + 14)
            painter.setPen(QtGui.QColor("#233448"))
            painter.drawText(5, y, title)
            x = 116
            for color, label in zip(self.colors, labels):
                painter.setPen(QtGui.QColor(color))
                painter.drawLine(x, y-4, x+13, y-4)
                painter.drawText(x+17, y, label)
                x += 32 + painter.fontMetrics().horizontalAdvance(label)
            if row == 2:
                for label, color in zip(self.limit_labels, self.limit_colors):
                    painter.fillRect(x, y-8, 6, 6, QtGui.QColor(color))
                    painter.setPen(QtGui.QColor(color))
                    painter.drawText(x+9, y, label)
                    x += 18 + painter.fontMetrics().horizontalAdvance(label)
        if not self.history:
            return
        end = self.history[-1].feedback_at
        start = max(self.history[0].sent_at, end - 8.0)
        span = max(0.1, end - start)
        stop = self.cursor + self.history.maxlen
        history = self.samples[stop-len(self.history):stop]
        first = max(0, history[:, 1].searchsorted(start) - 1)
        values = history[first:]
        xs = left + (values[:, :2] - start) * ((right-left) / span)
        for row, columns in enumerate(((2, 3), (4, 5), (6, 7, 8, 9))):
            top, bottom = row * row_height + 21, (row + 1) * row_height - 4
            series = values[:, columns[0]:columns[-1]+1]
            low, high = series.min(), series.max()
            margin = max(0.001, (high-low) * 0.06)
            low, high = low - margin, high + margin
            painter.setPen(QtGui.QColor("#506174"))
            painter.drawText(3, round(top+9), f"{high:.3g}")
            painter.drawText(3, round(bottom), f"{low:.3g}")
            painter.setPen(QtGui.QColor("#d5dde6"))
            painter.drawRect(QtCore.QRectF(left, top, right-left, bottom-top))
            painter.save()
            painter.setClipRect(QtCore.QRectF(left, top-1, right-left+1, bottom-top+2))
            for col, color in zip(columns, self.colors):
                x = xs[:, int(col in (3, 5, 9))]
                y = bottom - (values[:, col]-low) * ((bottom-top)/(high-low))
                if col == 2:
                    # 只在目标变更处画垂线，保留采用阶跃的实际控制拍时刻。
                    changes = np.flatnonzero(np.diff(y)) + 1
                    x = np.r_[x[0], np.repeat(x[changes], 2), x[-1]]
                    y = np.r_[y[0], np.column_stack((y[changes-1], y[changes])).ravel(), y[-1]]
                painter.setPen(QtGui.QPen(QtGui.QColor(color), 1))
                painter.drawPolyline(self.polygon(x, y))
            if row == 2:
                for lane, color in enumerate(self.limit_colors):
                    marked = np.unique(np.rint(xs[values[:, 10+lane] != 0, 0]))
                    if marked.size:
                        painter.setPen(QtGui.QPen(QtGui.QColor(color), 2))
                        painter.drawPoints(self.polygon(marked, round(top + 2 + lane*3)))
            painter.restore()
        painter.setPen(QtGui.QPen(QtGui.QColor("#736d60"), 1, QtCore.Qt.DashLine))
        for stamp, label in self.events:
            if stamp >= start:
                x = round(left + (stamp-start) / span * (right-left))
                for row in range(3):
                    painter.drawLine(x, round(row*row_height+21), x, round((row+1)*row_height)-4)
                painter.drawText(min(x+3, right-65), 31, label)
        painter.setPen(QtGui.QColor("#506174"))
        for fraction in (0, 0.5, 1):
            x = round(left + fraction * (right-left))
            painter.drawText(min(width-62, x-30), height-18, 62, 18, QtCore.Qt.AlignCenter,
                             f"{start-self.zero+fraction*span:.2f} s")


def render_frames(app, stream, output):
    config = stream.readline()
    if not config:
        return
    legends, limit_labels, metric_settings = json.loads(config)
    plot = ResponsePlot(legends, limit_labels)
    metrics = ResponseMetrics(*metric_settings)
    generation = None
    missed = 0
    memory = QtCore.QSharedMemory()
    for line in stream:
        request = json.loads(line)
        data = stream.read(request["count"] * SAMPLE_RECORD.size)
        if len(data) != request["count"] * SAMPLE_RECORD.size:
            return
        if generation != request["generation"] or request["missed"]:
            plot.clear()
            metrics.last = None
            missed = 0 if generation != request["generation"] else missed
            generation = request["generation"]
            missed += request["missed"]
        events = dict(request["events"])
        for i, fields in enumerate(SAMPLE_RECORD.iter_unpack(data)):
            if fields[0] == request["node"]:
                point = TunePoint(*fields[:-1], events.get(i, ""), fields[-1])
                plot.add(point)
                metrics.add(point)

        if memory.key() != request["key"]:
            if memory.isAttached():
                memory.detach()
            memory.setKey(request["key"])
            if not memory.attach():
                raise RuntimeError(memory.errorString())
        w, h, pixel_w, pixel_h, ratio = request["size"]
        image = QtGui.QImage(pixel_w, pixel_h, QtGui.QImage.Format_ARGB32_Premultiplied)
        image.setDevicePixelRatio(ratio)
        font = app.font()
        font.fromString(request["font"])
        painter = QtGui.QPainter(image)
        painter.setFont(font)
        plot.draw(painter, w, h)
        painter.end()
        # QPainter 可使 QImage 与传入的外部缓冲分离；在应答前发布实际绘制的像素。
        memmove(int(memory.data()), int(image.constBits()), image.sizeInBytes())
        summary = metrics.summary() if metrics.last else "等待采样"
        if missed:
            summary += f"　显示缓冲跳过 {missed} 拍，指标自缺口后统计"
        # 一帧写完才通知，主进程复制图像并提交下一批后才会再次写入。
        output.write(json.dumps([generation, summary]).encode("ascii") + b"\n")
        output.flush()


if __name__ == "__main__":
    render_frames(QtGui.QGuiApplication(sys.argv), sys.stdin.buffer, sys.stdout.buffer)
