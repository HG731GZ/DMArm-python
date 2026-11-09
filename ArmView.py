"""在同一基座坐标系正交投影两台 UR 的关节与末端坐标轴。"""

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets


class ArmView(QtWidgets.QWidget):
    # 水平面内斜视，基座 Z 轴在屏幕上朝上。
    camera = np.array([[.8, -.6, 0], [.3, .4, -.8660254]])

    def __init__(self, parent=None):
        super().__init__(parent)
        self.master = self.slave = None
        self.setMinimumSize(240, 165)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.fillRect(self.rect(), QtGui.QColor("#f4f8fa"))
        w, h = self.width(), self.height()
        # 给已到达的两臂取同一视窗，至少保留基座附近 0.4 m 的参照区域。
        bounds = [(-.20, -.20), (.20, .20)]
        for model in (self.master, self.slave):
            if model is not None:
                bounds.extend(model.transforms[:, :3, 3] @ self.camera.T)
                wrist = self.camera @ model.T60[:3, 3]
                bounds.extend((wrist - .075, wrist + .075))
        low, high = np.min(bounds, axis=0), np.max(bounds, axis=0)
        scale = min((w - 36) / (high[0] - low[0]), (h - 66) / (high[1] - low[1]))
        mid = (low + high) * scale / 2
        center = QtCore.QPointF(w / 2 - mid[0], h / 2 - mid[1])

        def project(point):
            x, y = self.camera @ point
            return center + QtCore.QPointF(x * scale, y * scale)

        def pen(color, width, style=QtCore.Qt.SolidLine):
            painter.setPen(QtGui.QPen(QtGui.QColor(color), width, style, QtCore.Qt.RoundCap))

        pen("#dce5e8", 1)
        for t in np.arange(-.4, .41, .1):
            painter.drawLine(project([t, -.4, 0]), project([t, .4, 0]))
            painter.drawLine(project([-.4, t, 0]), project([.4, t, 0]))

        for model, color, width, style in (
            (self.master, "#146b92", 5, QtCore.Qt.SolidLine),
            (self.slave, "#cc7428", 2, QtCore.Qt.DashLine),
        ):
            if model is None:
                continue
            pen(color, width, style)
            points = [project(p) for p in model.transforms[:, :3, 3]]
            for start, end in zip(points, points[1:]):
                painter.drawLine(start, end)
            painter.setBrush(QtGui.QColor(color))
            for point in points:
                painter.drawEllipse(point, width, width)
            for axis, tint in zip(model.T60[:3, :3].T, ("#cb493e", "#378348", "#4359bf")):
                pen(tint, 3 if model is self.master else 1.5, style)
                painter.drawLine(points[-1], project(model.T60[:3, 3] + axis * .075))

        if self.master is not None or self.slave is not None:
            pen("#526b75", 1)
            painter.drawText(center + QtCore.QPointF(6, 14), "基座")
        for x, label, color in ((10, "主端", "#146b92"), (74, "从端", "#cc7428")):
            pen(color, 3, QtCore.Qt.SolidLine if x == 10 else QtCore.Qt.DashLine)
            painter.drawLine(x, 13, x + 19, 13)
            painter.drawText(x + 24, 18, label)
        pen("#526b75", 1)
        if self.master is None and self.slave is None:
            painter.drawText(self.rect(), QtCore.Qt.AlignCenter, "等待姿态数据")
        else:
            painter.drawText(9, h - 7, "末端轴   X 红   Y 绿   Z 蓝")
