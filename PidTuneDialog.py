"""单轴调参的非模态窗口、已确认控制拍曲线与响应指标。"""

from math import degrees
from time import perf_counter

from PyQt5 import QtCore, QtGui, QtWidgets

from ArmController import HOLD_FIELDS
from ui_PidTuneDialog import Ui_PidTuneDialog


CONSTANT_NAMES = {
    "pos_kp": "HOLD_POS_KP", "pos_ki": "HOLD_POS_KI", "pos_kd": "HOLD_POS_KD",
    "vel_kp": "HOLD_VEL_KP", "vel_ki": "HOLD_VEL_KI", "vel_kd": "HOLD_VEL_KD",
    "pos_i_limit": "HOLD_POS_I_LIMIT_RAD_S", "vel_i_limit": "HOLD_VEL_I_LIMIT_NM",
    "speed_limit": "HOLD_SPEED_LIMIT_RAD_S", "pid_limit": "HOLD_PID_TORQUE_LIMIT_NM",
    "torque_rate": "HOLD_TORQUE_RATE_NM_S",
}


class PidTuneDialog(QtWidgets.QDialog):
    def __init__(self, parent, limits, hold, band, dwell, tail_time, gains, output_limits):
        super().__init__(parent)
        self.ui = ui = Ui_PidTuneDialog()
        ui.setupUi(self)
        self.limits = limits
        self.filter_s, self.max_dt = hold["filter_s"], hold["max_dt"]
        self.gains, self.output_limits = gains, output_limits
        self.defaults = tuple(hold[name] for name in HOLD_FIELDS)
        self.columns = self.defaults
        self.node = ui.axisBox.currentIndex() + 1
        self.state = None
        self.start_pending = False
        self.after_exit = None
        self.end_submitted = False
        self.last_paint = 0.0
        self.plot = ui.plotArea
        self.plot.configure(band, dwell, tail_time)
        self.plot.summaryChanged.connect(ui.metricLabel.setText)
        self.plot.setToolTip(f"回稳带 ±{band:g}°，连续保持 {dwell:g} s；末段平均取最近 {tail_time:g} s。")
        ui.lockTable.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        for row in range(4):
            for col in range(6):
                cell = QtWidgets.QTableWidgetItem("—")
                cell.setTextAlignment(QtCore.Qt.AlignCenter)
                ui.lockTable.setItem(row, col, cell)
        for name in HOLD_FIELDS:
            getattr(ui, name).textEdited.connect(self.draft_changed)
        ui.axisBox.currentIndexChanged.connect(self.select_axis)
        ui.startButton.clicked.connect(self.start_axis)
        ui.endButton.clicked.connect(lambda: self.submit("tune_end"))
        ui.stepButton.clicked.connect(lambda: self.submit("tune_step", ui.stepBox.value()))
        ui.originButton.clicked.connect(lambda: self.submit("tune_step", 0.0))
        ui.applyButton.clicked.connect(self.apply)
        ui.copyButton.clicked.connect(self.copy_parameters)
        self.load_axis()
        self.refresh(None, (), False)

    def load_axis(self):
        i = self.node - 1
        for name, column in zip(HOLD_FIELDS, self.columns):
            getattr(self.ui, name).setText(f"{column[i]:.12g}")
        self.ui.stepBox.setRange(-self.limits.step_deg[i], self.limits.step_deg[i])
        self.ui.stepBox.setValue(0)
        low, high = self.limits.travel_deg[i]
        self.ui.limitsLabel.setText(f"重力倍率 {self.gains[i]:g}　MIT 总限矩 ±{self.output_limits[i]:g} N·m　"
                                    f"滤波 {self.filter_s:g} s　反馈时限 {self.max_dt:g} s")
        self.ui.stepBox.setToolTip(f"距原位 ±{self.limits.step_deg[i]:g}°，绝对目标 [{low:.3f}, {high:.3f}]°")
        self.ui.lockTable.setToolTip(f"五个 PV 轴的锁定容差为 ±{self.limits.lock_deg:g}°")
        self.draft_changed()

    def draft(self):
        values = {}
        for name, column in zip(HOLD_FIELDS, self.columns):
            text = getattr(self.ui, name).text()
            applied = column[self.node-1]
            values[name] = applied if text == f"{applied:.12g}" else float(text)
        return values

    def draft_changed(self):
        try:
            draft = self.draft()
        except ValueError:
            self.ui.draftLabel.setText("待填写有效数值")
            return
        changed = any(draft[n] != c[self.node-1] for n, c in zip(HOLD_FIELDS, self.columns))
        self.ui.draftLabel.setText("草稿待应用" if changed else ("当前生效值" if self.state else "启动配置"))

    def submit(self, action, *values):
        worker = self.parent().worker
        return bool(worker and worker.submit(action, (self.node, *values)))

    def start_axis(self):
        self.start_pending = self.submit("tune_start")
        if self.start_pending:
            self.ui.startButton.setEnabled(False)

    def apply(self):
        try:
            values = self.draft()
        except ValueError:
            self.ui.draftLabel.setText("请输入完整数值")
            return
        self.submit("tune_apply", values)

    def copy_parameters(self):
        lines = [f"{CONSTANT_NAMES[name]} = {tuple(map(float, column))}"
                 for name, column in zip(HOLD_FIELDS, self.columns)]
        QtWidgets.QApplication.clipboard().setText("\n".join(lines))
        self.ui.draftLabel.setText("已复制六轴参数")

    def reset_plot(self):
        self.plot.clear()
        self.ui.metricLabel.setText("等待采样")

    def set_node(self, node):
        self.node = node
        with QtCore.QSignalBlocker(self.ui.axisBox):
            self.ui.axisBox.setCurrentIndex(node-1)
        self.reset_plot()
        self.load_axis()

    def select_axis(self, index):
        if self.start_pending or (self.state and self.state.tune_node):
            with QtCore.QSignalBlocker(self.ui.axisBox):
                self.ui.axisBox.setCurrentIndex(self.node-1)
            self.request_exit(index+1)
        else:
            self.set_node(index+1)

    def request_exit(self, destination):
        self.after_exit = destination
        if self.state and self.state.tune_node and not self.end_submitted:
            self.end_submitted = self.submit("tune_end")

    def reject(self):
        self.close()

    def closeEvent(self, event):
        if self.start_pending or (self.state and self.state.tune_node):
            event.ignore()
            self.request_exit("关闭")
        else:
            event.accept()

    def refresh(self, state, points, ready):
        previous = self.state
        self.state = state
        if state and (not previous or (not previous.opened and state.opened)):
            self.columns = state.tune_parameters
            self.reset_plot()
            self.load_axis()
        elif previous and not state:
            self.columns = self.defaults
            self.load_axis()
        elif state and previous.tune_revision != state.tune_revision:
            self.columns = state.tune_parameters
            self.draft_changed()
        if state and state.tune_node and previous and not previous.tune_node:
            self.reset_plot()
        if not state or not state.busy or state.tune_node:
            self.start_pending = False
        active = state and state.tune_node
        if self.after_exit is not None:
            if active and not self.end_submitted and not state.busy and state.tune_phase != "锁回 PV":
                self.end_submitted = self.submit("tune_end")
            if not active and not self.start_pending and not (state and state.busy):
                dest = self.after_exit
                self.after_exit = None
                self.end_submitted = False
                if state and state.streaming and state.coherent and all(m == "PV" for m, _ in state.motors[:6]):
                    if dest == "关闭":
                        self.close()
                    else:
                        self.set_node(dest)
        running = bool(active == self.node and state.tune_phase == "调参中" and not state.busy
                       and self.after_exit is None)
        ui = self.ui
        ui.axisBox.setEnabled(not self.start_pending and self.after_exit is None)
        ui.startButton.setEnabled(bool(ready and not self.start_pending and not active))
        ui.copyButton.setEnabled(bool(state and state.opened))
        ui.endButton.setEnabled(bool(active and not state.busy and state.tune_phase != "锁回 PV"))
        for button in (ui.applyButton, ui.stepButton, ui.originButton):
            button.setEnabled(running)
        ui.phaseLabel.setText(state.tune_phase if active else "空闲")
        ui.messageLabel.setText(state.message if state else "设备未连接")
        if points:
            self.plot.append(points)
        if self.isVisible() and perf_counter() - self.last_paint >= 0.1:
            self.last_paint = perf_counter()
            self.update_feedback()
            self.plot.request_frame(self.node)

    def update_feedback(self):
        state = self.state
        origin = state.tune_origin if state else None
        measured = target = None
        for i in range(6):
            mode, fb = state.motors[i] if state and state.opened else ("—", None)
            lock = state.tune_locks[i] if state and state.tune_locks and mode == "PV" else None
            position = fb.position if fb else None
            error = position-lock if position is not None and lock is not None else None
            if i+1 == self.node:
                measured = position
                axis = state.hold_details[i] if state and state.tune_node and state.hold_details else None
                target = axis.target if axis else origin
            for row, val in enumerate((mode, lock, position, error)):
                cell = self.ui.lockTable.item(row, i)
                cell.setText(val if row == 0 else self.angle(val))
                if row == 3:
                    over = error is not None and abs(degrees(error)) > self.limits.lock_deg
                    cell.setBackground(QtGui.QColor("#ffe3a6") if over else QtGui.QBrush())
        self.ui.positionLabel.setText(f"原位 {self.angle(origin)} °　　实测 {self.angle(measured)} °　　"
                                      f"当前目标 {self.angle(target)} °")

    @staticmethod
    def angle(value):
        return "—" if value is None else f"{degrees(value):.3f}"
