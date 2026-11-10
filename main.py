"""UR 同构主端窗口。选择 dmarm 解释器后直接运行本文件。"""

from math import degrees, pi
from pathlib import Path
import sys

from PyQt5 import QtCore, QtGui, QtWidgets

from ArmController import ArmController
from Robot_UR import Robot_UR
from ui_Form1 import Ui_Form1
from ui_ZeroDialog import Ui_ZeroDialog
from CanWorker import ALL, JOINTS, CanWorker
from USBCANFD import USBCANFD

PROJECT_DIR = Path(__file__).resolve().parent
CANFD_DEVICE_INDEX = 0  # ZLG 设备索引，从 0 起
CANFD_CHANNEL_INDEX = 0  # CAN FD 通道索引，从 0 起
CANFD_USE_TX_QUEUE = True  # CAN FD 周期发送方式，True 为队列发送，False 为直接批发
ZLG_SDK_DIR_WIN = PROJECT_DIR / "zlgcan_x64"  # 包含 DLL 和 kerneldlls 的厂商目录
ZLG_SDK_DIR_LINUX = PROJECT_DIR / "zlgcan_linux_x64"  # Linux x86_64 的厂商运行库目录
UDP_LOCAL_IP = "192.168.3.5"  # 主端监听地址
UDP_LOCAL_PORT = 6005  # 主端监听端口
UDP_REMOTE_IP = "192.168.3.14"  # 从端目的地址
UDP_REMOTE_PORT = 5005  # 从端目的端口
QUERY_TIMEOUT_S = 0.5  # 单条只读查询等待应答的时限，单位 s
POSE_PERIOD_S = 0.10  # 连续读取失能姿态的轮间隔，单位 s
UI_REFRESH_MS = 10  # 界面快照刷新间隔，单位 ms
CAN_PERIOD_S = 0.001  # CAN FD 周期目标间隔，单位 s；每轮收齐所选节点反馈后再发下一轮
PYTHON_SWITCH_INTERVAL_S = 0.00025  # Python 线程切换间隔，单位 s，缩短 CAN 回包就绪后等待界面线程的时间
PV_HOLD_SPEED_RAD_S = 0.05  # P3 原位持位限速，单位 rad/s
STOP_SPEED_RAD_S = 0.05  # 持位后连续三帧的停止速度阈值，单位 rad/s
STOP_SAMPLE_PERIOD_S = 0.010  # PV 停止确认的采样间隔，单位 s
STOP_TIMEOUT_S = 1.0  # 持位停止确认时限，单位 s

# 按 J1～J6 排列。把某一项改为 False，该轴 MIT 前馈力矩即为零，重启主窗生效。
GRAVITY_AXES = (True, True, True, True, True, True)
GRAVITY_GAIN = (1.1, 1.1, 1.1, 1.2, 1.1, 1.1)  # 原 C# 电机侧重力矩倍率，无单位
GRAVITY_RAMP_S = 1.0  # 开启补偿至设定倍率的渐入时间，单位 s
UR_HOME_RAD = (-pi/2, -pi/2, -pi/2, -pi/2, pi/2, 0)  # DH 预设关节角，单位 rad
PV_SPEED_LIMIT_RAD_S = 0.05  # 回预设位的首次调试限速，单位 rad/s；原 C# 为 0.5
HOME_TOLERANCE_RAD = 0.02  # 六轴到位角度容差，单位 rad


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = ui = Ui_Form1()
        ui.setupUi(self)
        self.last_feedback = None
        self.worker = None
        self.closing_window = False
        ui.channelLabel.setText(f"设备 {CANFD_DEVICE_INDEX}  /  通道 {CANFD_CHANNEL_INDEX}  ·  1 / 5 Mbit/s")
        ui.localEndpoint.setText(f"本机  {UDP_LOCAL_IP}:{UDP_LOCAL_PORT}")
        ui.remoteAddress.setText(f"从端  {UDP_REMOTE_IP}")
        ui.remotePort.setValue(UDP_REMOTE_PORT)
        for table in (ui.feedbackTable, ui.anglesTable):
            table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        ui.anglesTable.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        self.timer = QtCore.QTimer(self, interval=UI_REFRESH_MS)
        self.timer.timeout.connect(self.refresh)
        ui.openButton.clicked.connect(self.toggle_device)
        ui.readAllButton.clicked.connect(lambda: self.command("read"))
        ui.trackPose.clicked.connect(lambda on: self.command("pose_on" if on else "pose_off"))
        ui.enableButton.clicked.connect(lambda: self.command("enable", ALL))
        ui.disableButton.clicked.connect(lambda: self.command("disable_all", ALL))
        ui.clearButton.clicked.connect(lambda: self.command("clear", ALL))
        ui.sendButton.clicked.connect(self.toggle_sending)
        ui.mitButton.clicked.connect(lambda: self.command("mit"))
        ui.pvButton.clicked.connect(lambda: self.command("pv"))
        ui.gravityButton.clicked.connect(self.toggle_gravity)
        ui.homeButton.clicked.connect(lambda: self.command("home"))
        ui.toolMitButton.clicked.connect(lambda: self.command("tool_mit"))
        ui.toolPvtButton.clicked.connect(lambda: self.command("tool_pvt"))
        self.lock_buttons = [getattr(ui, f"lockJ{n}") for n in JOINTS]
        for node, button in zip(JOINTS, self.lock_buttons):
            button.clicked.connect(lambda checked, n=node: self.toggle_lock(n))
        self.zero_dialog = QtWidgets.QDialog(self)
        self.zero_ui = Ui_ZeroDialog()
        self.zero_ui.setupUi(self.zero_dialog)
        ui.zeroButton.clicked.connect(self.open_zero_dialog)
        self.zero_ui.jointBox.currentIndexChanged.connect(
            lambda index: self.zero_ui.applyButton.setEnabled(index >= 0))
        self.zero_ui.cancelButton.clicked.connect(self.zero_dialog.reject)
        self.zero_ui.applyButton.clicked.connect(self.zero_dialog.accept)
        self.zero_dialog.accepted.connect(self.set_selected_zero)
        self.tool_fields = (ui.toolValue0, ui.toolValue1, ui.toolValue2, ui.toolValue3,
                            ui.toolValue4, ui.toolValue5, ui.toolValue6, ui.toolValue7)

    def toggle_device(self):
        if self.worker is None:
            self.open_device()
        else:
            self.close_device()

    def toggle_sending(self):
        self.command("stop" if self.worker.state.streaming else "start")

    def toggle_gravity(self):
        self.command("gravity_off" if self.worker.state.gravity else "gravity_on")

    def toggle_lock(self, node):
        mode, _ = self.worker.state.motors[node - 1]
        self.command("unlock" if mode == "PV" else "lock", (node,))

    def open_zero_dialog(self):
        self.zero_ui.jointBox.setCurrentIndex(-1)
        self.zero_dialog.show()
        self.zero_dialog.raise_()
        self.zero_dialog.activateWindow()

    def set_selected_zero(self):
        index = self.zero_ui.jointBox.currentIndex()
        if index >= 0 and self.ui.zeroButton.isEnabled():
            self.command("zero", (JOINTS[index],))

    def command(self, action, nodes=()):
        if self.worker is not None:
            self.worker.submit(action, nodes)
            self.refresh()

    def open_device(self):
        bus = USBCANFD(ZLG_SDK_DIR_WIN, ZLG_SDK_DIR_LINUX, CANFD_DEVICE_INDEX,
                       CANFD_CHANNEL_INDEX, QUERY_TIMEOUT_S)
        self.worker = CanWorker(
            bus, self.ui.listenOnly.isChecked(), period=CAN_PERIOD_S, speed_limit=PV_HOLD_SPEED_RAD_S,
            stop_speed=STOP_SPEED_RAD_S, stop_timeout=STOP_TIMEOUT_S, pose_period=POSE_PERIOD_S,
            stop_sample_period=STOP_SAMPLE_PERIOD_S, queued_cycles=CANFD_USE_TX_QUEUE,
            controller=ArmController(UR_HOME_RAD, PV_SPEED_LIMIT_RAD_S, GRAVITY_GAIN,
                                     GRAVITY_AXES, GRAVITY_RAMP_S, HOME_TOLERANCE_RAD),
        )
        self.ui.frameLog.clear()
        self.worker.start()
        self.timer.start()
        self.refresh()

    def refresh(self):
        worker = self.worker
        if worker is None:
            return
        state, messages = worker.snapshot()
        if messages:
            self.ui.frameLog.appendPlainText("\n".join(messages))
        feedback = state.motors if state.opened and not state.faulted else (("—", None),) * len(ALL)
        if feedback != self.last_feedback:
            self.show_feedback(feedback)
            self.last_feedback = feedback
        self.ui.rxCount.setText(f"TX {state.tx}  RX {state.rx}  失败 {state.failed}")
        self.ui.controlStatus.setText(
            f"周期 {state.cycles} 次 · {state.hz:.1f} Hz  |  {state.message}")
        self.statusBar().showMessage(state.message)
        ui = self.ui
        alive = worker.is_alive()
        control = alive and state.opened and not worker.listen_only
        ready = control and not state.busy and not state.uncertain and not state.faulted
        idle = ready and not state.owned and not state.streaming
        ui.trackPose.setEnabled(idle and state.first_read)
        ui.trackPose.setChecked(state.tracking)
        idle = idle and not state.tracking
        ui.openButton.setText("关闭设备" if alive else "打开设备")
        ui.openButton.setEnabled(not alive or state.opened)
        ui.listenOnly.setEnabled(not alive)
        ui.readAllButton.setEnabled(idle)
        ui.enableButton.setEnabled(idle and state.coherent and state.verified == set(ALL))
        ui.clearButton.setEnabled(idle)
        ui.disableButton.setEnabled(control)
        ui.sendButton.setText("停止发送" if state.streaming else "开始发送")
        ui.sendButton.setEnabled((control and state.streaming) or (ready and state.owned == set(ALL)))
        powered = ready and state.coherent and state.owned == set(ALL)
        for btn in (ui.mitButton, ui.pvButton):
            btn.setEnabled((idle and set(JOINTS) <= state.verified) or (powered and not state.homing))
        can_zero = powered and not state.streaming
        ui.zeroButton.setEnabled(can_zero)
        if self.zero_dialog.isVisible() and not can_zero:
            self.zero_dialog.reject()
        active = ready and state.streaming and state.coherent
        six_active = active and set(JOINTS) <= state.owned
        modes = [mode for mode, _ in state.motors[:6]]
        ui.gravityButton.setEnabled(six_active)
        ui.gravityButton.setText("关闭重力补偿" if state.gravity else "开启重力补偿")
        ui.homeButton.setEnabled(six_active and not state.homing)
        for button in (ui.toolMitButton, ui.toolPvtButton):
            button.setEnabled(idle and 7 in state.verified)
        for node, button, mode, value in zip(JOINTS, self.lock_buttons, modes, state.commands):
            button.setEnabled(powered and mode in ("MIT", "PV") and not state.homing)
            button.setText(f"J{node} 释放" if mode == "PV" and node in state.owned else f"J{node} 锁定")
            unit = "rad" if mode == "PV" else "N·m"
            self.ui.anglesTable.item(node - 1, 5).setText("—" if value is None else f"{value:.3f} {unit}")
        if state.uncertain:
            ui.deviceStatus.setText("失能未确认")
        elif state.faulted:
            ui.deviceStatus.setText("设备通信异常")
        elif state.opened:
            ui.deviceStatus.setText("已连接 · 只听" if worker.listen_only else "已连接")
        else:
            ui.deviceStatus.setText("正在打开" if alive else "未连接")
        if not alive:
            worker.join()
            self.timer.stop()
            self.worker = None
            if self.closing_window:
                QtCore.QTimer.singleShot(0, self.close)

    def show_feedback(self, motors):
        joints = [fb for _, fb in motors[:6]]
        model = Robot_UR.from_feedback(joints) if all(joints) else None
        view = self.ui.armView
        view.master = model
        view.update()
        table = self.ui.anglesTable
        for i in range(6):
            values = ("—", "—", "—") if model is None else (
                f"{degrees(model.Angle[i]):.2f}", f"{model.Velocity[i]:.3f}", f"{model.Tau_G_Motor[i]:.3f}")
            for col, value in zip((1, 3, 4), values):
                table.item(i, col).setText(value)
        hot = QtGui.QBrush(QtGui.QColor("#ffc4c4"))
        for node, (mode_text, fb) in enumerate(motors, 1):
            if fb is None:
                fields = ["—"] * 6
            else:
                fields = [f"{fb.position:.4f}", f"{fb.velocity:.4f}", f"{fb.torque:.4f}",
                          str(fb.mos), str(fb.rotor), fb.status_text]
            if node <= 6:
                for column, value in enumerate([*fields, mode_text], 1):
                    self.ui.feedbackTable.item(node - 1, column).setText(value)
                for column in (4, 5):
                    over = fb is not None and (fb.mos if column == 4 else fb.rotor) > 60
                    self.ui.feedbackTable.item(node - 1, column).setBackground(hot if over else QtGui.QBrush())
            else:
                fields[0] = f"{degrees(fb.position):.2f}" if fb else "—"
                fields.insert(3, f"{fb.grip_force:.2f}" if fb else "—")
                for label, value in zip(self.tool_fields, [*fields, mode_text]):
                    label.setText(value)
                for label, value in zip(self.tool_fields[4:6], (fb.mos, fb.rotor) if fb else (0, 0)):
                    label.setStyleSheet("background-color: #ffc4c4" if value > 60 else "")

    def set_slave_angles(self, q):
        """接收已解码的从端六关节角（rad）；断开时用 None 清除。"""
        model = Robot_UR(q) if q is not None else None
        self.ui.armView.slave = model
        self.ui.armView.update()
        for i in range(6):
            value = f"{degrees(model.Angle[i]):.2f}" if model is not None else "—"
            self.ui.anglesTable.item(i, 2).setText(value)

    def close_device(self):
        if self.worker is None:
            return
        state = self.worker.state
        if state.uncertain:
            reply = QtWidgets.QMessageBox.question(
                self, "失能未确认", "请支撑并物理停机。确认现场已切断动力后释放适配器？",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No)
            if reply != QtWidgets.QMessageBox.Yes:
                self.closing_window = False
                return
            self.command("release")
        else:
            self.command("close")

    def closeEvent(self, event):
        if self.worker is not None:
            event.ignore()
            self.closing_window = True
            self.close_device()
        else:
            super().closeEvent(event)


def main():
    sys.setswitchinterval(PYTHON_SWITCH_INTERVAL_S)
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
