"""UR 同构主端窗口。选择 dmarm 解释器后直接运行本文件。"""

from math import degrees, pi
from datetime import datetime
from pathlib import Path
import sys
from time import perf_counter

from PyQt5 import QtCore, QtGui, QtWidgets

from ArmController import ArmController, HoldParameters
from Robot_UR import Robot_UR
from ui_Form1 import Ui_Form1
from ui_ZeroDialog import Ui_ZeroDialog
from CanWorker import ALL, JOINTS, CanWorker, TuneLimits
from PidTuneDialog import PidTuneDialog
from DMMotor import LIMITS, MOTOR_MODELS
from USBCANFD import USBCANFD
from UdpClass import UdpClass, encode_master

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
UDP_SEND_PERIOD_MS = 5  # 主端命令发送间隔，单位 ms
UDP_TIMEOUT_S = 0.15  # 从端状态新鲜时限，单位 s；运动联调前按现场确认
QUERY_TIMEOUT_S = 0.5  # 单条只读查询等待应答的时限，单位 s
POSE_PERIOD_S = 0.10  # 连续读取失能姿态的轮间隔，单位 s
UI_REFRESH_MS = 10  # 界面快照刷新间隔，单位 ms
CAN_CYCLE_DELAY_S = 0  # 本轮 CAN FD 反馈收齐后到下一轮的等待时间，单位 s；0 表示立即继续
TOOL_UPDATE_CYCLES = 50  # 每隔多少轮 CAN FD 周期发送一次夹钳指令，取正整数；1 表示每轮都发
PYTHON_SWITCH_INTERVAL_S = 0.00025  # Python 线程切换间隔，单位 s，缩短 CAN 回包就绪后等待界面线程的时间

# 按 J1～J6 排列。把某一项改为 False，该轴 MIT 前馈力矩即为零，重启主窗生效。
GRAVITY_AXES = (True, True, True, True, True, True)
GRAVITY_GAIN = (1.1, 1.1, 1.1, 1.2, 1.1, 1.1)  # 原 C# 电机侧重力矩倍率，无单位
# 双环参数按 J1～J6 排列。J1、J2 沿用 J3 负向 0.5° 阶跃的初调参数，J4、J5 沿用 J6 的初调参数。
HOLD_POS_KP = (8.0, 8.0, 8.0, 5.5, 5.5, 5.5)  # 位置环比例，单位 1/s
HOLD_POS_KI = (0.0,) * 6  # 位置环积分，单位 1/s²
HOLD_POS_KD = (0.0,) * 6  # 位置环速度反馈增益，无量纲
HOLD_VEL_KP = (8.0, 8.0, 8.0, 1.8, 1.8, 1.8)  # 速度环比例，单位 N·m·s/rad
HOLD_VEL_KI = (10.0, 10.0, 10.0, 2.0, 2.0, 2.0)  # 速度环积分，单位 N·m/rad
HOLD_VEL_KD = (0.0,) * 6  # 速度环加速度反馈增益，单位 N·m·s²/rad
# 六关节 MIT 指令的软件力矩上限取各自协议 T_MAX 的 95%，单位 N·m。
MOTOR_OUTPUT_TORQUE_LIMIT_NM = tuple(0.95 * LIMITS[model][2] for model in MOTOR_MODELS[:6])
# PID 分量取各轴软件总上限的 50%，单位 N·m；叠加重力前馈后再限制总力矩。
HOLD_PID_TORQUE_LIMIT_NM = tuple(0.5 * limit for limit in MOTOR_OUTPUT_TORQUE_LIMIT_NM)
HOLD_SPEED_LIMIT_RAD_S = (0.1,) * 6  # 位置环目标速度上限，单位 rad/s
HOLD_TORQUE_RATE_NM_S = (4.0, 4.0, 4.0, 12.0, 12.0, 12.0)  # 合成力矩变化率上限，单位 N·m/s
HOLD_POS_I_LIMIT_RAD_S = (0.0,) * 6  # 位置环积分项上限，单位 rad/s
HOLD_VEL_I_LIMIT_NM = (1.30, 1.30, 1.30, 0.10, 0.10, 0.10)  # 速度环积分项上限，单位 N·m
HOLD_D_FILTER_S = 0.02  # 加速度低通时间常数，单位 s
HOLD_MAX_DT_S = 0.05  # 完整六轴反馈时限，单位 s；联调时按含夹钳轮的实测周期填写
UR_HOME_RAD = (-pi/2, -pi/2, -pi/2, -pi/2, pi/2, 0)  # DH 预设关节角，单位 rad
PV_SPEED_LIMIT_RAD_S = 0.5  # PV 位置指令的速度上限，单位 rad/s

# 按 J1～J6 的原始电机坐标配置。绝对目标区间取各型号现用位置量程，单位 °。
TUNE_STEP_LIMIT_DEG = (30.0,) * 6  # 距本次原位的最大阶跃幅度，单位 °
TUNE_TRAVEL_DEG = tuple((-degrees(LIMITS[m][0]), degrees(LIMITS[m][0])) for m in MOTOR_MODELS[:6])
TUNE_LOCK_ERROR_DEG = 0.5  # 其余五轴 PV 就位及运行偏差容差，单位 °
TUNE_SETTLE_BAND_DEG = 0.1  # 回稳绝对位置误差带，单位 °
TUNE_SETTLE_TIME_S = 0.25  # 误差连续处于回稳带内的最短时间，单位 s
TUNE_TAIL_TIME_S = 0.5  # 末段平均位置误差的时间窗，单位 s


def hold_settings():
    return dict(pos_kp=HOLD_POS_KP, pos_ki=HOLD_POS_KI, pos_kd=HOLD_POS_KD,
                vel_kp=HOLD_VEL_KP, vel_ki=HOLD_VEL_KI, vel_kd=HOLD_VEL_KD,
                pid_limit=HOLD_PID_TORQUE_LIMIT_NM, speed_limit=HOLD_SPEED_LIMIT_RAD_S,
                torque_rate=HOLD_TORQUE_RATE_NM_S, pos_i_limit=HOLD_POS_I_LIMIT_RAD_S,
                vel_i_limit=HOLD_VEL_I_LIMIT_NM, filter_s=HOLD_D_FILTER_S, max_dt=HOLD_MAX_DT_S)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = ui = Ui_Form1()
        ui.setupUi(self)
        self.last_feedback = None
        self.worker = None
        self.udp = None
        self.displayed_frame = None
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
        self.timer.start()
        self.send_timer = QtCore.QTimer(self, interval=UDP_SEND_PERIOD_MS)
        self.send_timer.setTimerType(QtCore.Qt.PreciseTimer)
        self.send_timer.timeout.connect(self.publish_master)
        ui.publishMaster.toggled.connect(self.toggle_publish)
        ui.udpButton.clicked.connect(self.toggle_udp)
        ui.syncButton.clicked.connect(lambda: self.command("sync", (self.udp,)))
        ui.followButton.clicked.connect(self.toggle_follow)
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
        ui.pidHoldButton.clicked.connect(self.toggle_hold)
        self.tune_limits = TuneLimits(TUNE_STEP_LIMIT_DEG, TUNE_TRAVEL_DEG, TUNE_LOCK_ERROR_DEG)
        self.tune_dialog = PidTuneDialog(self, self.tune_limits, hold_settings(),
                                         TUNE_SETTLE_BAND_DEG, TUNE_SETTLE_TIME_S, TUNE_TAIL_TIME_S,
                                         tuple(g * on for g, on in zip(GRAVITY_GAIN, GRAVITY_AXES)),
                                         tuple(min(t, LIMITS[m][2]) for t, m in
                                               zip(MOTOR_OUTPUT_TORQUE_LIMIT_NM, MOTOR_MODELS)))
        ui.pidTuneButton.clicked.connect(self.open_tuning)
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

    def toggle_udp(self):
        if self.udp:
            self.close_udp()
            return
        try:
            self.udp = UdpClass((UDP_LOCAL_IP, UDP_LOCAL_PORT),
                                (UDP_REMOTE_IP, self.ui.remotePort.value()), UDP_TIMEOUT_S)
        except OSError as error:
            self.ui.udpStatus.setText(f"打开 UDP 失败：{error}")
            return
        self.udp.start()
        self.ui.remotePort.setEnabled(False)
        self.ui.udpButton.setText("关闭 UDP")
        self.refresh()

    def close_udp(self):
        self.ui.publishMaster.setChecked(False)
        self.udp.close()
        self.udp = None
        self.displayed_frame = None
        self.set_slave_angles(None)
        self.ui.udpButton.setText("打开 UDP")
        self.ui.remotePort.setEnabled(True)
        self.ui.udpStatus.setText("UDP 未打开")
        self.refresh_udp()

    def toggle_publish(self, checked):
        if checked:
            self.send_timer.start()
        else:
            self.send_timer.stop()

    @staticmethod
    def master_ready(state):
        return bool(state and state.opened and not state.faulted and not state.uncertain
                    and state.coherent and perf_counter() - state.feedback_at <= QUERY_TIMEOUT_S
                    and all(fb and fb.status in (0, 1) for _, fb in state.motors))

    def publish_master(self):
        state = self.worker.state if self.worker else None
        if not (self.ui.publishMaster.isChecked() and self.udp and self.master_ready(state)):
            self.ui.publishMaster.setChecked(False)
            return
        joints = [fb for _, fb in state.motors[:6]]
        q = Robot_UR.motor2dh([fb.position for fb in joints])
        dq = [fb.velocity / ratio for fb, ratio in zip(joints, Robot_UR.ratio)]
        try:
            self.udp.send(encode_master(q, dq, state.motors[6][1].position))
        except OSError as error:
            self.close_udp()
            self.ui.udpStatus.setText(f"UDP 发送失败：{error}")

    def toggle_follow(self):
        if self.worker.state.remote_action:
            self.command("remote_off")
        else:
            self.command("follow", (self.udp,))

    def refresh_udp(self, state=None):
        link = self.udp
        publish_ready = bool(link and self.master_ready(state))
        self.ui.publishMaster.setEnabled(publish_ready)
        if not publish_ready:
            self.ui.publishMaster.setChecked(False)
        frame = link.fresh() if link else None
        if frame is not self.displayed_frame:
            self.set_slave_angles(frame.q if frame else None)
            self.displayed_frame = frame
        if link:
            last = link.latest
            status = "等待有效从端帧"
            if last:
                stamp = datetime.fromtimestamp(last.wall_time).strftime("%H:%M:%S.%f")[:-3]
                grip = ", ".join(f"{v:.3f}" for v in last.grip)
                status = (f"{link.peer[0]}:{link.peer[1]}  末帧 {stamp}  "
                          f"{'在线' if frame else '断流'}  mode {last.mode}  grip [{grip}]")
            self.ui.udpStatus.setText(f"{status}  RX {link.rx}  TX {link.tx}  丢弃 {link.rejected}")
            if link.error:
                self.close_udp()
                self.ui.udpStatus.setText(f"UDP 接收失败：{link.error}")
                return
        active = bool(state and state.remote_action)
        ready = bool(frame and state and not state.tune_node and state.streaming and state.hold_state == "空闲"
                     and state.coherent and not state.busy and not state.uncertain and not state.faulted
                     and set(JOINTS) <= state.owned)
        self.ui.syncButton.setEnabled(ready and not active)
        self.ui.followButton.setEnabled(active or ready)
        self.ui.followButton.setText("停止遥操作" if active else "持续跟随从端")

    def toggle_sending(self):
        self.command("stop" if self.worker.state.streaming else "start")

    def toggle_gravity(self):
        self.command("gravity_off" if self.worker.state.gravity else "gravity_on")

    def toggle_hold(self):
        self.command("hold_off" if self.worker.state.hold_state == "持位" else "hold_on")

    def open_tuning(self):
        self.tune_dialog.show()
        self.refresh()
        self.tune_dialog.raise_()
        self.tune_dialog.activateWindow()

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
        try:
            bus = USBCANFD(ZLG_SDK_DIR_WIN, ZLG_SDK_DIR_LINUX, CANFD_DEVICE_INDEX, CANFD_CHANNEL_INDEX,
                           QUERY_TIMEOUT_S, joint_torque_limits=MOTOR_OUTPUT_TORQUE_LIMIT_NM)
            hold = HoldParameters(**hold_settings())
            controller = ArmController(UR_HOME_RAD, GRAVITY_GAIN, GRAVITY_AXES, hold, [bus.motors[n] for n in JOINTS])
        except ValueError as error:
            self.ui.controlStatus.setText(f"参数错误：{error}")
            return
        self.worker = CanWorker(
            bus, self.ui.listenOnly.isChecked(), cycle_delay=CAN_CYCLE_DELAY_S, speed_limit=PV_SPEED_LIMIT_RAD_S,
            pose_period=POSE_PERIOD_S, queued_cycles=CANFD_USE_TX_QUEUE,
            tool_update_cycles=TOOL_UPDATE_CYCLES,
            controller=controller,
            tune_limits=self.tune_limits,
        )
        self.ui.frameLog.clear()
        self.worker.start()
        self.timer.start()
        self.refresh()

    def refresh(self):
        worker = self.worker
        if worker is None:
            self.refresh_udp()
            self.tune_dialog.refresh(None, (), False)
            return
        state, messages, points = worker.snapshot()
        self.refresh_udp(state)
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
        ui.openButton.setEnabled(not alive or (state.opened and worker.can_close()))
        ui.listenOnly.setEnabled(not alive)
        ui.readAllButton.setEnabled(idle)
        ui.enableButton.setEnabled(idle and state.coherent and state.verified == set(ALL))
        ui.clearButton.setEnabled(idle)
        ui.disableButton.setEnabled(control)
        ui.sendButton.setText("停止发送" if state.streaming else "开始发送")
        ui.sendButton.setEnabled((control and state.streaming) or (ready and state.owned == set(ALL)))
        powered = ready and state.coherent and state.owned == set(ALL)
        free = state.hold_state == "空闲" and not state.tune_node
        for btn in (ui.mitButton, ui.pvButton):
            btn.setEnabled(free and ((idle and set(JOINTS) <= state.verified) or powered))
        can_zero = ready and state.coherent and state.first_read and not (state.streaming or state.tracking)
        ui.zeroButton.setEnabled(can_zero and not state.tune_node)
        if self.zero_dialog.isVisible() and not can_zero:
            self.zero_dialog.reject()
        active = ready and state.streaming and state.coherent
        six_active = active and set(JOINTS) <= state.owned
        modes = [mode for mode, _ in state.motors[:6]]
        hold_ready = (six_active and not state.remote_action and all(m == "MIT" for m in modes)
                      and state.joint_sample_at > 0 and perf_counter() - state.joint_sample_at <= HOLD_MAX_DT_S)
        tune_ready = (six_active and free and worker.queued_cycles and not state.remote_action
                      and all(m in ("MIT", "PV") for m in modes)
                      and perf_counter() - state.joint_sample_at <= HOLD_MAX_DT_S)
        self.tune_dialog.refresh(state, points, tune_ready)
        ui.pidHoldButton.setEnabled(not state.tune_node and ((free and hold_ready) or (ready and state.hold_state == "持位")))
        ui.pidHoldButton.setText({"空闲": "开启双环持位", "持位": "退出持位", "退出中": "退出中"}[state.hold_state])
        ui.gravityButton.setEnabled(free and six_active)
        ui.gravityButton.setText("关闭重力补偿" if state.gravity else "开启重力补偿")
        ui.homeButton.setEnabled(free and six_active)
        for button in (ui.toolMitButton, ui.toolPvtButton):
            button.setEnabled(idle and 7 in state.verified)
        for node, button, mode, value in zip(JOINTS, self.lock_buttons, modes, state.commands):
            button.setEnabled(free and powered and mode in ("MIT", "PV"))
            button.setText(f"J{node} 释放" if mode == "PV" and node in state.owned else f"J{node} 锁定")
            unit = "rad" if mode == "PV" else "N·m"
            item = ui.anglesTable.item(node - 1, 5)
            item.setText("—" if value is None else f"{value:.3f} {unit}")
            axis = state.hold_details[node - 1] if state.hold_details else None
            self.show_hold_axis(item, axis)
        if alive and state.uncertain:
            ui.deviceStatus.setText("失能未确认")
        elif state.faulted:
            ui.deviceStatus.setText("设备通信异常")
        elif state.opened:
            ui.deviceStatus.setText("已连接 · 只听" if worker.listen_only else "已连接")
        else:
            ui.deviceStatus.setText("正在打开" if alive else "未连接")
        if not alive:
            worker.join()
            self.worker = None
            if self.closing_window:
                QtCore.QTimer.singleShot(0, self.close)

    @staticmethod
    def show_hold_axis(item, axis):
        if axis is None:
            if item.toolTip():
                item.setToolTip("")
                item.setBackground(QtGui.QBrush())
            return
        flags = [name for name, reached in (("PID 限幅", axis.pid_limited), ("电机上限", axis.output_limited),
                                             ("变化率受限", axis.rate_limited)) if reached]
        lines = []
        if axis.target is not None:
            lines.extend((f"锁存角 {axis.target:.4f} rad　位置误差 {axis.error:.4f} rad",
                          f"目标速度 {axis.velocity_ref:.4f} rad/s"))
        lines.extend((f"重力前馈 {axis.gravity:.4f} N·m", f"PID 原始 / 限幅后 {axis.pid_raw:.4f} / {axis.pid:.4f} N·m",
                      f"电机有效上限 ±{axis.output_limit:.4f} N·m", "、".join(flags) if flags else "未受限"))
        item.setToolTip("\n".join(lines))
        item.setBackground(QtGui.QBrush(QtGui.QColor("#ffe3a6")) if flags else QtGui.QBrush())

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
        return self.worker.submit("close")

    def closeEvent(self, event):
        if self.worker is not None:
            event.ignore()
            if self.close_device():
                self.closing_window = True
        else:
            if self.udp:
                self.close_udp()
            super().closeEvent(event)


def main():
    sys.setswitchinterval(PYTHON_SWITCH_INTERVAL_S)
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
