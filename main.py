"""UR 同构主端窗口。选择 dmarm 解释器后直接运行本文件。"""

from math import degrees, isnan, pi
from datetime import datetime
from pathlib import Path
import sys

from PyQt5 import QtCore, QtGui, QtWidgets

from ArmController import ArmController
from FreeSpacePredictor import FreeSpaceParameters
from Robot_UR import ArmModel, Robot_UR
from ui_Form1 import Ui_Form1
from ui_ZeroDialog import Ui_ZeroDialog
from CanWorker import ALL, JOINTS, CanWorker
from CanProcess import CanProcessBus
from DMMotor import LIMITS, MOTOR_MODELS
from TeleopLog import TeleopLog
from TeleopPublisher import TeleopPublisher
from USBCANFD import USBCANFD
from UdpClass import UdpClass

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
UDP_TIMEOUT_S = 1.0  # 从端状态新鲜时限，单位 s；运动联调前按现场确认
TELEOP_LOG_ENABLED = True  # 遥操作时等从端开启跟随后自动记录 CSV，设为 False 时可用按钮手动记录
QUERY_TIMEOUT_S = 0.5  # 单条只读查询等待应答的时限，单位 s
POSE_PERIOD_S = 0.10  # 连续读取失能姿态的轮间隔，单位 s
UI_REFRESH_MS = 10  # 界面快照刷新间隔，单位 ms
POSE_DISPLAY_MS = 50  # 主从姿态图、从端关节表及 UDP 统计的显示间隔，单位 ms
ARM_MODEL_URDF = PROJECT_DIR / "ur_master_arm" / "ur_master_arm.urdf"  # 主从共用的实体模型入口
SLAVE_MODEL_OPACITY = 0.40  # 从端模型不透明度，无量纲，0 为全透明，1 为实心
CAN_CYCLE_DELAY_S = 0  # 本轮 CAN FD 反馈收齐后到下一轮的等待时间，单位 s；0 表示立即继续
CAN_COMMAND_MAX_AGE_S = 0.50  # 从控制所用六轴反馈起算的指令有效期，单位 s；超龄由 CAN 进程逐轴失能并回读
TOOL_UPDATE_CYCLES = 50  # 每隔多少轮 CAN FD 周期发送一次夹钳指令，取正整数；1 表示每轮都发
PYTHON_SWITCH_INTERVAL_S = 0.00025  # 主进程中控制、UDP、日志和界面线程的切换间隔，单位 s

# 按 J1～J6 排列。把某一项改为 False，该轴 MIT 前馈力矩即为零，重启主窗生效。
GRAVITY_AXES = (True, True, True, True, True, True)
GRAVITY_GAIN = (1.1, 1.1, 1.1, 1.2, 1.1, 1.1)  # 原 C# 电机侧重力矩倍率，无单位

FORCE_FEEDBACK_REQUIRES_SLAVE_FOLLOW = True  # True 时从端 teleop_state 为 0 会关闭力反馈并禁用按钮
# J1～J6 反馈先合成弹簧和阻尼，再按关节侧上限限幅。
FORCE_FEEDBACK_STIFFNESS_NM_PER_RAD = (40.0, 40.0, 40.0, 15.0, 15.0, 15.0)  # 关节刚度，单位 N·m/rad
FORCE_FEEDBACK_DAMPING_NM_S_PER_RAD = (0.10, 0.10, 0.10, 0.05, 0.05, 0.05)  # 主端本地阻尼，力矩为负系数乘实测关节速度，单位 N·m·s/rad
FORCE_FEEDBACK_MAX_AGE_S = 0.05  # 力反馈使用的完整六轴采样时限，单位 s
FORCE_FEEDBACK_VELOCITY_FILTER_S = 0.010  # 日志中速度差的一阶低通时间常数，单位 s；本地阻尼直接使用主端实测速度
# J1～J6 关节侧反馈分量的对称上限，单位 N·m；0 关闭该轴的弹簧和阻尼。
FORCE_FEEDBACK_LIMIT_NM = (6, 6, 6, 2, 2, 2)

# True 时预测按权重抵消原始主从弹簧，最多抵消到零。False 时使用原始弹簧。两种模式均另加主端本地阻尼。
FORCE_FEEDBACK_PREDICTION_ENABLED = True
FORCE_FEEDBACK_PREDICTION_WEIGHT = (0.95, 0.95, 0.95, 0.95, 0.95, 0.95)  # J1～J6 位置补偿权重，无量纲，范围 0～1；0 不补偿，1 全补偿
# 以下采用 2026-11-24 无接触日志的速度差分辨识结果，依次对应 J1～J6。
FREE_SPACE_SPEED_LIMIT_RAD_S = (0.734, 0.773, 0.755, 0.925, 0.889, 0.885)  # 预测速度上限，rad/s
FREE_SPACE_ACCEL_POSITIVE_RAD_S2 = (6.134876, 3.580531, 3.924482, 10.721505, 10.734643, 9.488518)  # 正向起步幅值，rad/s²
FREE_SPACE_ACCEL_NEGATIVE_RAD_S2 = (6.113493, 2.549822, 2.591227, 9.804715, 9.251685, 12.863736)  # 负向起步幅值，rad/s²
FREE_SPACE_BRAKING_RAD_S2 = (8.559878, 6.437806, 6.325182, 11.855675, 11.749363, 22.444958)  # 双向制动幅值，rad/s²
FREE_SPACE_DELAY_S = (0.057, 0.072, 0.074, 0.034, 0.034, 0.034)  # 含显式加减速模型的等效滞后，s
FREE_SPACE_STEP_S = 0.0025  # 预测器固定积分步长，s；与离线回放保持一致

# 六关节 MIT 指令的软件力矩上限取各自协议 T_MAX 的 95%，单位 N·m。
MOTOR_OUTPUT_TORQUE_LIMIT_NM = tuple(0.95 * LIMITS[model][2] for model in MOTOR_MODELS[:6])
UR_HOME_RAD = (-pi/2, -pi/2, -pi/2, -pi/2, pi/2, 0)  # DH 预设关节角，单位 rad
PV_SPEED_LIMIT_RAD_S = 0.5  # PV 位置指令的速度上限，单位 rad/s


class MainWindow(QtWidgets.QMainWindow):
    teleop_log_message = QtCore.pyqtSignal(str)
    teleop_log_error = QtCore.pyqtSignal(object, str)

    def __init__(self):
        super().__init__()
        self.ui = ui = Ui_Form1()
        ui.setupUi(self)
        self.arm_model = ArmModel.from_urdf(ARM_MODEL_URDF)
        self.display_data = self.arm_model.model.createData()
        ui.armView.set_model(ARM_MODEL_URDF, self.arm_model.assembly, SLAVE_MODEL_OPACITY)
        self.last_feedback = None
        self.worker = None
        self.udp = None
        self.displayed_frame = None
        self.drawn_models = (None, None)
        self.closing_window = False
        ui.channelLabel.setText(f"设备 {CANFD_DEVICE_INDEX}  /  通道 {CANFD_CHANNEL_INDEX}  ·  1 / 5 Mbit/s")
        ui.localEndpoint.setText(f"本机  {UDP_LOCAL_IP}:{UDP_LOCAL_PORT}")
        ui.remoteAddress.setText(f"从端  {UDP_REMOTE_IP}")
        ui.remotePort.setValue(UDP_REMOTE_PORT)
        for table in (ui.feedbackTable, ui.anglesTable):
            table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        for column in (0, 7, 8):
            ui.anglesTable.horizontalHeader().setSectionResizeMode(column, QtWidgets.QHeaderView.ResizeToContents)
        self.timer = QtCore.QTimer(self, interval=UI_REFRESH_MS)
        self.timer.timeout.connect(self.refresh)
        self.timer.start()
        self.display_timer = QtCore.QTimer(self, interval=POSE_DISPLAY_MS)
        self.display_timer.timeout.connect(self.refresh_pose_display)
        self.display_timer.start()
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
        ui.forceFeedbackButton.clicked.connect(self.toggle_force_feedback)
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
        self.teleop_log_message.connect(ui.frameLog.appendPlainText)
        self.teleop_log_error.connect(self.log_write_failed)
        self.teleop_log = TeleopLog(PROJECT_DIR / "logs", self.teleop_log_message.emit, self.teleop_log_error.emit)
        self.publisher = TeleopPublisher(UDP_SEND_PERIOD_MS / 1000, QUERY_TIMEOUT_S, TELEOP_LOG_ENABLED,
                                        self.latest_state, self.teleop_log, self.teleop_log_message.emit)
        ui.publishMaster.toggled.connect(self.publisher.set_publishing)
        ui.logButton.clicked.connect(self.publisher.set_recording)
        self.teleop_log.start()
        self.publisher.start()
        QtWidgets.QApplication.instance().aboutToQuit.connect(self.publisher.close)

    def latest_state(self):
        worker = self.worker
        return worker.latest_state() if worker else None

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
                                (UDP_REMOTE_IP, self.ui.remotePort.value()), UDP_TIMEOUT_S,
                                command_history_s=UDP_TIMEOUT_S + max(FREE_SPACE_DELAY_S) + 2 * FREE_SPACE_STEP_S)
        except OSError as error:
            self.ui.udpStatus.setText(f"打开 UDP 失败：{error}")
            return
        self.udp.start()
        self.publisher.set_link(self.udp)
        if self.worker is not None:
            self.worker.slave_link = self.udp
        self.ui.remotePort.setEnabled(False)
        self.ui.udpButton.setText("关闭 UDP")
        self.refresh_pose_display()
        self.refresh()

    def close_udp(self):
        self.publisher.set_link(None)
        self.ui.publishMaster.setChecked(False)
        if self.worker is not None:
            self.worker.slave_link = None
        self.udp.close()
        self.udp = None
        self.ui.udpButton.setText("打开 UDP")
        self.ui.remotePort.setEnabled(True)
        self.ui.udpStatus.setText("UDP 未打开")
        self.refresh_pose_display()
        self.refresh_udp()

    def log_write_failed(self, first_sample, message):
        self.ui.frameLog.appendPlainText(message)
        self.publisher.log_failed(first_sample)

    def toggle_follow(self):
        if self.worker.state.remote_action:
            self.command("remote_off")
        else:
            self.command("follow", (self.udp,))

    def refresh_pose_display(self):
        link = self.udp
        frame = link.fresh() if link else None
        if frame is not self.displayed_frame:
            self.show_slave_frame(frame)
            self.displayed_frame = frame
        view = self.ui.armView
        models = (view.master, view.slave)
        if models != self.drawn_models:
            self.ui.masterLegend.setEnabled(view.master is not None)
            self.ui.slaveLegend.setEnabled(view.slave is not None)
            self.ui.assemblyPreviewLabel.setVisible(view.master is None and view.slave is None)
            view.update()
            self.drawn_models = models
        if link:
            last = link.latest
            status = "等待有效从端帧"
            if last:
                stamp = datetime.fromtimestamp(last.wall_time).strftime("%H:%M:%S.%f")[:-3]
                grip = ", ".join("无效" if isnan(v) else f"{v:.3f}" for v in last.grip)
                status = (f"{link.peer[0]}:{link.peer[1]}  末帧 {stamp}  "
                          f"{'在线' if frame else '断流'}  遥操作 {last.teleop_state}  "
                          f"夹爪[开度, 原始电流, 原始位置] [{grip}]")
            self.ui.udpStatus.setText(f"{status}  RX {link.rx}  TX {link.tx}  丢弃 {link.rejected}")

    def refresh_udp(self, state=None):
        recording = self.publisher.log_mode is not None
        self.ui.logButton.setChecked(recording)
        self.ui.logButton.setText("停止记录" if recording else "记录日志")
        publishing = self.publisher.publishing
        if self.ui.publishMaster.isChecked() != publishing:
            with QtCore.QSignalBlocker(self.ui.publishMaster):
                self.ui.publishMaster.setChecked(publishing)
        link = self.udp
        if link and link.error:
            self.close_udp()
            self.ui.udpStatus.setText(link.error)
            return
        publish_ready = bool(link and self.publisher.master_ready(state))
        self.ui.publishMaster.setEnabled(publish_ready or publishing)
        active = bool(state and state.remote_action)
        ready = bool(state and link and link.fresh() and state.streaming
                     and state.coherent and not state.busy
                     and not state.uncertain and not state.faulted and set(JOINTS) <= state.owned)
        self.ui.syncButton.setEnabled(ready and not active)
        self.ui.followButton.setEnabled(active or ready)
        self.ui.followButton.setText("停止遥操作" if active else "持续跟随从端")

    def toggle_sending(self):
        self.command("stop" if self.worker.state.streaming else "start")

    def toggle_gravity(self):
        self.command("gravity_off" if self.worker.state.gravity else "gravity_on")

    def toggle_force_feedback(self):
        if self.worker.state.force_feedback:
            self.command("force_off")
        else:
            self.command("force_on", (self.udp,))

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
            device = USBCANFD(ZLG_SDK_DIR_WIN, ZLG_SDK_DIR_LINUX, CANFD_DEVICE_INDEX, CANFD_CHANNEL_INDEX,
                              QUERY_TIMEOUT_S, joint_torque_limits=MOTOR_OUTPUT_TORQUE_LIMIT_NM)
            bus = CanProcessBus(device, CAN_CYCLE_DELAY_S, TOOL_UPDATE_CYCLES, CAN_COMMAND_MAX_AGE_S)
            prediction = FreeSpaceParameters(FREE_SPACE_SPEED_LIMIT_RAD_S, FREE_SPACE_ACCEL_POSITIVE_RAD_S2,
                                             FREE_SPACE_ACCEL_NEGATIVE_RAD_S2, FREE_SPACE_BRAKING_RAD_S2,
                                             FREE_SPACE_DELAY_S, FREE_SPACE_STEP_S)
            controller = ArmController(UR_HOME_RAD, GRAVITY_GAIN, GRAVITY_AXES,
                                       arm_model=self.arm_model, gravity_data=self.arm_model.model.createData(),
                                       force_stiffness=FORCE_FEEDBACK_STIFFNESS_NM_PER_RAD,
                                       force_damping=FORCE_FEEDBACK_DAMPING_NM_S_PER_RAD,
                                       force_velocity_filter_s=FORCE_FEEDBACK_VELOCITY_FILTER_S,
                                       force_limit=FORCE_FEEDBACK_LIMIT_NM,
                                       force_prediction_enabled=FORCE_FEEDBACK_PREDICTION_ENABLED,
                                       force_prediction_weight=FORCE_FEEDBACK_PREDICTION_WEIGHT,
                                       free_space_parameters=prediction)
            self.worker = CanWorker(
                bus, self.ui.listenOnly.isChecked(), speed_limit=PV_SPEED_LIMIT_RAD_S,
                pose_period=POSE_PERIOD_S, queued_cycles=CANFD_USE_TX_QUEUE,
                force_feedback_max_age=FORCE_FEEDBACK_MAX_AGE_S,
                force_requires_slave_follow=FORCE_FEEDBACK_REQUIRES_SLAVE_FOLLOW,
                controller=controller, slave_link=self.udp,
            )
        except ValueError as error:
            self.ui.controlStatus.setText(f"参数错误：{error}")
            return
        self.ui.frameLog.clear()
        self.worker.start()
        self.timer.start()
        self.refresh()

    def refresh(self):
        worker = self.worker
        if worker is None:
            self.refresh_udp()
            return
        state, messages = worker.snapshot()
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
        for btn in (ui.mitButton, ui.pvButton):
            btn.setEnabled((idle and set(JOINTS) <= state.verified) or powered)
        can_zero = ready and state.coherent and state.first_read and not (state.streaming or state.tracking)
        ui.zeroButton.setEnabled(can_zero)
        if self.zero_dialog.isVisible() and not can_zero:
            self.zero_dialog.reject()
        active = ready and state.streaming and state.coherent
        six_active = active and set(JOINTS) <= state.owned
        modes = [mode for mode, _ in state.motors[:6]]
        ui.gravityButton.setEnabled(six_active)
        ui.gravityButton.setText("关闭重力补偿" if state.gravity else "开启重力补偿")
        force_frame = self.udp.fresh() if self.udp else None
        ui.forceFeedbackButton.setEnabled(
            ready and worker.slave_allows_force_feedback(force_frame)
            and (state.force_feedback or (force_frame is not None and worker.can_reflect_force(state)))
        )
        ui.forceFeedbackButton.setText("关闭阻抗力反馈" if state.force_feedback else "开启阻抗力反馈")
        ui.homeButton.setEnabled(six_active)
        for button in (ui.toolMitButton, ui.toolPvtButton):
            button.setEnabled(idle and 7 in state.verified)
        for node, button, mode, value in zip(JOINTS, self.lock_buttons, modes, state.commands):
            button.setEnabled(powered and mode in ("MIT", "PV"))
            button.setText(f"J{node} 释放" if mode == "PV" and node in state.owned else f"J{node} 锁定")
            unit = "rad" if mode == "PV" else "N·m"
            item = ui.anglesTable.item(node - 1, 5)
            item.setText("—" if value is None else f"{value:.3f} {unit}")
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

    def show_feedback(self, motors):
        joints = [fb for _, fb in motors[:6]]
        model = self.arm_model.pose(*Robot_UR.joint_state(joints), self.display_data) if all(joints) else None
        view = self.ui.armView
        view.master = model
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

    def show_slave_frame(self, frame):
        """用同一帧刷新从端姿态、实际速度、关节力矩及实际加速度，断流时清除。"""
        model = self.arm_model.pose(frame.q, frame.dq, self.display_data) if frame else None
        self.ui.armView.slave = model
        table = self.ui.anglesTable
        for i in range(6):
            angle = speed = torque = acceleration = "—"
            if frame:
                angle = f"{degrees(model.Angle[i]):.2f}"
                speed = f"{frame.dq[i]:.3f}"
                torque = "无效" if isnan(frame.torque[i]) else f"{frame.torque[i]:.3f}"
                acceleration = "无效" if isnan(frame.ddq[i]) else f"{frame.ddq[i]:.3f}"
            table.item(i, 2).setText(angle)
            table.item(i, 6).setText(speed)
            table.item(i, 7).setText(torque)
            table.item(i, 8).setText(acceleration)

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
