"""UR 同构主端窗口。选择 dmarm 解释器后直接运行本文件。"""

from math import degrees, isnan, pi
from datetime import datetime
from pathlib import Path
import sys
from time import perf_counter

from PyQt5 import QtCore, QtGui, QtWidgets

from ArmController import ArmController, HoldParameters
from FreeSpacePredictor import FreeSpaceParameters
from Robot_UR import ArmModel, Robot_UR
from ui_Form1 import Ui_Form1
from ui_ZeroDialog import Ui_ZeroDialog
from CanWorker import ALL, JOINTS, CanWorker, TuneLimits
from CanProcess import CanProcessBus
from PidTuneDialog import PidTuneDialog
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
FORCE_FEEDBACK_VELOCITY_FILTER_S = 0.010  # 日志中速度差的一阶低通时间常数，单位 s；本地阻尼直接使用主端实测速度
# J1～J6 关节侧反馈分量的对称上限，单位 N·m；0 关闭该轴的弹簧和阻尼，不影响重力补偿或持位。
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
        ready = bool(state and link and link.fresh() and not state.tune_node and state.streaming
                     and state.hold_state == "空闲" and state.coherent and not state.busy
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
            device = USBCANFD(ZLG_SDK_DIR_WIN, ZLG_SDK_DIR_LINUX, CANFD_DEVICE_INDEX, CANFD_CHANNEL_INDEX,
                              QUERY_TIMEOUT_S, joint_torque_limits=MOTOR_OUTPUT_TORQUE_LIMIT_NM)
            bus = CanProcessBus(device, CAN_CYCLE_DELAY_S, TOOL_UPDATE_CYCLES, CAN_COMMAND_MAX_AGE_S)
            hold = HoldParameters(**hold_settings())
            prediction = FreeSpaceParameters(FREE_SPACE_SPEED_LIMIT_RAD_S, FREE_SPACE_ACCEL_POSITIVE_RAD_S2,
                                             FREE_SPACE_ACCEL_NEGATIVE_RAD_S2, FREE_SPACE_BRAKING_RAD_S2,
                                             FREE_SPACE_DELAY_S, FREE_SPACE_STEP_S)
            controller = ArmController(UR_HOME_RAD, GRAVITY_GAIN, GRAVITY_AXES, hold,
                                       [bus.motors[n] for n in JOINTS],
                                       arm_model=self.arm_model, gravity_data=self.arm_model.model.createData(),
                                       force_stiffness=FORCE_FEEDBACK_STIFFNESS_NM_PER_RAD,
                                       force_damping=FORCE_FEEDBACK_DAMPING_NM_S_PER_RAD,
                                       force_velocity_filter_s=FORCE_FEEDBACK_VELOCITY_FILTER_S,
                                       force_limit=FORCE_FEEDBACK_LIMIT_NM,
                                       force_prediction_enabled=FORCE_FEEDBACK_PREDICTION_ENABLED,
                                       force_prediction_weight=FORCE_FEEDBACK_PREDICTION_WEIGHT,
                                       free_space_parameters=prediction)
        except ValueError as error:
            self.ui.controlStatus.setText(f"参数错误：{error}")
            return
        self.worker = CanWorker(
            bus, self.ui.listenOnly.isChecked(), speed_limit=PV_SPEED_LIMIT_RAD_S,
            pose_period=POSE_PERIOD_S, queued_cycles=CANFD_USE_TX_QUEUE,
            force_requires_slave_follow=FORCE_FEEDBACK_REQUIRES_SLAVE_FOLLOW,
            controller=controller,
            tune_limits=self.tune_limits,
            slave_link=self.udp,
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
        force_frame = self.udp.fresh() if self.udp else None
        ui.forceFeedbackButton.setEnabled(
            ready and worker.slave_allows_force_feedback(force_frame)
            and (state.force_feedback or (force_frame is not None and worker.can_reflect_force(state)))
        )
        ui.forceFeedbackButton.setText("关闭阻抗力反馈" if state.force_feedback else "开启阻抗力反馈")
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
        if axis.reflection:
            lines.append(f"电机侧力反馈 {axis.reflection:+.4f} N·m")
        lines.extend((f"重力前馈 {axis.gravity:.4f} N·m", f"PID 原始 / 限幅后 {axis.pid_raw:.4f} / {axis.pid:.4f} N·m",
                      f"电机有效上限 ±{axis.output_limit:.4f} N·m", "、".join(flags) if flags else "未受限"))
        item.setToolTip("\n".join(lines))
        item.setBackground(QtGui.QBrush(QtGui.QColor("#ffe3a6")) if flags else QtGui.QBrush())

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
