"""UR 主端 MIT 重力补偿、关节阻抗力反馈、双环持位和 PV 预设位目标换算。"""

from dataclasses import dataclass, replace
from math import expm1, isfinite
from typing import NamedTuple

import numpy as np

from Robot_UR import Robot_UR
from FreeSpacePredictor import FreeSpacePredictor


HOLD_FIELDS = ("pos_kp", "pos_ki", "pos_kd", "vel_kp", "vel_ki", "vel_kd", "pid_limit",
               "speed_limit", "torque_rate", "pos_i_limit", "vel_i_limit")


def six_values(name, values):
    result = np.asarray(values, dtype=float)
    if result.shape != (6,) or not np.isfinite(result).all():
        raise ValueError(f"{name} 须按 J1～J6 填写六个有限值")
    return result


@dataclass
class HoldParameters:
    pos_kp: tuple
    pos_ki: tuple
    pos_kd: tuple
    vel_kp: tuple
    vel_ki: tuple
    vel_kd: tuple
    pid_limit: tuple
    speed_limit: tuple
    torque_rate: tuple
    pos_i_limit: tuple
    vel_i_limit: tuple
    filter_s: float
    max_dt: float

    def __post_init__(self):
        for name in HOLD_FIELDS:
            values = six_values(name, getattr(self, name))
            if (values < 0).any():
                raise ValueError(f"{name} 各项须为非负数")
            setattr(self, name, values)
        if (self.speed_limit == 0).any() or (self.torque_rate == 0).any():
            raise ValueError("持位目标速度和力矩变化率上限须为正数")
        if not isfinite(self.filter_s) or self.filter_s <= 0 or not isfinite(self.max_dt) or self.max_dt <= 0:
            raise ValueError("微分滤波时间和最大采样间隔须为正的有限秒数")


class HoldAxis(NamedTuple):
    target: float | None
    error: float | None
    velocity_ref: float | None
    gravity: float
    reflection: float
    pid_raw: float
    pid: float
    output_limit: float
    pid_limited: bool
    output_limited: bool
    rate_limited: bool


class PredictionObservation(NamedTuple):
    sampled_at: float
    slave_received_unix_s: float
    q: tuple
    dq: tuple
    position_error: tuple
    velocity_error: tuple
    raw_position_error: tuple
    weighted_position_error: tuple
    stiffness_limits: tuple
    weighted_torque: tuple

    def candidate_torque(self, errors, damping):
        """日志线程用同一拍的位置误差和本地阻尼还原候选力矩。"""
        return tuple(max(-limit, min(limit, k * error + local))
                     for (k, limit), error, local in zip(self.stiffness_limits, errors, damping))


class ArmController:
    def __init__(self, home, gravity_gain, gravity_axes, hold, motors, *, arm_model, gravity_data,
                 force_stiffness, force_damping, force_velocity_filter_s, force_limit,
                 force_prediction_enabled, force_prediction_weight, free_space_parameters):
        self.home = six_values("预设角", home)
        self.arm_model = arm_model
        self.gravity_data = gravity_data
        self.gravity_gain = six_values("重力补偿倍率", gravity_gain)
        self.gravity_axes = six_values("重力补偿轴", gravity_axes)
        self.gravity_on = False
        self.force_stiffness = six_values("关节反馈刚度", force_stiffness)
        self.force_damping = six_values("关节反馈阻尼", force_damping)
        self.force_limit = six_values("关节反馈力矩上限", force_limit)
        self.stiffness_limits = tuple(zip(self.force_stiffness.tolist(), self.force_limit.tolist()))
        self.force_prediction_weight = six_values("预测位置补偿权重", force_prediction_weight)
        if (self.force_stiffness < 0).any() or (self.force_damping < 0).any() or (self.force_limit < 0).any():
            raise ValueError("关节反馈刚度、阻尼和力矩上限均须为非负数")
        if ((self.force_prediction_weight < 0) | (self.force_prediction_weight > 1)).any():
            raise ValueError("预测位置补偿权重各项须位于 0～1")
        if not isfinite(force_velocity_filter_s) or force_velocity_filter_s < 0:
            raise ValueError("力反馈速度差滤波时间常数须为非负有限秒数")
        self.force_velocity_filter_s = force_velocity_filter_s
        self.force_prediction_enabled = force_prediction_enabled
        self.predictor = FreeSpacePredictor(free_space_parameters)
        self.reset_velocity_filter()
        self.hold = hold
        self.output_limit = np.array([m.output_limit for m in motors])
        self.motor_speeds = np.array([m.max_velocity for m in motors])
        self.speed_limit = np.minimum(hold.speed_limit, self.motor_speeds)
        self.end_hold()

    def motor_home(self, motors):
        """按原 UR 主端的传动比把预设 DH 角换成电机目标角。"""
        targets = self.motor_targets(self.home)
        for node, target in targets.items():
            if abs(target) > motors[node].max_position:
                raise ValueError(f"J{node} 目标超出电机位置量程")
        return targets

    @staticmethod
    def motor_targets(angles):
        return dict(enumerate(angles / Robot_UR.ratio, 1))

    def gravity_commands(self, feedback):
        angles = Robot_UR.motor2dh([f.position for f in feedback])
        return self.arm_model.motor_gravity(angles, self.gravity_data) * self.gravity_gain * self.gravity_axes

    def reset_velocity_filter(self):
        self.filtered_velocity_error = None
        self.local_damping_torque = None
        self.last_velocity_sample_at = None
        self.reset_prediction()

    def reset_prediction(self):
        self.predictor.reset()
        self.prediction_frame = None
        self.predicted_q = self.predicted_dq = None
        self.prediction_velocity_error = None
        self.last_prediction_sample_at = None
        self.prediction_observation = None

    def update_prediction(self, feedback, frame, commands, sampled_at, max_dt):
        """新从端帧按实际发布历史推进预测，残差滤波使用控制采用的 CAN 样本间隔。"""
        if frame is not self.prediction_frame:
            if self.predictor.origin is None:
                self.predictor.initialize(frame.received, frame.q, frame.dq)
            self.predicted_q, self.predicted_dq = self.predictor.advance(frame.received, commands)
            self.prediction_frame = frame
        dt = sampled_at - self.last_prediction_sample_at if self.last_prediction_sample_at is not None else None
        residual = np.asarray(frame.dq) - self.predicted_dq
        if dt is None or dt > max_dt or self.force_velocity_filter_s == 0:
            self.prediction_velocity_error = residual
        elif dt > 0:
            alpha = -expm1(-dt / self.force_velocity_filter_s)
            self.prediction_velocity_error += alpha * (residual - self.prediction_velocity_error)
        self.last_prediction_sample_at = sampled_at
        error = Robot_UR.wrap(np.asarray(frame.q) - self.predicted_q)
        raw_error = Robot_UR.wrap(np.asarray(frame.q) - Robot_UR.motor2dh([f.position for f in feedback]))
        weight = self.force_prediction_weight
        weighted_error = (1 - weight) * raw_error + weight * error
        raw_spring = self.force_stiffness * raw_error
        # 预测最多抵消原始主从弹簧，按轴约束后再叠加本地阻尼。
        spring = np.clip(self.force_stiffness * weighted_error,
                         np.minimum(0.0, raw_spring), np.maximum(0.0, raw_spring))
        joint_torque = np.clip(spring + self.local_damping_torque, -self.force_limit, self.force_limit)
        self.prediction_observation = PredictionObservation(
            sampled_at, frame.wall_time, tuple(Robot_UR.wrap(self.predicted_q)), tuple(self.predicted_dq),
            tuple(error), tuple(self.prediction_velocity_error),
            tuple(raw_error), tuple(weighted_error), self.stiffness_limits, tuple(joint_torque),
        )

    def update_velocity_error(self, feedback, slave_dq, sampled_at, max_dt):
        """本地阻尼直接取本拍主端速度；主从滤波速度差仅用于日志观测。"""
        master_velocity = Robot_UR.joint_velocity(feedback)
        self.local_damping_torque = -self.force_damping * master_velocity
        dt = sampled_at - self.last_velocity_sample_at if self.last_velocity_sample_at is not None else None
        if dt == 0 and self.force_velocity_filter_s > 0:
            return
        velocity_error = np.asarray(slave_dq) - master_velocity
        if dt is None or dt > max_dt or self.force_velocity_filter_s == 0:
            self.filtered_velocity_error = velocity_error
        else:
            alpha = -expm1(-dt / self.force_velocity_filter_s)
            self.filtered_velocity_error += alpha * (velocity_error - self.filtered_velocity_error)
        self.last_velocity_sample_at = sampled_at

    def impedance_torque(self, feedback, slave_q):
        """位置弹簧叠加主端本地阻尼，关节侧合成限幅后换算为电机力矩。"""
        if self.force_prediction_enabled:
            return np.asarray(self.prediction_observation.weighted_torque) / Robot_UR.ratio
        master_q = Robot_UR.motor2dh([f.position for f in feedback])
        error = Robot_UR.wrap(np.asarray(slave_q) - master_q)
        joint_torque = self.force_stiffness * error + self.local_damping_torque
        return np.clip(joint_torque, -self.force_limit, self.force_limit) / Robot_UR.ratio

    def begin_hold(self, q_hold, feedback, last_torque, axes=slice(0, 6)):
        self.axes = axes
        self.q_hold = np.array(q_hold)
        self.last_velocity = np.array([f.velocity for f in feedback[axes]])
        self.acceleration = np.zeros_like(self.last_velocity)
        self.pos_i = np.zeros_like(self.last_velocity)
        self.vel_i = np.zeros_like(self.last_velocity)
        self.previous = np.clip(last_torque, -self.output_limit[axes], self.output_limit[axes])
        self.hold_state = "持位"
        self.gravity_on = True

    def set_axis_reference(self, target):
        self.q_hold[0] = target

    def apply_axis_parameters(self, index, values, feedback, last_torque):
        """整体校验成功后替换参数，保留当前原始电机角目标。"""
        updated = {}
        for name in HOLD_FIELDS:
            column = getattr(self.hold, name).copy()
            column[index] = values[name]
            updated[name] = column
        self.hold = replace(self.hold, **updated)
        self.speed_limit[index] = min(self.hold.speed_limit[index], self.motor_speeds[index])
        self.begin_hold(self.q_hold, feedback, [last_torque], self.axes)

    def begin_release(self):
        self.hold_state = "退出中"
        self.q_hold = self.last_velocity = self.acceleration = self.pos_i = self.vel_i = None
        self.details = ()

    def end_hold(self):
        self.hold_state = "空闲"
        self.q_hold = self.last_velocity = self.acceleration = self.pos_i = self.vel_i = None
        self.previous = None
        self.details = ()

    def step_hold(self, feedback, dt, reflected=None):
        """每个完整样本计算一次，首帧 dt 为零，保持已确认的上一输出。"""
        p = self.hold
        axes = range(6)[self.axes]
        gravity = self.gravity_commands(feedback)
        holding = self.hold_state == "持位"
        alpha = dt / (p.filter_s + dt)
        commands = np.empty(len(axes))
        goals = np.empty(len(axes))
        details = [None] * 6
        # 状态按活动轴排列，参数、实测值及诊断按整臂轴号排列。
        for j, i in enumerate(axes):
            target = error = v_ref = None
            pid_raw = pid = 0.0
            if holding:
                velocity = feedback[i].velocity
                if dt > 0:
                    acceleration = (velocity - self.last_velocity[j]) / dt
                    self.acceleration[j] += alpha * (acceleration - self.acceleration[j])
                self.last_velocity[j] = velocity
                target = self.q_hold[j]
                error = target - feedback[i].position
                v_raw = p.pos_kp[i] * error + self.pos_i[j] - p.pos_kd[i] * velocity
                v_ref = max(-self.speed_limit[i], min(self.speed_limit[i], v_raw))
                v_error = v_ref - velocity
                pid_raw = p.vel_kp[i] * v_error + self.vel_i[j] - p.vel_kd[i] * self.acceleration[j]
                pid = max(-p.pid_limit[i], min(p.pid_limit[i], pid_raw))

            limit = self.output_limit[i]
            reflection = reflected[i] if reflected is not None else 0.0
            request = gravity[i] + pid + reflection
            goal = max(-limit, min(limit, request))
            step = p.torque_rate[i] * dt
            command = max(self.previous[j] - step, min(self.previous[j] + step, goal))
            commands[j], goals[j] = command, goal

            if holding and p.pid_limit[i] > 0:
                # 本拍输出已确定，积分只在不会加剧现有限幅时累加到下一拍。
                gaps = (pid_raw - pid, request - goal, goal - command)
                dq = p.pos_ki[i] * error * dt
                dv = p.vel_ki[i] * v_error * dt
                if dq * (v_raw - v_ref) <= 0 and all(dq * gap <= 0 for gap in gaps):
                    self.pos_i[j] = max(-p.pos_i_limit[i], min(p.pos_i_limit[i], self.pos_i[j] + dq))
                if all(dv * gap <= 0 for gap in gaps):
                    self.vel_i[j] = max(-p.vel_i_limit[i], min(p.vel_i_limit[i], self.vel_i[j] + dv))
            details[i] = HoldAxis(target, error, v_ref, gravity[i], reflection, pid_raw, pid, limit,
                                  pid_raw != pid, abs(request) >= limit, goal != command)
        self.previous = commands
        self.details = tuple(details)
        return commands, goals
