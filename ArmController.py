"""UR 主端 MIT 重力补偿、关节阻抗力反馈和 PV 预设位目标换算。"""

from math import expm1, isfinite
from typing import NamedTuple

import numpy as np

from Robot_UR import Robot_UR
from FreeSpacePredictor import FreeSpacePredictor


def six_values(name, values):
    result = np.asarray(values, dtype=float)
    if result.shape != (6,) or not np.isfinite(result).all():
        raise ValueError(f"{name} 须按 J1～J6 填写六个有限值")
    return result


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
    def __init__(self, home, gravity_gain, gravity_axes, *, arm_model, gravity_data,
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
