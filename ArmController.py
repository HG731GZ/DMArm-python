"""UR 本地 MIT 重力补偿、双环持位和 PV 预设位目标换算。"""

from dataclasses import dataclass, replace
from math import isfinite
from typing import NamedTuple

import numpy as np

from Robot_UR import Robot_UR


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


class ArmController:
    def __init__(self, home, gravity_gain, gravity_axes, hold, motors, *, force_gain, force_limit):
        self.home = six_values("预设角", home)
        self.gravity_gain = six_values("重力补偿倍率", gravity_gain)
        self.gravity_axes = six_values("重力补偿轴", gravity_axes)
        self.gravity_on = False
        self.force_gain = six_values("力反馈缩放系数", force_gain)
        self.force_limit = six_values("关节反馈力矩上限", force_limit)
        if (self.force_gain < 0).any() or (self.force_gain > 1).any() or (self.force_limit <= 0).any():
            raise ValueError("力反馈系数须在 0～1 之间，关节反馈力矩上限须为正数")
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
        return Robot_UR.gravity_at(angles) * self.gravity_gain * self.gravity_axes

    def reflected_torque(self, net_torque):
        """在关节侧反向、缩放及对称限幅，再按虚功关系换算为电机力矩。"""
        source = np.array([v if isfinite(v) else 0.0 for v in net_torque])
        joint = np.clip(-self.force_gain * source, -self.force_limit, self.force_limit)
        return joint / Robot_UR.ratio

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
