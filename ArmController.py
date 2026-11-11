"""UR 本地 MIT 重力补偿、双环持位和 PV 预设位目标换算。"""

from dataclasses import dataclass
from math import isfinite
from typing import NamedTuple

import numpy as np

from Robot_UR import Robot_UR


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
        for name in ("pos_kp", "pos_ki", "pos_kd", "vel_kp", "vel_ki", "vel_kd", "pid_limit",
                     "speed_limit", "torque_rate", "pos_i_limit", "vel_i_limit"):
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
    pid_raw: float
    pid: float
    output_limit: float
    pid_limited: bool
    output_limited: bool
    rate_limited: bool


class ArmController:
    def __init__(self, home, gravity_gain, gravity_axes, hold, motors):
        self.home = six_values("预设角", home)
        self.gravity_gain = six_values("重力补偿倍率", gravity_gain)
        self.gravity_axes = six_values("重力补偿轴", gravity_axes)
        self.gravity_on = False
        self.hold = hold
        self.output_limit = np.array([m.output_limit for m in motors])
        self.speed_limit = np.minimum(hold.speed_limit, [m.max_velocity for m in motors])
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

    def begin_hold(self, q_hold, feedback, last_torque):
        self.q_hold = np.array(q_hold)
        self.last_velocity = np.array([f.velocity for f in feedback])
        self.acceleration = np.zeros(6)
        self.pos_i = np.zeros(6)
        self.vel_i = np.zeros(6)
        self.previous = np.clip(last_torque, -self.output_limit, self.output_limit)
        self.hold_state = "持位"
        self.gravity_on = True

    def begin_release(self):
        self.hold_state = "退出中"
        self.q_hold = self.last_velocity = self.acceleration = self.pos_i = self.vel_i = None
        self.details = ()

    def end_hold(self):
        self.hold_state = "空闲"
        self.q_hold = self.last_velocity = self.acceleration = self.pos_i = self.vel_i = None
        self.previous = None
        self.details = ()

    def step_hold(self, feedback, dt):
        """每个完整样本计算一次，首帧 dt 为零，保持已确认的上一输出。"""
        p = self.hold
        gravity = self.gravity_commands(feedback)
        if self.hold_state == "持位":
            velocity = np.array([f.velocity for f in feedback])
            if dt > 0:
                acceleration = (velocity - self.last_velocity) / dt
                self.acceleration += dt / (p.filter_s + dt) * (acceleration - self.acceleration)
            self.last_velocity = velocity
            error = self.q_hold - [f.position for f in feedback]
            v_raw = p.pos_kp * error + self.pos_i - p.pos_kd * velocity
            v_ref = np.clip(v_raw, -self.speed_limit, self.speed_limit)
            v_error = v_ref - velocity
            pid_raw = p.vel_kp * v_error + self.vel_i - p.vel_kd * self.acceleration
            pid = np.clip(pid_raw, -p.pid_limit, p.pid_limit)
        else:
            pid_raw = pid = np.zeros(6)

        request = gravity + pid
        goal = np.clip(request, -self.output_limit, self.output_limit)
        step = p.torque_rate * dt
        command = np.clip(goal, self.previous - step, self.previous + step)
        self.previous = command

        if self.hold_state == "持位":
            # 先发本拍输出，再决定下一拍积分。任一内环限幅都可冻结同向的两级积分。
            inner_gaps = np.array([pid_raw - pid, request - goal, goal - command])
            dq = p.pos_ki * error * dt
            dv = p.vel_ki * v_error * dt
            pos_ok = ((dq * (v_raw - v_ref) <= 0) & (dq * inner_gaps <= 0).all(axis=0))
            vel_ok = (dv * inner_gaps <= 0).all(axis=0)
            enabled = p.pid_limit > 0
            self.pos_i = np.clip(self.pos_i + dq * (pos_ok & enabled), -p.pos_i_limit, p.pos_i_limit)
            self.vel_i = np.clip(self.vel_i + dv * (vel_ok & enabled), -p.vel_i_limit, p.vel_i_limit)
            target = self.q_hold
        else:
            target = error = v_ref = (None,) * 6
        self.details = tuple(map(HoldAxis, target, error, v_ref, gravity, pid_raw, pid, self.output_limit,
                                 pid_raw != pid, abs(request) >= self.output_limit, goal != command))
        return command, goal
