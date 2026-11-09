"""UR 本地 MIT 重力补偿和 PV 预设位目标换算。"""

from math import remainder, tau

from Robot_UR import Robot_UR


class ArmController:
    def __init__(self, home, home_speed, gravity_gain, gravity_axes, ramp_seconds, arrival_rad):
        self.home = home
        self.home_speed = home_speed
        self.gravity_gain = gravity_gain
        self.gravity_axes = gravity_axes
        self.ramp_seconds = ramp_seconds
        self.arrival_rad = arrival_rad
        self.gravity_on = False
        self.gravity_start = 0.0
        self.home_targets = {}
        self.home_deadline = 0.0

    def reset(self):
        self.gravity_on = False
        self.home_targets.clear()

    def motor_home(self, motors):
        """把预设 DH 角映射到最接近当前原始反馈的电机角，跨圈时保持连续。"""
        targets = {}
        for node, (angle, ratio) in enumerate(zip(self.home, Robot_UR.ratio), 1):
            motor = motors[node]
            current = motor.feedback.position
            target = current + remainder(angle * ratio - current, tau)
            if abs(target) > motor.max_position:
                raise ValueError(f"J{node} 连续预设目标超出电机位置反馈量程")
            targets[node] = target
        return targets

    def gravity_commands(self, motors, now):
        model = Robot_UR.from_feedback([motors[n].feedback for n in range(1, 7)])
        elapsed = now - self.gravity_start
        scale = 1.0 if elapsed >= self.ramp_seconds else elapsed / self.ramp_seconds
        return scale * model.Tau_G_Motor * self.gravity_gain * self.gravity_axes
