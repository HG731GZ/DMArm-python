"""UR 本地 MIT 重力补偿和 PV 预设位目标换算。"""

from Robot_UR import Robot_UR


class ArmController:
    def __init__(self, home, gravity_gain, gravity_axes):
        self.home = home
        self.gravity_gain = gravity_gain
        self.gravity_axes = gravity_axes
        self.gravity_on = False

    def motor_home(self, motors):
        """按原 UR 主端的传动比把预设 DH 角换成电机目标角。"""
        targets = {}
        for node, (angle, ratio) in enumerate(zip(self.home, Robot_UR.ratio), 1):
            target = angle / ratio
            if abs(target) > motors[node].max_position:
                raise ValueError(f"J{node} 预设目标超出电机位置量程")
            targets[node] = target
        return targets

    def gravity_commands(self, motors):
        model = Robot_UR.from_feedback([motors[n].feedback for n in range(1, 7)])
        return model.Tau_G_Motor * self.gravity_gain * self.gravity_axes
