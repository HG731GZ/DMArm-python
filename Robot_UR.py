"""UR 同构主端的关节换算、正运动学与静态重力矩。"""

from math import cos, sin

import numpy as np


class Robot_UR:
    # 电机角 / DH 关节角的带符号传动比；J3 的电机正向与关节正向相反。
    ratio = np.array([1, 1, -1, 1, 1, 1])
    # 各节相对平移，单位 m；扭转角依次为 0、90、0、0、90、−90 度。
    offsets = np.array([(0, 0, 0), (0, 0, 0), (-.210, 0, 0),
                        (-.200, 0, .062), (0, -.073, 0), (0, .036, 0)])
    twist_cos = (1, 0, 1, 1, 0, 0)
    twist_sin = (0, 1, 0, 0, 1, -1)
    # 原 UR 模型带示教器的质量 (kg) 和各节局部质心 (m)。
    mass_com = np.array([
        [.548728, 0, -.002649, -.003134],
        [.750278, -.164665, 0, .061074],
        [.653139, -.151553, 0, -.002837],
        [.431678, 0, -.004718, -.001152],
        [.431678, 0, .004718, -.001152],
        [.277642, -.000290, .009891, .037114],
    ])
    # 本节质量一阶矩，加上下游总质量乘以到下一关节的固定偏移，单位 kg·m。
    _subtree_mass = np.cumsum(mass_com[::-1, 0])[::-1]
    _link_moments = mass_com[:, :1] * mass_com[:, 1:]
    _link_moments[:-1] += offsets[1:] * _subtree_mass[1:, None]
    _link_moments = _link_moments.tolist()

    @staticmethod
    def wrap(q):
        return np.pi - np.remainder(np.pi - q, 2 * np.pi)

    @classmethod
    def _wrapped_joints(cls, q):
        q = np.asarray(q, dtype=float)
        if q.shape != (6,) or not np.isfinite(q).all():
            raise ValueError("姿态需要六个有限的关节角（rad）")
        return cls.wrap(q)

    @classmethod
    def motor2dh(cls, positions):
        return cls._wrapped_joints(np.asarray(positions) / cls.ratio)

    @classmethod
    def joint_velocity(cls, feedback):
        """按带符号传动比将六轴电机速度换算为 DH 关节角速度（rad/s）。"""
        return np.array([f.velocity for f in feedback]) / cls.ratio

    @classmethod
    def joint_state(cls, feedback):
        """将六轴电机反馈换算为 DH 关节角（rad）及同向的关节角速度（rad/s）。"""
        return cls.motor2dh([f.position for f in feedback]), cls.joint_velocity(feedback)

    @classmethod
    def from_feedback(cls, feedback):
        return cls(*cls.joint_state(feedback))

    def __init__(self, q, velocity=None):
        self.Angle = self._wrapped_joints(q)
        self.Velocity = velocity
        self.transforms = np.empty((6, 4, 4))
        parent = np.eye(4)
        for i, (c, s, ca, sa) in enumerate(zip(np.cos(self.Angle), np.sin(self.Angle),
                                               self.twist_cos, self.twist_sin)):
            local = np.eye(4)
            local[:3, :3] = (
                (c, -s, 0), (ca*s, ca*c, -sa), (sa*s, sa*c, ca))
            local[:3, 3] = self.offsets[i]
            parent = parent @ local
            self.transforms[i] = parent
        self.T10, self.T20, self.T30, self.T40, self.T50, self.T60 = self.transforms
        self.Tau_G_Motor = self.gravity_torque()

    def gravity_torque(self):
        return self.gravity_at(self.Angle)

    @classmethod
    def gravity_at(cls, angles):
        """按六个 DH 关节角递推电机侧抗重力矩（N·m）。"""
        # 由基座向末端，把竖直向上的补偿加速度转到各关节系。
        gx, gy, gz = 0.0, 0.0, 9.80665
        joints = []
        for angle, ca, sa in zip(angles, cls.twist_cos, cls.twist_sin):
            c, s = cos(angle), sin(angle)
            gy, gz = ca * gy + sa * gz, -sa * gy + ca * gz
            gx, gy = c * gx + s * gy, -s * gx + c * gy
            joints.append((gx, gy, c, s, ca, sa))

        # 由末端向基座合并质量一阶矩 h。绕本节 z 轴的力矩为 (h × g) 的 z 分量。
        hx = hy = hz = 0.0
        torque = np.empty(6)
        for i in range(5, -1, -1):
            gx, gy, c, s, ca, sa = joints[i]
            mx, my, mz = cls._link_moments[i]
            hx, hy, hz = hx + mx, hy + my, hz + mz
            torque[i] = hx * gy - hy * gx
            hx, hy = c * hx - s * hy, s * hx + c * hy
            hy, hz = ca * hy - sa * hz, sa * hy + ca * hz
        return torque / cls.ratio
