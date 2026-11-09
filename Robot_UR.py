"""UR 同构主端的关节换算、正运动学与静态重力矩。"""

import numpy as np


class Robot_UR:
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

    @staticmethod
    def wrap(q):
        return np.pi - np.remainder(np.pi - q, 2 * np.pi)

    @classmethod
    def motor2dh(cls, positions):
        return cls.wrap(np.asarray(positions) / cls.ratio)

    @classmethod
    def from_feedback(cls, feedback):
        return cls(cls.motor2dh([f.position for f in feedback]),
                   np.array([f.velocity for f in feedback]) / cls.ratio)

    def __init__(self, q, velocity=None):
        q = np.asarray(q, dtype=float)
        if q.shape != (6,) or not np.isfinite(q).all():
            raise ValueError("姿态需要六个有限的关节角（rad）")
        self.Angle = self.wrap(q)
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
        """在基座系把各节的抗重力矩投到上游关节轴，换算为电机侧 N·m。"""
        origins = self.transforms[:, :3, 3]
        rotations = self.transforms[:, :3, :3]
        centers = origins + (rotations @ self.mass_com[:, 1:, None])[:, :, 0]
        forces = np.zeros((6, 3))
        forces[:, 2] = 9.80665 * self.mass_com[:, 0]
        torque = np.empty(6)
        for i in range(6):
            moment = np.cross(centers[i:] - origins[i], forces[i:]).sum(axis=0)
            torque[i] = moment @ rotations[i, :, 2]
        return torque / self.ratio
