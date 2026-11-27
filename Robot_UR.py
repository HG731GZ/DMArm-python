"""UR 主端的电机换算、姿态快照和 Pinocchio 刚体模型。"""

import numpy as np
import pinocchio as pin


# CAD 装配零位对应的控制侧 DH 角，按 J1～J6 排列，单位 rad。
ASSEMBLY_DH_RAD = np.array((0, -np.pi / 2, 0, -np.pi / 2, 0, 0))
CAD_LINKS = ("base_link", *(f"DH{i}" for i in range(1, 7)))


class Robot_UR:
    """保存一帧六轴姿态，并提供电机与 DH 关节的无状态换算。"""

    # 电机角 / DH 关节角的带符号传动比，J3 的电机正向与关节正向相反。
    ratio = np.array([1, 1, -1, 1, 1, 1])

    def __init__(self, angles, velocity, transforms, cad_poses, motor_gravity):
        self.Angle = angles
        self.Velocity = velocity
        self.transforms = transforms
        self.T10, self.T20, self.T30, self.T40, self.T50, self.T60 = transforms
        self.cad_poses = cad_poses
        self.Tau_G_Motor = motor_gravity

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


class ArmModel:
    """共享已初始化的模型。每个调用线程持有独立的 Pinocchio Data。"""

    @classmethod
    def from_urdf(cls, path):
        return cls(pin.buildModelFromUrdf(str(path)))

    def __init__(self, model):
        self.model = model
        model.gravity.linear = np.array((0., 0., -9.80665))
        self.neutral = pin.neutral(model)
        self.frame_ids = [model.getFrameId(name, pin.BODY) for name in CAD_LINKS]
        self.assembly = self._cad_poses(self.neutral, model.createData())
        self.control_from_world = np.linalg.inv(self.assembly[1])
        # J2、J3 的控制 DH 系相对 CAD 连杆绕局部 Z 轴转半周。
        self.cad_from_dh = np.repeat(np.eye(4)[None], 6, axis=0)
        self.cad_from_dh[1:3] = np.diag((-1., -1., 1., 1.))

    def configuration(self, q_dh):
        return pin.integrate(self.model, self.neutral, q_dh - ASSEMBLY_DH_RAD)

    def _cad_poses(self, config, data):
        pin.framesForwardKinematics(self.model, data, config)
        placements = data.oMf
        return np.stack([placements[i].homogeneous for i in self.frame_ids])

    def _link_poses(self, config, data):
        cad_poses = self._cad_poses(config, data)
        return self.control_from_world @ cad_poses[1:] @ self.cad_from_dh, cad_poses

    def _motor_gravity(self, config, data):
        return pin.computeGeneralizedGravity(self.model, data, config) / Robot_UR.ratio

    def motor_gravity(self, q_dh, data):
        """由本拍 DH 角求六轴倍率前的电机抗重力矩，单位 N·m。"""
        return self._motor_gravity(self.configuration(q_dh), data)

    def pose(self, q_dh, dq_dh, data):
        """从同一配置取得 CAD 世界位姿、控制 DH 位姿及电机力矩的独立快照。"""
        angles = Robot_UR._wrapped_joints(q_dh)
        config = self.configuration(angles)
        transforms, cad_poses = self._link_poses(config, data)
        return Robot_UR(angles, np.array(dq_dh, copy=True), transforms, cad_poses,
                        self._motor_gravity(config, data))
