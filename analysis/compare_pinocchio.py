"""用主端日志和固定采样离线比较 Pinocchio 与迁移前的 DH 计算，报告误差和单次调用耗时。"""

import argparse
import csv
from functools import partial
from pathlib import Path
import sys
from time import perf_counter_ns
from types import SimpleNamespace

import numpy as np
import pinocchio as pin

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ArmController import ArmController
from DMMotor import DMMotor, Feedback, MOTOR_MODELS
from Robot_UR import ASSEMBLY_DH_RAD, ArmModel, Robot_UR
from analysis.legacy_ur_73566ae import Robot_UR as LegacyRobot
from main import ARM_MODEL_URDF, GRAVITY_AXES, GRAVITY_GAIN, MOTOR_OUTPUT_TORQUE_LIMIT_NM, UR_HOME_RAD


ZERO_VELOCITY = np.zeros(6)
GAIN = np.array(GRAVITY_GAIN)
AXES = np.array(GRAVITY_AXES, dtype=float)


class ComparisonArm(ArmModel):
    """让数值核对和计时调用实际的姿态与前馈入口。"""

    def __init__(self, model):
        super().__init__(model)
        self.data = model.createData()
        controller = SimpleNamespace(arm_model=self, gravity_data=model.createData(),
                                     gravity_gain=GAIN, gravity_axes=AXES)
        self.gravity_commands = partial(ArmController.gravity_commands, controller)

    def gravity_at(self, dh):
        return self.motor_gravity(dh, self.data)

    def poses_at(self, dh):
        angles = Robot_UR._wrapped_joints(dh)
        transforms, _ = self._link_poses(self.configuration(angles), self.data)
        return angles, transforms

    def state(self, dh):
        pose = self.pose(dh, ZERO_VELOCITY, self.data)
        return pose.Angle, pose.transforms, pose.Tau_G_Motor


def legacy_commands(feedback):
    angles = LegacyRobot.motor2dh([f.position for f in feedback])
    return LegacyRobot.gravity_at(angles) * GAIN * AXES


def legacy_state(dh):
    arm = LegacyRobot(dh)
    return arm.Angle, arm.transforms, arm.Tau_G_Motor


class LegacyPoses(LegacyRobot):
    """单独计时现有正运动学时，跳过构造器末尾的重力矩调用。"""

    def gravity_torque(self):
        return None


def legacy_poses(dh):
    arm = LegacyPoses(dh)
    return arm.Angle, arm.transforms


def old_mass_model(urdf, axes, use_old_mass):
    """在内存副本中换回旧质心，可同时换回旧质量，用于定位静态力矩差。"""
    model = urdf.model.copy()
    for i in axes:
        joint = model.getJointId(f"joint{i + 1}")
        frame = model.frames[urdf.frame_ids[i + 1]]
        old = LegacyRobot.mass_com[i]
        com_link = (urdf.cad_from_dh[i] @ np.r_[old[1:], 1.])[:3]
        inertia = model.inertias[joint]
        mass = old[0] if use_old_mass else inertia.mass
        model.inertias[joint] = pin.Inertia(mass, frame.placement.act(com_link), inertia.inertia)
    return ComparisonArm(model)


def load_log(path):
    columns = [f"master_j{i}_q_rad" for i in range(1, 7)]
    with path.open(encoding="utf-8-sig", newline="") as file:
        header = next(csv.reader(file))
        indices = [header.index(name) for name in columns]
        motor_columns = [f"master_j{i}_motor_position_rad" for i in range(1, 7)]
        if all(name in header for name in motor_columns):
            indices.extend(header.index(name) for name in motor_columns)
        angles = np.genfromtxt(file, delimiter=",", usecols=indices, missing_values="NULL", ndmin=2)
    complete = angles[np.isfinite(angles[:, :6]).all(axis=1), :6]
    if not len(complete):
        raise ValueError("日志中没有完整的六轴主端关节角")
    motor = angles[:, 6:]
    return complete, motor[np.isfinite(motor).all(axis=1)]


def compare_poses(label, samples, arms, old_com, old_mass):
    reference = [legacy_state(q) for q in samples]
    old_t = np.stack([state[1] for state in reference])
    old_g = np.stack([state[2] for state in reference])
    print(f"\n{label}，共 {len(samples)} 组")
    for name, arm in arms.items():
        states = [arm.state(q) for q in samples]
        t = np.stack([state[1] for state in states])
        g = np.stack([state[2] for state in states])
        position = np.linalg.norm(t[:, :, :3, 3] - old_t[:, :, :3, 3], axis=2)
        # 两旋转阵之差的 Frobenius 范数为 2√2 sin(θ/2)，小角度也能分辨。
        diff = np.linalg.norm(t[:, :, :3, :3] - old_t[:, :, :3, :3], axis=(2, 3))
        rotation = 2 * np.arcsin(np.clip(diff / (2 * np.sqrt(2)), 0., 1.))
        print(name, "各连杆最大位置差(m)", position.max(axis=0))
        print(name, "各连杆最大转角差(rad)", rotation.max(axis=0))
        print(name, "各轴重力矩最大绝对差(N·m)", np.max(np.abs(g - old_g), axis=0))
        print(name, "各轴重力矩均方根差(N·m)", np.sqrt(np.mean((g - old_g) ** 2, axis=0)))
    for note, arm in (("仅换回 J6 旧质心", old_com), ("换回六轴旧质量和质心", old_mass)):
        delta = np.array([arm.gravity_at(q) for q in samples]) - old_g
        print(note, "最大绝对差(N·m)", np.max(np.abs(delta)))


def compare_commands(label, positions, arm, motors):
    """从原始电机位置经过前馈与 MIT 编码，分别统计浮点差和力矩码差。"""
    float_max = 0.
    code_max = changed_frames = changed_codes = 0
    for row in positions:
        feedback = [Feedback(p, 0., 0., 0, 0, 1) for p in row]
        old, new = legacy_commands(feedback), arm.gravity_commands(feedback)
        float_max = max(float_max, np.max(np.abs(old - new)))
        differences = []
        for motor, a, b in zip(motors, old, new):
            old_code, new_code = (int.from_bytes(motor.encode_mit(0., 0., t, 0., 0.)[-2:], "big") & 0xfff
                                  for t in (a, b))
            differences.append(abs(old_code - new_code))
        changed_frames += any(differences)
        changed_codes += np.count_nonzero(differences)
        code_max = max(code_max, max(differences))
    print(f"{label}：{len(positions)} 组，前馈最大差 {float_max:.8g} N·m，"
          f"MIT 力矩码有差异 {changed_frames} 组 / {changed_codes} 个轴次，最大码差 {code_max}")


def compare_dynamics(arms, samples, rng):
    """以相同的关节速度、加速度比较两份 CAD 模型的刚体动力学。"""
    maxima = np.zeros(3)
    for q in samples:
        dq = rng.uniform(-1., 1., 6)
        ddq = rng.uniform(-3., 3., 6)
        results = []
        for arm in arms.values():
            config = arm.configuration(q)
            m, d = arm.model, arm.data
            results.append((pin.computeGeneralizedGravity(m, d, config).copy(),
                            pin.crba(m, d, config).copy(), pin.rnea(m, d, config, dq, ddq).copy()))
        maxima = np.maximum(maxima, [np.max(np.abs(a - b)) for a, b in zip(*results)])
    print("\nURDF 与 MJCF 的重力矩(N·m)、质量矩阵(kg·m²)、逆动力学(N·m)最大元素差", maxima)


def timings(samples, arms):
    feedback = [tuple(Feedback(float(p), 0., 0., 0, 0, 1) for p in q * Robot_UR.ratio) for q in samples]
    jobs = {
        "DH 角到电机重力矩": (samples, [LegacyRobot.gravity_at] + [a.gravity_at for a in arms.values()]),
        "六轴电机反馈到前馈指令": (feedback, [legacy_commands]
                                  + [a.gravity_commands for a in arms.values()]),
        "仅六个控制侧位姿": (samples, [legacy_poses] + [a.poses_at for a in arms.values()]),
        "六个控制侧位姿和电机重力矩": (samples, [legacy_state] + [a.state for a in arms.values()]),
    }
    for label, (inputs, impls) in jobs.items():
        times = np.empty((len(impls), 12, len(inputs)), dtype=np.int64)
        for run in impls:
            for x in inputs:
                run(x)
        for repeat in range(times.shape[1]):
            for offset in range(len(impls)):
                i = (repeat + offset) % len(impls)
                run = impls[i]
                for k, x in enumerate(inputs):
                    start = perf_counter_ns()
                    run(x)
                    times[i, repeat, k] = perf_counter_ns() - start
        for name, elapsed in zip(("迁移前", *arms), times):
            print(label, name, "单次 P50 / P95 (µs)", np.percentile(elapsed, [50, 95]) / 1000.)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path, help="含主端六轴 DH 角的遥操作 CSV")
    parser.add_argument("--numeric-only", action="store_true", help="仅核对数值，跳过耗时测量")
    args = parser.parse_args()
    arms = {"URDF": ComparisonArm.from_urdf(ARM_MODEL_URDF),
            "MJCF": ComparisonArm(pin.buildModelFromMJCF(str(ARM_MODEL_URDF.with_suffix(".xml"))))}
    print("Python", sys.version.split()[0], "NumPy", np.__version__, "Pinocchio", pin.__version__)
    for name, arm in arms.items():
        print(name, list(arm.model.names)[1:], "nq / nv", arm.model.nq, arm.model.nv)

    urdf = arms["URDF"]
    tool = urdf.model.frames[urdf.frame_ids[6]]
    cad_com = tool.placement.actInv(urdf.model.inertias[tool.parentJoint].lever)
    com_dh = (np.linalg.inv(urdf.cad_from_dh[5]) @ np.r_[cad_com, 1.])[:3]
    print("J6 旧质心与 CAD 质心（控制侧 DH6，mm）", 1000 * LegacyRobot.mass_com[5, 1:], 1000 * com_dh)
    old_com = old_mass_model(urdf, [5], False)
    old_mass = old_mass_model(urdf, range(6), True)
    log, motor_positions = load_log(args.log)
    rng = np.random.default_rng(20261127)
    synthetic = rng.uniform(-np.pi, np.pi, (4096, 6))
    np.set_printoptions(precision=8)
    compare_poses("日志", log, arms, old_com, old_mass)
    samples = np.vstack([np.zeros(6), ASSEMBLY_DH_RAD, UR_HOME_RAD, synthetic])
    compare_poses("控制零位、装配位、预设位及固定采样", samples, arms, old_com, old_mass)
    motors = [DMMotor(i, model, limit) for i, (model, limit) in
              enumerate(zip(MOTOR_MODELS, MOTOR_OUTPUT_TORQUE_LIMIT_NM), 1)]
    if motor_positions.size:
        compare_commands("日志原始电机位置", motor_positions, urdf, motors)
    compare_commands("基准姿态及固定采样", samples * Robot_UR.ratio, urdf, motors)
    compare_dynamics(arms, synthetic, rng)
    for name, method in (("迁移前", LegacyRobot.gravity_at), ("URDF", urdf.gravity_at)):
        print("项目预设位", name, "六轴电机重力矩(N·m)", method(np.array(UR_HOME_RAD)))
    if args.numeric_only:
        return
    # 一半取等间隔日志行，一半取全角域样本。对同一输入序列轮换三种实现的执行顺序。
    inputs = np.empty((1024, 6))
    inputs[::2] = log[np.linspace(0, len(log) - 1, 512, dtype=int)]
    inputs[1::2] = synthetic[:512]
    timings(inputs, arms)


if __name__ == "__main__":
    main()
