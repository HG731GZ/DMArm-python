"""把已有遥操作日志按顺序送入项目控制器，生成上线算法的离线预览。"""

from collections import deque
import argparse
import csv
import json
from pathlib import Path
import sys
from time import perf_counter
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import main as settings
from ArmController import ArmController, HoldParameters
from DMMotor import DMMotor, Feedback, MOTOR_MODELS
from FreeSpacePredictor import FreeSpaceParameters
from Robot_UR import Robot_UR
from TeleopLog import CSV_COLUMNS, TeleopSample
from UdpClass import MasterCommand, SlaveFrame, encode_master


def main():
    parser = argparse.ArgumentParser(description="按当前控制参数离线预览已有轨迹；不连接 CAN 或 UDP。")
    parser.add_argument("source", type=Path, help="输入遥操作日志")
    parser.add_argument("--output", type=Path, required=True, help="预览输出目录")
    parser.add_argument("--prediction", action="store_true", help="在离线预览中启用预测位置补偿")
    parser.add_argument("--weight", type=float, nargs=6, default=settings.FORCE_FEEDBACK_PREDICTION_WEIGHT,
                        help="J1～J6 的预测位置补偿权重，默认读取 main.py")
    args = parser.parse_args()
    raw = np.genfromtxt(args.source, delimiter=",", names=True, encoding="utf-8-sig",
                        missing_values="NULL", filling_values=np.nan)
    p = FreeSpaceParameters(settings.FREE_SPACE_SPEED_LIMIT_RAD_S, settings.FREE_SPACE_ACCEL_POSITIVE_RAD_S2,
                            settings.FREE_SPACE_ACCEL_NEGATIVE_RAD_S2, settings.FREE_SPACE_BRAKING_RAD_S2,
                            settings.FREE_SPACE_DELAY_S, settings.FREE_SPACE_STEP_S)
    motors = [DMMotor(i + 1, model, limit) for i, (model, limit) in enumerate(
        zip(MOTOR_MODELS, settings.MOTOR_OUTPUT_TORQUE_LIMIT_NM))]
    ctl = ArmController(settings.UR_HOME_RAD, settings.GRAVITY_GAIN, settings.GRAVITY_AXES,
                        HoldParameters(**settings.hold_settings()), motors,
                        force_stiffness=settings.FORCE_FEEDBACK_STIFFNESS_NM_PER_RAD,
                        force_damping=settings.FORCE_FEEDBACK_DAMPING_NM_S_PER_RAD,
                        force_velocity_filter_s=settings.FORCE_FEEDBACK_VELOCITY_FILTER_S,
                        force_limit=settings.FORCE_FEEDBACK_LIMIT_NM,
                        force_prediction_enabled=args.prediction, force_prediction_weight=args.weight,
                        free_space_parameters=p)
    t = raw["elapsed_s"]
    groups = {}
    for side, fields in (("master", ("q_rad", "dq_rad_s")),
                         ("slave", ("q_rad", "dq_rad_s", "torque_nm", "ddq_rad_s2"))):
        for field in fields:
            groups[side + "_" + field] = np.column_stack([raw[f"{side}_j{i}_{field}"] for i in range(1, 7)])
    history = deque()
    retention = settings.UDP_TIMEOUT_S + max(p.delay) + 2 * p.step_s
    frame = None
    previous_wall = None
    torque = np.zeros((len(t), 6))
    original = np.zeros_like(torque)
    elapsed = []
    ready_rows = 0
    evaluated = np.zeros(len(t), dtype=bool)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "live_model_preview.csv").open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(CSV_COLUMNS)
        for k, now in enumerate(t):
            master_q = groups["master_q_rad"][k]
            master_dq = groups["master_dq_rad_s"][k]
            history.append(MasterCommand(float(now), tuple(master_q)))
            while len(history) > 1 and history[1].sent_at < now - retention:
                history.popleft()
            wall = raw["slave_received_unix_s"][k]
            if wall != previous_wall:
                received = float(now - (raw["time_unix_s"][k] - wall))
                frame = SlaveFrame(tuple(groups["slave_q_rad"][k]), int(raw["slave_mode"][k]),
                                   tuple(raw[f"slave_gripper_{field}"][k] for field in ("open", "current_raw", "position_raw")),
                                   tuple(groups["slave_dq_rad_s"][k]), tuple(groups["slave_torque_nm"][k]),
                                   int(raw["slave_teleop_state"][k]), tuple(groups["slave_ddq_rad_s2"][k]),
                                   received, float(wall), int(raw["slave_generation"][k]))
                previous_wall = wall
            feedback = tuple(Feedback(float(raw[f"master_j{i}_motor_position_rad"][k]),
                                      float(raw[f"master_j{i}_motor_velocity_rad_s"][k]),
                                      float(raw[f"master_j{i}_motor_torque_nm"][k]), 0, 0,
                                      int(raw[f"master_j{i}_motor_status"][k])) for i in range(1, 7))
            begin = perf_counter()
            ctl.update_velocity_error(feedback, frame.dq, float(now), ctl.hold.max_dt)
            if args.prediction and frame.received >= history[0].sent_at:
                ctl.update_prediction(feedback, frame, tuple(history), float(now), ctl.hold.max_dt)
                ready_rows += 1
            if not args.prediction or ctl.prediction_observation is not None:
                torque[k] = ctl.impedance_torque(feedback, frame.q) * Robot_UR.ratio
                error = Robot_UR.wrap(np.asarray(frame.q) - Robot_UR.motor2dh([fb.position for fb in feedback]))
                original[k] = np.clip(ctl.force_stiffness * error + ctl.local_damping_torque,
                                      -ctl.force_limit, ctl.force_limit)
                evaluated[k] = True
            elapsed.append(perf_counter() - begin)
            grip = Feedback(float(raw["master_gripper_motor_position_rad"][k]),
                            float(raw["master_gripper_motor_velocity_rad_s"][k]),
                            float(raw["master_gripper_motor_torque_nm"][k]), 0, 0,
                            int(raw["master_gripper_motor_status"][k]))
            # 候选力矩写入真实日志序列化入口；开关状态及 MIT 指令列保留原日志，不代表预览力矩已实际输出。
            state = SimpleNamespace(opened=True, faulted=False,
                feedback_at=float(now - raw["master_feedback_max_age_ms"][k] / 1000),
                cycles=int(raw["master_can_cycle"][k]), force_feedback=bool(raw["master_force_feedback_on"][k]),
                motors=tuple(("MIT", fb) for fb in (*feedback, grip)),
                commands=tuple(raw[f"master_j{i}_mit_command_nm"][k] for i in range(1, 7)),
                filtered_velocity_error=tuple(ctl.filtered_velocity_error),
                local_damping_torque=tuple(ctl.local_damping_torque),
                force_prediction_enabled=args.prediction, force_prediction_weight=tuple(ctl.force_prediction_weight),
                prediction=ctl.prediction_observation)
            packet = encode_master(master_q, master_dq, float(raw["master_udp_grip_1"][k]))
            sample = TeleopSample(float(now), float(raw["time_unix_s"][k]), state, packet, frame)
            writer.writerow(sample.csv_row(0.))
    dt = np.r_[np.diff(t), 0.] * evaluated
    duration = sum(dt)
    rms = lambda values: np.sqrt(np.sum(values ** 2 * dt[:, None], axis=0) / duration)
    work = np.sum(np.maximum(0, -torque * groups["master_dq_rad_s"]) * dt[:, None], axis=0)
    original_rms, output_rms = rms(original), rms(torque)
    result = dict(说明="按固定的历史轨迹预览当前控制器与日志序列化；未打开 CAN 或 UDP，不模拟人机闭环。",
                  source=str(args.source), prediction_enabled=args.prediction, prediction_weight=args.weight,
                  stiffness_nm_per_rad=ctl.force_stiffness.tolist(), damping_nm_s_per_rad=ctl.force_damping.tolist(),
                  feedback_limit_nm=ctl.force_limit.tolist(),
                  row_count=len(t), ready_rows=ready_rows, csv_columns=len(CSV_COLUMNS),
                  original_rms_nm=original_rms.tolist(), output_rms_nm=output_rms.tolist(),
                  rms_reduction_percent=[float(100 * (1 - out / base)) if base > 0 else None
                                         for base, out in zip(original_rms, output_rms)],
                  net_feedback_work_j=np.sum(torque * groups["master_dq_rad_s"] * dt[:, None], axis=0).tolist(),
                  resisting_work_j=work.tolist(), total_resisting_work_j=float(sum(work)),
                  computation_ms=np.quantile(np.array(elapsed) * 1000, [.5, .99, 1]).tolist(),
                  slowest_at_elapsed_s=float(t[np.argmax(elapsed)]))
    (args.output / "live_model_preview.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
