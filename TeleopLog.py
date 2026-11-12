"""按遥操作节拍采样，在后台将自动或手动记录的主从数据分次写入 CSV。"""

import csv
from datetime import datetime
from itertools import chain
from queue import SimpleQueue
from threading import Thread
from time import perf_counter, time
from typing import NamedTuple

from CanWorker import JOINTS, Snapshot
from Robot_UR import Robot_UR
from UdpClass import HEADER, MASTER, SlaveFrame


def joint_columns(side, quantity):
    return tuple(f"{side}_j{n}_{quantity}" for n in JOINTS)


MOTOR_NAMES = (*[f"j{n}" for n in JOINTS], "gripper")
MOTOR_FIELDS = ("position_rad", "velocity_rad_s", "torque_nm", "status")
SLAVE_COLUMNS = (
    "slave_received_unix_s", "slave_generation", "slave_mode", "slave_grip_1", "slave_grip_2", "slave_grip_3",
    *joint_columns("slave", "q_rad"), *joint_columns("slave", "dq_rad_s"), *joint_columns("slave", "net_torque_nm"),
)
MASTER_COLUMNS = (
    "master_can_cycle", "master_feedback_max_age_ms", "master_force_feedback_on",
    *joint_columns("master", "q_rad"), "master_udp_mode", "master_udp_grip_1", "master_udp_grip_2", "master_udp_grip_3",
    *joint_columns("master", "dq_rad_s"),
    *(f"master_{name}_motor_{field}" for name in MOTOR_NAMES for field in MOTOR_FIELDS), "master_gripper_force_n",
    *(f"master_j{n}_{field}" for n in JOINTS for field in ("mit_command_nm", "pv_command_rad")),
)
CSV_COLUMNS = ("time_unix_s", "elapsed_s", *MASTER_COLUMNS, *SLAVE_COLUMNS)


class TeleopSample(NamedTuple):
    sampled_at: float
    wall_time: float
    state: Snapshot | None
    packet: bytes | None
    slave: SlaveFrame | None

    def master_motion(self, motors):
        if self.packet is not None:
            # 本拍发过命令时，保留实际发布的 DH 关节角、速度和限幅后的夹钳命令。
            return MASTER.unpack_from(self.packet, len(HEADER))
        # 手动采样可只有部分电机回包，各轴独立换算，四个 UDP 命令字段留空。
        q, dq = [None] * len(JOINTS), [None] * len(JOINTS)
        for i, ((_, fb), ratio) in enumerate(zip(motors, Robot_UR.ratio)):
            if fb is not None:
                q[i] = Robot_UR.wrap(fb.position / ratio)
                dq[i] = fb.velocity / ratio
        return (*q, None, None, None, None, *dq)

    def master_values(self):
        state = self.state
        if state is None or not state.opened or state.faulted:
            return (None,) * len(MASTER_COLUMNS)
        age_ms = (self.sampled_at - state.feedback_at) * 1000 if state.feedback_at else None
        row = [state.cycles, age_ms, int(state.force_feedback), *self.master_motion(state.motors)]
        for _, fb in state.motors:
            if fb is None:
                row.extend((None,) * len(MOTOR_FIELDS))
            else:
                row.extend((fb.position, fb.velocity, fb.torque, fb.status))
        grip = state.motors[6][1]
        row.append(grip.grip_force if grip is not None else None)
        for (mode, _), command in zip(state.motors, state.commands):
            row.extend((command if mode == "MIT" else None, command if mode == "PV" else None))
        return row

    def csv_row(self, started_at):
        frame = self.slave
        row = [self.wall_time, self.sampled_at - started_at, *self.master_values()]
        if frame is None:
            row.extend((None,) * len(SLAVE_COLUMNS))
        else:
            row.extend((frame.wall_time, frame.generation, frame.mode, *frame.grip, *frame.q, *frame.dq, *frame.net_torque))
        return ["NULL" if value is None else value for value in row]


class TeleopLog(Thread):
    _SHUTDOWN = object()

    def __init__(self, directory, report, report_error):
        super().__init__(name="DMArm-Teleop-Log")
        self.directory = directory
        self.report = report
        self.report_error = report_error
        self.samples = SimpleQueue()
        self.first_sample = None

    def record(self, state, packet, slave):
        sample = TeleopSample(perf_counter(), time(), state, packet, slave)
        if self.first_sample is None:
            self.first_sample = sample
        self.samples.put(sample)

    def end_session(self):
        if self.first_sample is not None:
            self.samples.put(None)
            self.first_sample = None

    def close(self):
        self.end_session()
        self.samples.put(self._SHUTDOWN)
        self.join()

    def run(self):
        for first in iter(self.samples.get, self._SHUTDOWN):
            remaining = iter(self.samples.get, None)
            stamp = datetime.fromtimestamp(first.wall_time).strftime("%Y%m%d_%H%M%S_%f")
            path = self.directory / f"teleop_{stamp}.csv"
            try:
                self.directory.mkdir(exist_ok=True)
                with path.open("x", encoding="utf-8-sig", newline="") as output:
                    writer = csv.writer(output)
                    writer.writerow(CSV_COLUMNS)
                    self.report(f"开始记录遥操作：{path}")
                    for sample in chain((first,), remaining):
                        writer.writerow(sample.csv_row(first.sampled_at))
            except OSError as error:
                self.report_error(first, f"本次遥操作日志写入失败：{path}，{error}")
                # 本次失败只提示一次，消费至该段末尾后再处理下一次遥操作。
                for _ in remaining:
                    pass
            else:
                self.report(f"遥操作日志已保存：{path}")
