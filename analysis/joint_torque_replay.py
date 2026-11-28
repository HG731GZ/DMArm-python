"""用已确认的无接触记录拟合六轴空载力矩，离线比较三种材料的主端反馈。"""

import ast
import json
import sys
from bisect import bisect_right
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from joint_torque_figures import make_figures

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from FreeSpacePredictor import FreeSpaceParameters, FreeSpacePredictor

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"
FREE = LOGS / "teleop_20261124_114847_442757.csv"
MATERIALS = LOGS / "力反馈实验-铁块硬海绵泡泡纸"
OUT = LOGS / "关节力矩离线回放_20261128"
SLAVE_LIMIT = np.array([24.0, 24.0, 18.0, 6.0, 6.0, 6.0])  # 无接触辨识文件实际出现的双向平台，N·m。
GAP = 0.050  # 回包或可用观测超过该间隔时中断持续判定，s。
FILTER = 0.020  # 力矩残差因果低通时间常数，s。
DWELL = 0.030  # 连续达到反馈判据的最短时长，s。
LEVEL = 0.20  # 主端关节侧反馈变化的工程判据，N·m。
RIDGE = 0.02  # 标准化特征岭回归的均方损失正则系数。
EARLY, MAX_WEIGHT, GAIN = 0.20, 0.75, 0.20  # 小提示占比、最大混合权重、关节力矩反射比例。
BLOCKS = [0.0, 14.0, 29.0, 38.5, 45.0, 52.5, 65.0]  # 空载记录各主激励块的边界，s。
FAMILIES = ("单轴运动", "六轴耦合运动", "姿态与六轴耦合运动")

# 仅用于对既有按压录像式回放划窗，三列为下行起点、进入末段、预测进给最低点，单位 s。
# 根据项目 URDF 的预测 DH6 高度标注，末段从首次进入最低点上方 20 模型 mm 开始。
# 窗口和方向只供事后评分，六轴反馈计算不使用这些时刻、材料名称或法兰高度。
WINDOWS = {
    "铁块": [
        (13.14, 14.724, 18.002),
        (19.05, 20.021, 21.962),
        (23.03, 24.432, 26.921),
        (27.93, 28.587, 31.028),
    ],
    "硬海绵": [
        (7.50, 11.657, 14.295),
        (15.33, 17.060, 18.830),
        (20.34, 22.025, 23.310),
        (24.89, 25.541, 27.776),
        (28.80, 29.467, 30.571),
    ],
    "泡泡纸": [
        (4.66, 7.634, 10.413),
        (11.85, 13.566, 14.998),
        (16.18, 18.179, 19.402),
        (20.83, 21.834, 22.732),
    ],
}


def settings():
    """从主入口取反馈与预测常量，不加载界面和硬件。"""
    names = (
        "FORCE_FEEDBACK_STIFFNESS_NM_PER_RAD",
        "FORCE_FEEDBACK_DAMPING_NM_S_PER_RAD",
        "FORCE_FEEDBACK_LIMIT_NM",
        "FORCE_FEEDBACK_PREDICTION_WEIGHT",
        "FREE_SPACE_SPEED_LIMIT_RAD_S",
        "FREE_SPACE_ACCEL_POSITIVE_RAD_S2",
        "FREE_SPACE_ACCEL_NEGATIVE_RAD_S2",
        "FREE_SPACE_BRAKING_RAD_S2",
        "FREE_SPACE_DELAY_S",
        "FREE_SPACE_STEP_S",
    )
    definitions = {
        n.targets[0].id: n.value
        for n in ast.parse((ROOT / "main.py").read_text("utf-8-sig")).body
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
    }
    return [np.asarray(ast.literal_eval(definitions[k])) for k in names]


K, B, MASTER_LIMIT, ALPHA, SPEED, ACC_POS, ACC_NEG, BRAKE, DELAY, STEP = settings()


def cols(side, field):
    return [f"{side}_j{j}_{field}" for j in range(1, 7)]


def six(frame, side, field):
    return frame[cols(side, field)].to_numpy(dtype=float)


def wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


class Trace(NamedTuple):
    rx: np.ndarray
    t: np.ndarray
    q: np.ndarray
    v: np.ndarray
    acc: np.ndarray
    tau: np.ndarray
    qp: np.ndarray
    vp: np.ndarray
    base: np.ndarray
    damping: np.ndarray
    spring: np.ndarray


class Command(NamedTuple):
    sent_at: float
    q: np.ndarray


def load_trace(path, free=False, *, feedback_on=True):
    """配对同一从端包。输出时刻取首次记录到相应控制观测的主机时刻。"""
    d = pd.read_csv(path, na_values=["NULL"], float_precision="round_trip")
    origin = d.time_unix_s.iloc[0]
    rec, pair = "slave_received_unix_s", "master_prediction_slave_received_unix_s"
    packets = d.loc[d.slave_teleop_state.eq(2)].drop_duplicates(rec).copy()
    all_rx = packets[rec].to_numpy() - origin
    all_torque = six(packets, "slave", "torque_nm")
    if free:
        sent = d.loc[d.master_udp_mode.notna()]
        sent_t = (sent.time_unix_s - origin).to_numpy()
        commands = [Command(t, q) for t, q in zip(sent_t, six(sent, "master", "q_rad"))]
        a = packets.loc[packets[rec] - origin >= sent_t[0]]
        rx = a[rec].to_numpy() - origin
        q, v = six(a, "slave", "q_rad"), six(a, "slave", "dq_rad_s")
        predictor = FreeSpacePredictor(
            FreeSpaceParameters(SPEED, ACC_POS, ACC_NEG, BRAKE, DELAY, float(STEP))
        )
        predictor.initialize(rx[0], q[0], v[0])
        predictions = []
        previous = rx[0]
        for stamp in rx:
            left = max(0, bisect_right(sent_t, previous - DELAY.max() - 2 * STEP) - 1)
            right = bisect_right(sent_t, stamp)
            predictions.append(predictor.advance(stamp, commands[left:right]))
            previous = stamp
        qp, vp = (np.stack(s) for s in zip(*predictions))
        damping = -B * six(a, "master", "dq_rad_s")
        raw_error = wrap(q - six(a, "master", "q_rad"))
        raw = K * raw_error
        spring = np.clip(
            K * ((1 - ALPHA) * raw_error + ALPHA * wrap(q - qp)), np.minimum(0.0, raw), np.maximum(0.0, raw)
        )
        base = np.clip(spring + damping, -MASTER_LIMIT, MASTER_LIMIT)
        observed = len(packets)
    else:
        obs = d.loc[
            d.master_force_feedback_on.eq(int(feedback_on)) & d.master_prediction_ready.eq(1)
        ].drop_duplicates(pair)
        observed = len(obs)
        payload = [rec] + sum(
            [cols("slave", f) for f in ("q_rad", "dq_rad_s", "ddq_rad_s2", "torque_nm")], []
        )
        a = obs.drop(columns=payload).merge(
            packets[payload], left_on=pair, right_on=rec, validate="one_to_one"
        )
        rx = a[rec].to_numpy() - origin
        q, v = six(a, "slave", "q_rad"), six(a, "slave", "dq_rad_s")
        qp, vp = six(a, "master", "predicted_slave_q_rad"), six(a, "master", "predicted_slave_dq_rad_s")
        base, damping = six(a, "master", "weighted_feedback_nm"), six(a, "master", "local_damping_nm")
        # 原材料记录均未限幅。若候选已限幅，这里给出与日志相符的最小弹簧幅值；
        # 补录分析对这些帧另外计算融合差值的上下界。
        spring = base - damping
    trace = Trace(
        rx,
        a.time_unix_s.to_numpy() - origin,
        q,
        v,
        six(a, "slave", "ddq_rad_s2"),
        six(a, "slave", "torque_nm"),
        qp,
        vp,
        base,
        damping,
        spring,
    )
    if (
        not all(np.isfinite(s).all() for s in trace)
        or (np.diff(rx) <= 0).any()
        or (np.diff(trace.t) <= 0).any()
    ):
        raise ValueError(f"{path.name} 的有效输入需要严格递增的时刻和有限的六轴观测")
    return trace, all_rx, all_torque, observed


def runs(mask, t, maximum_gap=GAP):
    """返回连续真值区段的首尾索引，数据中断处拆段。"""
    ids = np.flatnonzero(mask)
    cuts = np.flatnonzero((np.diff(ids) > 1) | (np.diff(t[ids]) > maximum_gap)) + 1
    return [(part[0], part[-1]) for part in np.split(ids, cuts) if len(part)]


def reversal_events(tr):
    """从六轴实测速度找方向反转；选择过程不读取力矩或回放结果。"""
    events = []
    for j in range(6):
        bouts = []
        for sign in (-1, 1):
            for a, b in runs(sign * tr.v[:, j] >= 0.45 * SPEED[j], tr.rx):
                if tr.rx[b] - tr.rx[a] >= 0.030:
                    bouts.append((a, b, sign))
        bouts.sort()
        for (_, b, old), (a, _, new) in zip(bouts, bouts[1:]):
            left, right = tr.rx[b], tr.rx[a]
            if old == new or right - left > 0.60:
                continue
            start, end = left - 0.10, right + 0.12
            x = tr.rx[(tr.rx >= start) & (tr.rx <= end)]
            gap = np.diff(np.r_[start, x, end]).max()
            events.append(
                dict(
                    axis=j + 1,
                    start=start,
                    end=end,
                    turn_s=right - left,
                    max_gap_s=gap,
                    complete=bool(gap <= GAP),
                )
            )
    return sorted(events, key=lambda e: e["start"])


def split_free(tr, events):
    """每个激励块的前半训练、随后四分之一标定、末段核对。完整快速换向整窗额外留出。"""
    t = tr.rx
    block = np.searchsorted(BLOCKS[1:-1], t)
    edges = np.array([*BLOCKS[:-1], t[-1] + 0.001])
    progress = (t - edges[block]) / (edges[block + 1] - edges[block])
    role = np.select([progress < 0.50, progress < 0.75], ["拟合", "标定"], default="留出")
    for e in events:
        if e["complete"]:
            role[(t >= e["start"] - 0.12) & (t <= e["end"] + 0.12)] = "留出"
    borders = t[1:][role[1:] != role[:-1]]
    for b in borders:
        role[(role != "留出") & (abs(t - b) < 0.12)] = "隔离"
    return role


def features(tr):
    """姿态谐波、各轴速度和记录的加速度，以及低速平滑的方向摩擦项。"""
    return np.column_stack([np.sin(tr.q), np.cos(tr.q), tr.v, tr.acc, np.tanh(tr.v / 0.05)])


def torque_model(x_all, tau, training, family):
    x = x_all[training]
    mean, scale = x.mean(axis=0), x.std(axis=0)
    z = (x - mean) / scale
    target = tau[training]
    offset = target.mean(axis=0)
    coef = np.zeros((z.shape[1], 6))
    for j in range(6):
        terms = {
            "单轴运动": [12 + j, 18 + j, 24 + j],
            "六轴耦合运动": np.arange(12, 30),
            "姿态与六轴耦合运动": np.arange(30),
        }[family]
        part = z[:, terms]
        coef[terms, j] = np.linalg.solve(
            part.T @ part + RIDGE * len(z) * np.eye(len(terms)), part.T @ (target[:, j] - offset[j])
        )
    return dict(family=family, mean=mean, scale=scale, coef=coef, offset=offset)


def nominal_torque(x, model):
    z = (x - model["mean"]) / model["scale"]
    return np.clip(z @ model["coef"] + model["offset"], -SLAVE_LIMIT, SLAVE_LIMIT)


def write_coefficients(model):
    """将标准化系数还原到原始输入单位，输出可直接手算的模型。"""
    coef = model["coef"] / model["scale"][:, None]
    bias = model["offset"] - model["mean"] @ coef
    terms = [
        f"{term}_J{j}"
        for term in ("角度正弦", "角度余弦", "速度_rad_s", "加速度_rad_s2", "双曲正切速度除0.05")
        for j in range(1, 7)
    ]
    rows = []
    for j, b in enumerate(bias):
        rows.append((j + 1, "常数", b))
        rows.extend((j + 1, name, k) for name, k in zip(terms, coef[:, j]) if k != 0)
    pd.DataFrame(rows, columns=["输出关节", "输入项", "原单位系数"]).to_csv(
        OUT / "六轴空载系数.csv", index=False, encoding="utf-8-sig"
    )


def select_model(tr, fit):
    """仅在外层拟合区内轮换连续 0.9 秒小块，验证块前后各留 0.12 秒间隔。"""
    x = features(tr)
    fold = np.floor(tr.rx / 0.9).astype(int) % 4
    rows = []
    for family in FAMILIES:
        errors = []
        for k in range(4):
            val = fit & (fold == k)
            train = fit.copy()
            for a, b in runs(val, tr.rx):
                train[(tr.rx >= tr.rx[a] - 0.12) & (tr.rx <= tr.rx[b] + 0.12)] = False
            m = torque_model(x, tr.tau, train, family)
            errors.append(nominal_torque(x[val], m) - tr.tau[val])
        rmse = np.sqrt(np.mean(np.vstack(errors) ** 2, axis=0))
        rows.append(
            dict(
                model=family,
                mean_normalized_rmse=np.mean(rmse / SLAVE_LIMIT),
                **{f"j{j + 1}_rmse_nm": v for j, v in enumerate(rmse)},
            )
        )
    comparison = pd.DataFrame(rows)
    chosen = comparison.loc[comparison.mean_normalized_rmse.idxmin(), "model"]
    return torque_model(x, tr.tau, fit, chosen), comparison


def residuals(tr, model):
    nominal = nominal_torque(features(tr), model)
    effort = tr.tau - nominal
    filtered = effort.copy()
    fresh = np.ones(len(tr.t), dtype=bool)
    since = tr.t[0]
    for i in range(len(tr.t)):
        if i == 0 or max(tr.t[i] - tr.t[i - 1], tr.rx[i] - tr.rx[i - 1]) > GAP:
            since = tr.t[i]
        else:
            a = -np.expm1(-(tr.t[i] - tr.t[i - 1]) / FILTER)
            filtered[i] = filtered[i - 1] + a * (effort[i] - filtered[i - 1])
        fresh[i] = tr.t[i] - since >= 3 * FILTER
    motion = np.column_stack([wrap(tr.q - tr.qp), tr.v - tr.vp])
    return nominal, filtered, motion, fresh


def score(values, scale):
    return np.sqrt(np.mean((values / scale) ** 2, axis=1))


def calibrate(effort, motion, mask, percentiles=(0.95, 0.99)):
    result = {}
    for key, values in (("effort", effort), ("motion", motion)):
        sample = values[mask]
        scale = np.quantile(abs(sample), 0.9, axis=0)
        low, high = np.quantile(score(sample, scale), percentiles)
        result[key] = dict(scale=scale, low=low, high=high)
    return result


def replay(tr, model, calibration, *, early=EARLY):
    nominal, effort, motion, fresh = residuals(tr, model)
    evidence = []
    for key, values in (("effort", effort), ("motion", motion)):
        cal = calibration[key]
        evidence.append(
            np.clip((score(values, cal["scale"]) - cal["low"]) / (cal["high"] - cal["low"]), 0, 1)
        )
    weight = MAX_WEIGHT * evidence[0] * (early + (1 - early) * evidence[1])
    fused = np.clip(
        (1 - weight[:, None]) * tr.spring - weight[:, None] * GAIN * effort + tr.damping,
        -MASTER_LIMIT,
        MASTER_LIMIT,
    )
    return dict(
        nominal=nominal,
        effort=effort,
        fresh=fresh,
        weight=weight,
        fused=fused,
        delta=fused - tr.base,
        ce=evidence[0],
        cq=evidence[1],
    )


def confirmed(signal, tr, valid, level=LEVEL):
    """确认时刻取持续条件满足后的首个观测，不回填到越阈的第一帧。"""
    result = []
    for a, b in runs((signal >= level) & valid, tr.t):
        stamp = np.searchsorted(tr.t, tr.t[a] + DWELL)
        if stamp <= b:
            result.append((tr.t[a], tr.t[stamp], tr.t[b]))
    return result


def onset(events, left, right):
    """单列入窗时已经越阈且随后得到确认的提示，避免算作本次新出现。"""
    for start, hit, end in events:
        if start < left <= end and hit <= right:
            return np.nan, "入窗已起"
        if left <= start and hit <= right:
            return hit, "新出现"
    return np.nan, "未达判据"


def first_cap(rx, torque, left, right, fraction=1.0):
    """任何轴首次达到平台，即使只持续一个原始回包也计入。"""
    meets = abs(torque) >= SLAVE_LIMIT * fraction - 1e-6
    i = np.searchsorted(rx, left, side="right") - 1
    if i >= 0 and left - rx[i] <= GAP and meets[i].any():
        return left, "/".join(str(j + 1) for j in np.flatnonzero(meets[i])), True
    found = np.flatnonzero((rx >= left) & (rx <= right) & meets.any(axis=1))
    if not len(found):
        return np.nan, "", False
    i = found[0]
    return rx[i], "/".join(str(j + 1) for j in np.flatnonzero(meets[i])), False


def timing_outcome(sat, at_entry, hint, state):
    if at_entry:
        return "入窗已限幅"
    if not np.isfinite(sat):
        return "窗口未限幅"
    if state != "新出现":
        return state
    if abs(sat - hint) <= 0.020:
        return "时序接近"
    return "限幅前" if sat > hint else "限幅后"


def sustained_caps(rx, torque):
    """同轴同号连续处于精确平台至少 30 ms，事后以该段首包为平台起点。"""
    caps = []
    for j, limit in enumerate(SLAVE_LIMIT):
        for sign in (-1, 1):
            for a, b in runs(sign * torque[:, j] >= limit - 1e-6, rx):
                if rx[b] - rx[a] >= DWELL:
                    caps.append((rx[a], rx[b], j + 1))
    return sorted(caps)


def press_metrics(name, tr, result, rx, torque):
    rows, panels = [], []
    q = np.unwrap(tr.qp, axis=0)
    caps = sustained_caps(rx, torque)
    for number, (approach, left, right) in enumerate(WINDOWS[name], 1):
        axis = np.array([np.interp(right, tr.rx, v) - np.interp(left, tr.rx, v) for v in q.T])
        axis /= np.linalg.norm(axis)
        old = -tr.base @ axis
        new = -result["fused"] @ axis
        extra = new - old
        row = dict(material=name, press=number, approach_s=approach, start_s=left, end_s=right)
        for key, y in (("old", old), ("new", new), ("extra", extra)):
            events = confirmed(y, tr, result["fresh"])
            row[key + "_s"], row[key + "_state"] = onset(events, left, right)
        sat, axes, at_entry = first_cap(rx, torque, left, right)
        approach_sat, first_axes, first_at_entry = first_cap(rx, torque, approach, right)
        full_direction = np.array([np.interp(right, tr.rx, v) - np.interp(approach, tr.rx, v) for v in q.T])
        full_direction /= np.linalg.norm(full_direction)
        full_hint, full_state = onset(
            confirmed(-result["delta"] @ full_direction, tr, result["fresh"]), approach, right
        )
        near, _, _ = first_cap(rx, torque, left, right, 0.95)
        plateau = next(((a, b, j) for a, b, j in caps if b >= left and a <= right), None)
        plat_s = max(left, plateau[0]) if plateau else np.nan
        plat_entry = bool(plateau and plateau[0] < left)
        lead = (sat - row["extra_s"]) * 1000
        inside = (tr.t >= left) & (tr.t <= right) & result["fresh"]
        row.update(
            sat_s=sat,
            sat_axes=axes,
            capped_at_entry=at_entry,
            first_downstroke_cap_s=approach_sat,
            first_downstroke_axes=first_axes,
            full_hint_s=full_hint,
            full_hint_state=full_state,
            full_outcome=timing_outcome(approach_sat, first_at_entry, full_hint, full_state),
            sustained_cap_s=plat_s,
            sustained_cap_axis=plateau[2] if plateau else np.nan,
            sustained_outcome=timing_outcome(plat_s, plat_entry, row["extra_s"], row["extra_state"]),
            cap95_s=near,
            lead_ms=lead,
            peak_extra_nm=extra[inside].max(),
            peak_current_nm=old[inside].max(),
            peak_fused_nm=new[inside].max(),
        )
        row["outcome"] = timing_outcome(sat, at_entry, row["extra_s"], row["extra_state"])
        rows.append(row)
        panels.append((row, old, new))
    return rows, panels


def write_trace(name, tr, result, role):
    data = dict(
        slave_receive_s=tr.rx,
        observation_s=tr.t,
        partition=role,
        valid=result["fresh"],
        effort_evidence=result["ce"],
        motion_evidence=result["cq"],
        weight=result["weight"],
    )
    for key, matrix in (
        ("slave_torque_nm", tr.tau),
        ("free_torque_nm", result["nominal"]),
        ("filtered_residual_nm", result["effort"]),
        ("current_nm", tr.base),
        ("fused_nm", result["fused"]),
        ("change_nm", result["delta"]),
    ):
        for i, col in enumerate(matrix.T, 1):
            data[f"j{i}_{key}"] = col
    pd.DataFrame(data).to_csv(
        OUT / f"回放_{name}.csv", index=False, encoding="utf-8-sig", float_format="%.7g"
    )


def exposure(t, mask):
    within = mask[:-1] & mask[1:] & (np.diff(t) <= GAP)
    return np.diff(t)[within].sum()


def main():
    OUT.mkdir(exist_ok=True)
    tr, _, _, _ = load_trace(FREE, free=True)
    turns = reversal_events(tr)
    role = split_free(tr, turns)
    fit = role == "拟合"
    model, model_comparison = select_model(tr, fit)
    write_coefficients(model)
    model_comparison.to_csv(OUT / "空载拟合模型选择.csv", index=False, encoding="utf-8-sig")
    _, effort, motion, fresh = residuals(tr, model)
    calibration = calibrate(effort, motion, (role == "标定") & fresh)
    result = replay(tr, model, calibration)
    hold = (role == "留出") & fresh
    amplitude = np.linalg.norm(result["delta"], axis=1)
    alarms = confirmed(amplitude, tr, hold)
    for e in turns:
        mask = (tr.rx >= e["start"]) & (tr.rx <= e["end"]) & hold
        coverage = np.diff(
            np.r_[e["start"], tr.t[(tr.t >= e["start"]) & (tr.t <= e["end"]) & hold], e["end"]]
        ).max()
        e["complete"] = bool(e["complete"] and coverage <= GAP)
        e["peak_change_nm"] = amplitude[mask].max() if mask.any() and e["complete"] else np.nan
        e["false_alarm"] = (
            any(start <= e["end"] and end >= e["start"] and hit <= e["end"] for start, hit, end in alarms)
            if e["complete"]
            else None
        )
    pd.DataFrame(turns).to_csv(OUT / "换向事件.csv", index=False, encoding="utf-8-sig")
    errors = tr.tau[hold] - result["nominal"][hold]
    const = tr.tau[hold] - np.median(tr.tau[fit], axis=0)
    metrics = pd.DataFrame(
        dict(
            axis=np.arange(1, 7),
            mae_nm=abs(errors).mean(axis=0),
            rmse_nm=np.sqrt((errors**2).mean(axis=0)),
            p95_abs_nm=np.quantile(abs(errors), 0.95, axis=0),
            constant_rmse_nm=np.sqrt((const**2).mean(axis=0)),
        )
    )
    metrics.to_csv(OUT / "六轴留出误差.csv", index=False, encoding="utf-8-sig")
    write_trace("无接触", tr, result, role)
    summary = dict(
        free_file=FREE.name,
        fit_packets=int(fit.sum()),
        calibration_packets=int(((role == "标定") & fresh).sum()),
        holdout_packets=int(hold.sum()),
        holdout_s=exposure(tr.t, hold),
        alarms=alarms,
        held_rmse_change=np.sqrt(np.mean(amplitude[hold] ** 2)),
        held_p95_change=np.quantile(amplitude[hold], 0.95),
        held_peak_change=amplitude[hold].max(),
        fit_q_min=tr.q[fit].min(axis=0),
        fit_q_max=tr.q[fit].max(axis=0),
        sensitivity={str(v): len(confirmed(amplitude, tr, hold, v)) for v in (0.10, 0.20, 0.30)},
        free_gap_over_50ms=int((np.diff(tr.rx) > GAP).sum()),
        free_q_min=tr.q.min(axis=0),
        free_q_max=tr.q.max(axis=0),
        materials={},
    )
    items, press_rows = {}, []
    raw_materials = {}
    for name in WINDOWS:
        (path,) = MATERIALS.glob(f"{name}-*.csv")
        material, rx, torque, available = load_trace(path)
        res = replay(material, model, calibration)
        rows, panels = press_metrics(name, material, res, rx, torque)
        press_rows.extend(rows)
        entry = dict(observed=available, matched=len(material.t))
        mask = np.zeros(len(material.t), dtype=bool)
        for _, left, right in WINDOWS[name]:
            mask |= (material.t >= left) & (material.t <= right)
        outside = (material.q < summary["fit_q_min"]) | (material.q > summary["fit_q_max"])
        entry.update(
            outside_pose_by_axis=outside[mask].mean(axis=0), outside_pose_any=outside[mask].any(axis=1).mean()
        )
        outside_source = (material.q < tr.q.min(axis=0)) | (material.q > tr.q.max(axis=0))
        entry.update(
            outside_source_by_axis=outside_source[mask].mean(axis=0),
            press_min_q=material.q[mask].min(axis=0),
            press_max_q=material.q[mask].max(axis=0),
        )
        summary["materials"][name] = entry
        write_trace(name, material, res, np.where(mask, "加载末段", "其余轨迹"))
        items[name] = (material, res, panels)
        raw_materials[name] = (rx, torque)
    pd.DataFrame(press_rows).to_csv(OUT / "逐次按压.csv", index=False, encoding="utf-8-sig")
    sweep = []
    for lo, hi in ((0.90, 0.98), (0.95, 0.99), (0.975, 0.995)):
        cal = calibrate(effort, motion, (role == "标定") & fresh, (lo, hi))
        candidate = replay(tr, model, cal)
        alarm = confirmed(np.linalg.norm(candidate["delta"], axis=1), tr, hold)
        count_turns = sum(
            any(s <= e["end"] and end >= e["start"] and hit <= e["end"] for s, hit, end in alarm)
            for e in turns
            if e["complete"]
        )
        scores = []
        for name, (material, _, _) in items.items():
            rows, _ = press_metrics(name, material, replay(material, model, cal), *raw_materials[name])
            scores.extend(rows)
        table = pd.DataFrame(scores)
        sweep.append(
            dict(
                low_quantile=lo,
                high_quantile=hi,
                free_alarm_runs=len(alarm),
                turns_with_alarm=count_turns,
                local_early=(table.outcome == "限幅前").sum(),
                local_late=(table.outcome == "限幅后").sum(),
                local_close=(table.outcome == "时序接近").sum(),
                full_early=(table.full_outcome == "限幅前").sum(),
            )
        )
    pd.DataFrame(sweep).to_csv(OUT / "门槛敏感性.csv", index=False, encoding="utf-8-sig")
    with (OUT / "模型与统计.json").open("w", encoding="utf-8") as f:
        json.dump(
            dict(
                说明="力矩单位为 N·m，主机接收和可用时刻单位为 s。系数行按正弦、余弦、速度、加速度、平滑方向各六轴排列。",
                回放口径={
                    "数据": "11 月 24 日 11:48:47 明确无接触，存档副本与派生回放不重复使用。材料日志按预测观测的从端接收时间精确配对。空载文件首帧早于首条可用指令，之后重放项目的自由运动预测器。",
                    "分组": "按原空载激励块分出前 50% 拟合、随后 25% 标定、末 25% 核对。完整快速换向另整窗留出，拟合及标定与分界隔开 120 ms。已有自由运动预测器过去用过这份记录，留出评价针对新增的空载力矩模型和门槛。",
                    "跟随": "将六轴实测减预测的位置与速度残差逐量按标定区绝对值第 90 百分位归一化，取 12 项的均方根，再按标定区 95% 到 99% 分位线性升至 1。力矩分数对低通后六轴力矩残差作同样处理。",
                    "反馈": "w = 0.75 × 力矩分数 × (0.2 + 0.8 × 跟随分数)。同一 DH 坐标下，合成 (1-w) × 当前弹簧 - w × 0.2 × 有符号力矩残差 + 本地阻尼，然后按当前关节反馈上限限幅。现有材料反馈候选均未到主端反馈上限。",
                    "换向": "同轴相反方向均达到该轴速度上限的 45% 且各维持 30 ms，两段间不超过 0.60 s。向前加 0.10 s、向后加 0.12 s 作评价窗。缺包或观测缺口大于 50 ms 的窗口不计入分母。",
                    "提示": "无接触误提示以六轴新旧反馈差模长判断。按压时以反向于该次六轴预测净进给的反馈差投影判断。达到 0.20 N·m 并维持 30 ms，以首次满足持续时间的日志可用时刻计时。该判据尚未经过人体感知标定。",
                    "时序": "逐次按压表同时给出整次下行和末段加载两种口径。整程和末段分别用各自净进给方向作离线评分。平台取任一轴首次到达日志平台的从端原始回包。另列同轴同号持续 30 ms 的平台与 95% 幅值时间。时差在 ±20 ms 内归为时序接近。",
                    "字段": "逐次按压表 old、new、extra 分别对应当前阻力、融合阻力、两者之差。lead_ms 为末段首限幅减新增阻力确认。full_outcome 为整次下行首限幅对照。回放 CSV 中 valid 为时序评价可用标志，partition 记录数据分组。传入时间、接收时间和物理初触需要分别解释。",
                    "复现": "在项目根目录使用带 NumPy、pandas、Matplotlib 的 Python 执行 python analysis/joint_torque_replay.py。",
                },
                ridge=RIDGE,
                scale_speed=0.05,
                early=EARLY,
                max_weight=MAX_WEIGHT,
                gain=GAIN,
                level_nm=LEVEL,
                dwell_s=DWELL,
                filter_s=FILTER,
                gap_s=GAP,
                slave_limit=SLAVE_LIMIT,
                model=model,
                calibration=calibration,
                summary=summary,
            ),
            f,
            ensure_ascii=False,
            default=lambda v: v.tolist(),
            indent=2,
            allow_nan=False,
        )
    make_figures(OUT, tr, result, role, turns, alarms, metrics, items, raw_materials, SLAVE_LIMIT, LEVEL)
    print(metrics.round(3).to_string(index=False))
    print("自由空间留出", summary["holdout_s"], "秒，误提示", len(alarms), "段")
    print(
        pd.DataFrame(press_rows)[
            ["material", "press", "extra_state", "sat_axes", "outcome", "lead_ms"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
