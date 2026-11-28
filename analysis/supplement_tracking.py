"""在主机同一时间轴上对照实发指令与从端回包，量化手动折返的越界量。"""

from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

import joint_torque_replay as r
from joint_torque_figures import BLUE, GREEN, RED, line, plt, save, style


class Motion(NamedTuple):
    t: np.ndarray
    command: np.ndarray
    rx: np.ndarray
    q: np.ndarray
    v: np.ndarray
    tau: np.ndarray


def read_motion(path):
    """同一从端回包只取一次，角度展开后统一到指令所在的圈。"""
    tc, rc = "time_unix_s", "slave_received_unix_s"
    fs = [tc, rc, "master_udp_mode", "slave_teleop_state", *r.cols("master", "q_rad")]
    fs += sum([r.cols("slave", key) for key in ("q_rad", "dq_rad_s", "torque_nm")], [])
    d = pd.read_csv(path, usecols=fs, na_values=["NULL"], float_precision="round_trip")
    origin = d[tc].iloc[0]
    sent = d.loc[d.master_udp_mode.eq(1)]
    packets = d.loc[d.slave_teleop_state.eq(2)].drop_duplicates(rc)
    command = np.rad2deg(np.unwrap(r.six(sent, "master", "q_rad"), axis=0))
    q = np.rad2deg(np.unwrap(r.six(packets, "slave", "q_rad"), axis=0))
    q += 360 * np.round((command[0] - q[0]) / 360)
    return Motion(
        sent[tc].to_numpy() - origin,
        command,
        packets[rc].to_numpy() - origin,
        q,
        r.six(packets, "slave", "dq_rad_s"),
        r.six(packets, "slave", "torque_nm"),
    )


def largest_gap(t, left, right):
    stamps = t[(t >= left) & (t <= right)]
    return np.diff(np.r_[left, stamps, right]).max()


def command_extrema(y):
    """指令角度双向突出度至少 5°；同向候选之间保留实际更远的目标。"""
    extrema = []
    peaks = sorted((p, s) for s in (-1, 1) for p in find_peaks(s * y, prominence=5.0)[0])
    for p, s in peaks:
        if extrema and extrema[-1][1] == s:
            if s * y[p] > s * y[extrema[-1][0]]:
                extrema[-1] = (p, s)
        else:
            extrema.append((p, s))
    return extrema


def excursions(label, m):
    """同向越过本趟指令最远位置才算越界，取值时不平移主从信号。"""
    rows = []
    for j, y in enumerate(m.command.T):
        ev = command_extrema(y)
        for (a, _), (p, s), (b, _) in zip(ev, ev[1:], ev[2:]):
            z = s * y
            span = z[p] - z[a]
            if min(span, z[p] - z[b]) < 5.0:
                continue
            # 折点附近与目标相差不超过 0.75° 的连续指令区间，用来辨认是否有短暂停留。
            enter = a + np.flatnonzero(z[a:p] < z[p] - 0.75)[-1] + 1
            leave = p + np.flatnonzero(z[p : b + 1] < z[p] - 0.75)[0] - 1
            t0 = m.t[enter]
            end = min(m.t[b], m.t[leave] + 0.60)
            # 进入目标带前，从端应当确实从本次进给一侧接近，以免把上一趟遗留的越界算进来。
            pre = max(m.t[a], t0 - 0.30)
            before = (m.rx >= pre) & (m.rx <= t0)
            if not before.any() or np.min(s * m.q[before, j]) > z[p] - 0.5:
                continue
            if max(largest_gap(m.t, pre, end), largest_gap(m.rx, pre, end)) > r.GAP:
                continue
            ix = np.flatnonzero((m.rx >= t0) & (m.rx <= end))
            if not len(ix):
                continue
            k = ix[np.argmax(s * m.q[ix, j])]
            hold = m.t[leave] - t0
            inside = ix[m.rx[ix] <= m.t[leave]]
            held_peak = np.max(s * m.q[inside, j] - z[p]) if hold >= 0.30 and len(inside) else np.nan
            rows.append(
                dict(
                    记录=label,
                    轴=j + 1,
                    方向=s,
                    目标角_deg=y[p],
                    进给行程_deg=span,
                    指令折点_s=m.t[p],
                    进带_s=t0,
                    离带_s=m.t[leave],
                    取峰截止_s=end,
                    目标带停留_s=hold,
                    从端峰值_s=m.rx[k],
                    指令折点时跟随欠程_deg=z[p] - s * np.interp(m.t[p], m.rx, m.q[:, j]),
                    有符号越界_deg=s * m.q[k, j] - z[p],
                    持位内有符号越界_deg=held_peak,
                )
            )
    return rows


def tracking_figure(out, data, turns):
    """给出持位越界、快速折返越界与明显欠程各一例，三个面板均可从事件表定位。"""
    held = turns[turns.记录.str.startswith("常规")].dropna(subset=["持位内有符号越界_deg"])
    fast = turns[turns.记录.str.startswith("往返")]
    examples = [
        (held.loc[held.持位内有符号越界_deg.idxmax()], "目标短暂停留", True),
        (fast.loc[fast.有符号越界_deg.idxmax()], "快速折返出现越界", False),
        (fast.loc[fast.有符号越界_deg.idxmin()], "快速折返出现欠程", False),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.1), constrained_layout=True)
    fig.suptitle("从端是否越过了本趟指令的最远角度", fontsize=16, weight="bold")
    for ax, (e, title, held_only) in zip(axes, examples):
        m = data[e.记录]
        j = int(e.轴) - 1
        left = e.进带_s - 0.55
        right = (e.离带_s if held_only else e.取峰截止_s) + 0.16
        # 长时间持位时，放大越界真正出现的时刻。
        if held_only:
            k = np.flatnonzero((m.rx >= e.进带_s) & (m.rx <= e.离带_s))
            peak = k[np.argmax(e.方向 * m.q[k, j])]
        else:
            peak = np.searchsorted(m.rx, e.从端峰值_s)
        left = max(left, m.rx[peak] - 1.10)
        right = min(right, m.rx[peak] + 0.75)
        for t, y, color, label in (
            (m.t, m.command[:, j], BLUE, "主端实发角"),
            (m.rx, m.q[:, j], GREEN, "从端实测角"),
        ):
            use = (t >= left) & (t <= right)
            line(ax, t[use], y[use], color, label=label)
        ax.axhline(e.目标角_deg, color="#7d858b", ls="--", lw=1, label="本趟最远指令")
        ax.scatter(m.rx[peak], m.q[peak, j], s=34, zorder=4, color=RED)
        value = e.持位内有符号越界_deg if held_only else e.有符号越界_deg
        text = f"越过 {value:.1f}°" if value > 0 else f"还差 {-value:.1f}°"
        ax.set(title=f"{e.记录} · J{j + 1} · {title}\n{text}", xlabel="记录时间 (s)", ylabel="关节角 (°)")
        style(ax)
    axes[0].legend(fontsize=8, frameon=False, loc="best")
    save(fig, out / "指令折返与实测位置.png")
