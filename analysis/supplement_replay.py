"""使用补录的六份自由空间轨迹，整条留出验证并回放已标注的三份按压记录。"""

import json

import numpy as np
import pandas as pd

import joint_torque_replay as r
from joint_torque_figures import BLUE, GREEN, RED, line, plt, save, style
from supplement_tracking import excursions, largest_gap, read_motion, tracking_figure


DATA = r.LOGS / "从端性能测试-更多工况"
OUT = r.LOGS / "补录六轴分析_20261128"
SETS = {
    "常规": (
        "常规快速运动-等待从端到位-20261128_163535.csv",
        "常规快速运动-等待从端到位-20261128_163852_713886.csv",
    ),
    "慢速": ("慢速小幅度运动-20261128_164942_887235.csv", "慢速小幅度运动-20261128_165051_417407.csv"),
    "往返": ("快速往返-20261128_165137_957340.csv", "快速往返-20261128_165246_357364.csv"),
}
OLD, NEW = "原空载标定", "补录后标定"
STRICT = "补录后仅双项同时越阈"


def table(name, rows):
    df = pd.DataFrame(rows)
    df.to_csv(OUT / (name + ".csv"), index=False, encoding="utf-8-sig", float_format="%.7g")
    return df


def train(dev):
    """四份开发文件轮流整份留出，先选模型，再用所选模型的折外残差标定门槛。"""
    xs = {k: r.features(tr) for k, tr in dev.items()}

    def fit(keys, family):
        x = np.vstack([xs[k] for k in keys])
        tau = np.vstack([dev[k].tau for k in keys])
        return r.torque_model(x, tau, np.ones(len(x), bool), family)

    comparison, folds = [], {}
    for family in r.FAMILIES:
        normalized = []
        for k, tr in dev.items():
            model = fit([i for i in dev if i != k], family)
            folds[family, k] = model
            error = r.nominal_torque(xs[k], model) - tr.tau
            rmse = np.sqrt(np.mean(error**2, axis=0))
            normalized.append(np.mean(rmse / r.SLAVE_LIMIT))
        comparison.append(dict(模型=family, 逐文件等权归一化均方根误差=np.mean(normalized)))
    selected = min(comparison, key=lambda row: row["逐文件等权归一化均方根误差"])["模型"]
    effort, motion = [], []
    for k, tr in dev.items():
        _, e, m, valid = r.residuals(tr, folds[selected, k])
        effort.append(e[valid])
        motion.append(m[valid])
    e = np.vstack(effort)
    cal = r.calibrate(e, np.vstack(motion), np.ones(len(e), bool))
    return (fit(list(dev), selected), cal), comparison


def free_metrics(label, config, tr, motion, result):
    """同时核算主端旧候选限幅导致的幅值不确定区间。"""
    valid = result["fresh"]
    capped = abs(tr.base) >= r.MASTER_LIMIT - 1.0e-9
    # 旧反馈已达上限时，真实弹簧可能更大。融合输出与弹簧同方向单调，
    # 该轴新增幅值介于零和用最小弹簧还原的幅值之间。其他轴可精确还原。
    upper = np.linalg.norm(result["delta"], axis=1)
    lower = np.linalg.norm(np.where(capped, 0.0, result["delta"]), axis=1)
    high, low = (r.confirmed(signal, tr, valid) for signal in (upper, lower))
    whi, wlo = [], []
    for event in r.reversal_events(motion):
        a, b = event["start"], event["end"]
        if not event["complete"] or a < tr.t[0] or b > tr.t[-1] or largest_gap(tr.t[valid], a, b) > r.GAP:
            continue
        whi.append(any(hit <= b and end >= a for _, hit, end in high))
        wlo.append(any(hit <= b and end >= a for _, hit, end in low))
    summary = dict(
        记录=label,
        标定=config,
        有效秒=r.exposure(tr.t, valid),
        误提示段=len(high),
        置零核对段数=len(low),
        完整逐轴换向窗=len(whi),
        有误提示窗=sum(whi),
        置零核对有误提示窗=sum(wlo),
        差值模长P95_Nm=np.quantile(upper[valid], 0.95),
        主端原候选限幅帧=capped[valid].any(axis=1).sum(),
        受限帧是否改变逐帧达到判据=bool(np.any(((lower >= r.LEVEL) != (upper >= r.LEVEL)) & valid)),
    )
    rmse = np.sqrt(np.mean((result["nominal"][valid] - tr.tau[valid]) ** 2, axis=0))
    axes = [dict(记录=label, 标定=config, 轴=j, 均方根误差_Nm=x) for j, x in enumerate(rmse, 1)]
    events = []
    for a, b, c in high:
        ix = np.flatnonzero((tr.t >= a) & (tr.t <= c))
        k = ix[np.argmax(upper[ix])]
        events.append(
            dict(
                记录=label,
                标定=config,
                越阈_s=a,
                确认_s=b,
                结束_s=c,
                峰值时刻_s=tr.t[k],
                峰值模长上界_Nm=upper[k],
                峰值贡献最大轴=1 + np.argmax(abs(result["delta"][k])),
                力矩分数=result["ce"][k],
                跟随分数=result["cq"][k],
                融合权重=result["weight"][k],
            )
        )
    return summary, axes, events


def free_figure(traces, curves):
    fig, axes = plt.subplots(3, 1, figsize=(10.4, 6.7), constrained_layout=True)
    fig.suptitle("独立空载记录 · 新增反馈的六轴模长", fontsize=16, weight="bold")
    colors = {OLD: "#88979e", NEW: BLUE, STRICT: GREEN}
    for ax, label in zip(axes, ["常规B", "慢速B", "往返B"]):
        tr = traces[label]
        for config in (OLD, NEW, STRICT):
            t, y = curves[label, config]
            line(ax, t, y, colors[config], config, width=1.2)
        ax.axhline(r.LEVEL, color=RED, ls="--", lw=0.95, label="0.20 N·m 判据")
        ax.set(xlim=(0, tr.t[-1]), ylabel="新增模长 (N·m)", title=label)
        style(ax)
    axes[0].legend(ncol=4, fontsize=8, frameon=False)
    axes[-1].set_xlabel("各文件相对时间 (s)")
    save(fig, OUT / "独立空载的反馈变化.png")


def press_figure(panels, strict_panels):
    fig, axes = plt.subplots(5, 3, figsize=(12, 12.7))
    fig.subplots_adjust(top=0.91, bottom=0.045, left=0.07, right=0.985, wspace=0.24, hspace=0.47)
    fig.suptitle("三种材料的 13 次按压 · 当前反馈与两种力矩融合方式", fontsize=16, y=0.987, weight="bold")
    fig.text(
        0.07,
        0.95,
        "蓝线为当前反馈，绿线保留力矩的小幅独立贡献，紫色虚线要求跟随与力矩同时越阈。橙线为末段首次限幅，绿点为绿线的新增阻力确认。",
        fontsize=9,
    )
    for col, (name, (tr, items)) in enumerate(panels.items()):
        for ax, (row, base, new), (_, _, strict) in zip(axes[:, col], items, strict_panels[name]):
            a, b = row["start_s"], row["end_s"]
            take = (tr.t >= a) & (tr.t <= b)
            line(ax, tr.t[take] - a, base[take], BLUE, width=1.25)
            line(ax, tr.t[take] - a, new[take], GREEN, width=1.45)
            line(ax, tr.t[take] - a, strict[take], "#865893", width=1.0, ls="--", alpha=0.9)
            if np.isfinite(row["sat_s"]):
                ax.axvline(row["sat_s"] - a, color=RED, ls="--", lw=1.15)
            if np.isfinite(row["extra_s"]):
                k = np.searchsorted(tr.t, row["extra_s"])
                ax.scatter(tr.t[k] - a, new[k], s=18, color=GREEN, zorder=4)
            ax.set_title(f"{name} {row['press']} · {row['outcome']}", fontsize=10)
            ax.set_xlim(0, b - a)
            if col == 0:
                ax.set_ylabel("阻力投影 (N·m)", fontsize=8)
            style(ax)
        for ax in axes[len(items) :, col]:
            ax.axis("off")
        axes[len(items) - 1, col].set_xlabel("末段时间 (s)")
    save(fig, OUT / "三种材料的反馈曲线.png")


def main():
    OUT.mkdir(exist_ok=True)
    plt.rcParams.update(
        {"font.family": ["Microsoft YaHei", "DejaVu Sans"], "font.size": 9, "axes.unicode_minus": False}
    )
    old = json.loads((r.OUT / "模型与统计.json").read_text("utf-8"))
    traces, raw, info, turns = {}, {}, [], []
    dev = {"原辨识": r.load_trace(r.FREE, free=True)[0]}
    for kind, filenames in SETS.items():
        for letter, filename in zip("AB", filenames):
            path = DATA / filename
            label = kind + letter
            tr, rx, _, n = r.load_trace(path, feedback_on=False)
            traces[label] = tr
            m = raw[label] = read_motion(path)
            if letter == "A":
                dev[label] = tr
            # 按相邻原始回包间隔累计平台时长，超过 50 ms 的缺口不计入分母。
            dt = np.diff(m.rx)
            weights = np.where(dt <= r.GAP, dt, 0.0)
            capped = (abs(m.tau[:-1]) >= r.SLAVE_LIMIT - 1e-6).any(axis=1)
            info.append(
                dict(
                    记录=label,
                    文件=path.name,
                    用途="开发" if letter == "A" else "独立验证",
                    时长_s=m.t[-1] - m.t[0],
                    原始有效回包=len(rx),
                    记录到的预测包=n,
                    精确配对包=len(tr.t),
                    最大回包间隔_ms=1000 * np.diff(rx).max(),
                    超50ms回包缺口=(np.diff(rx) > r.GAP).sum(),
                    任一从轴限幅时间百分比=100 * weights[capped].sum() / weights.sum(),
                )
            )
            if kind != "慢速":
                turns.extend(excursions(label, m))
    table("文件与用途", info)
    turns = table("逐轴越界事件", turns)
    tracking_figure(OUT, raw, turns)
    new, selection = train(dev)
    table("按整份文件选择空载模型", selection)
    configurations = {
        OLD: (old["model"], old["calibration"], r.EARLY),
        NEW: (*new, r.EARLY),
        STRICT: (*new, 0.0),
    }
    summary, errors, alarms, presses = [], [], [], []
    panels, strict_panels, curves = {}, {}, {}
    for name, (model, cal, early) in configurations.items():
        all_err = []
        for label, tr in traces.items():
            if not label.endswith("B"):
                continue
            replay = r.replay(tr, model, cal, early=early)
            row, joints, events = free_metrics(label, name, tr, raw[label], replay)
            summary.append(row)
            alarms.extend(events)
            valid = replay["fresh"]
            curves[label, name] = (tr.t[valid], np.linalg.norm(replay["delta"][valid], axis=1))
            if name != STRICT:
                errors.extend(joints)
                all_err.append((tr.tau - replay["nominal"])[valid])
        if all_err:
            rmse = np.sqrt(np.mean(np.vstack(all_err) ** 2, axis=0))
            errors.extend(
                dict(记录="全部留出包", 标定=name, 轴=j, 均方根误差_Nm=x) for j, x in enumerate(rmse, 1)
            )
        for material in r.WINDOWS:
            (p,) = r.MATERIALS.glob(material + "-*.csv")
            tr, rx, tau, _ = r.load_trace(p)
            result = r.replay(tr, model, cal, early=early)
            rows, samples = r.press_metrics(material, tr, result, rx, tau)
            presses.extend(dict(标定=name, **row) for row in rows)
            if name == NEW:
                panels[material] = tr, samples
            elif name == STRICT:
                strict_panels[material] = samples
    free = table("空载整文件验证", summary)
    table("六轴力矩拟合误差", errors)
    table("空载误提示时刻", alarms)
    pressure = table("逐次按压时序", presses)
    press_figure(panels, strict_panels)
    free_figure(traces, curves)
    with (OUT / "补录模型和回放口径.json").open("w", encoding="utf-8") as f:
        json.dump(
            dict(
                模型=new[0],
                门槛=new[1],
                开发记录=[r.FREE.name] + [i["文件"] for i in info if i["用途"] == "开发"],
                留出记录=[i["文件"] for i in info if i["用途"] == "独立验证"],
                力反馈="沿用原回放的六轴带符号力矩融合、0.20 N·m 与 30 ms 判据、20 ms 时序分辨和固定材料窗。记录时间取主机收到数据的时间。",
                标定="在四份开发文件内逐份轮换验证并等权选择模型，合并所选模型的折外自由空间残差计算第95/99百分位。之后使用完整开发集拟合系数。",
                双项消融="探索性对照仅将力矩分数独立贡献比例从0.20改成0，其余参数、评分条件和模型全部固定。所有策略以现用预测弹簧加本地阻尼为反馈参照。",
                超调="逐轴选择双向突出度至少5°的相邻指令折返。目标带为本趟最远指令的0.75°以内，进入前0.30 s内从端须经过进给侧。比较进入目标带后直到下一折点或离带后0.60 s中较早者的从端最远角；持位另取连续在带0.30 s以上且尚未离带的从端峰值。越界为负表示仍差若干度。近折点的指令和从端回包都要求相邻采样不超过50 ms。两端不完整的折返不入样。",
                范围="空载误提示采用本文件完整有效时段；换向分母按原始速度逐轴取窗、排除回包和可用观测缺口，多个关节的窗可以重叠。",
                限幅还原="仅常规B的少数原主端反馈候选触及主端上限。对这类轴分别按最小相容弹簧和新增幅值置零核对；两者包围可实现的新增幅值区间。",
                复现="在项目根目录运行 python analysis/supplement_replay.py，使用 NumPy、pandas、SciPy 和 Matplotlib，原空载模型取自上一轮的模型与统计.json。",
            ),
            f,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
            default=lambda v: v.tolist(),
        )
    print(free.round(3).to_string(index=False))
    print(pressure.groupby(["标定", "outcome"]).size().to_string())
    print(turns.groupby(["记录", "轴"]).有符号越界_deg.agg(["size", "max"]).round(2).to_string())


if __name__ == "__main__":
    main()
