"""绘制关节力矩离线回放中的空载拟合、换向和逐次按压曲线。"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

BLUE, GREEN, GRAY, RED, GOLD = "#245c86", "#007969", "#536b79", "#ad452d", "#9a6a10"


def line(ax, t, y, color, label=None, width=1.5, **kwargs):
    """相邻观测间隔超过 50 ms 时在曲线上留白。"""
    gaps = np.flatnonzero(np.diff(t) > 0.050) + 1
    ax.plot(
        np.insert(t, gaps, np.nan),
        np.insert(y.astype(float), gaps, np.nan),
        color=color,
        lw=width,
        label=label,
        **kwargs,
    )


def style(ax):
    ax.grid(axis="y", color="#dce3e5", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["bottom", "left"]].set_color("#adbdc3")
    ax.set_axisbelow(True)
    ax.tick_params(length=3)


def save(fig, path):
    fig.savefig(path, dpi=155, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def grouped_legend(fig, keys, y):
    handles = [Line2D([], [], color=color, lw=2, ls=ls, label=label) for color, ls, label in keys]
    fig.legend(
        handles=handles, loc="upper left", bbox_to_anchor=(0.07, y), ncol=len(keys), frameon=False, fontsize=9
    )


def make_figures(out, tr, result, role, turns, alarms, metrics, items, raw_materials, limits, level):
    plt.rcParams.update(
        {
            "font.family": ["Microsoft YaHei", "DejaVu Sans"],
            "font.size": 10,
            "text.color": "#253d48",
            "axes.labelcolor": "#364e59",
            "xtick.color": "#49606d",
            "ytick.color": "#49606d",
        }
    )
    fig, axes = plt.subplots(3, 2, figsize=(12, 7.6))
    fig.subplots_adjust(left=0.075, right=0.98, top=0.83, bottom=0.08, wspace=0.16, hspace=0.46)
    fig.text(0.07, 0.965, "六轴自由空间力矩拟合", fontsize=20, weight="bold")
    fig.text(0.07, 0.922, "各图放大该轴主激励时段。浅蓝底为拟合与门槛标定均未使用的留出区。", fontsize=11)
    grouped_legend(fig, [(GRAY, "-", "从端回传"), (GREEN, "-", "空载模型")], 0.909)
    for i, (ax, (left, right)) in enumerate(
        zip(axes.flat, [(3.7, 12.5), (17.5, 28.6), (29, 38.4), (38.4, 45.1), (45, 52.5), (52.5, 60.1)])
    ):
        use = (tr.rx >= left) & (tr.rx <= right)
        t = tr.rx[use]
        ax.fill_between(
            t,
            0,
            1,
            where=role[use] == "留出",
            step="mid",
            transform=ax.get_xaxis_transform(),
            color="#dfedf3",
            linewidth=0,
        )
        line(ax, t, tr.tau[use, i], GRAY, width=1.25)
        line(ax, t, result["nominal"][use, i], GREEN, width=1.35)
        ax.set(xlim=(left, right), title=f"J{i + 1}    留出 RMSE {metrics.rmse_nm[i]:.2f} N·m")
        ax.set_ylabel("力矩（N·m）")
        style(ax)
    for ax in axes[-1]:
        ax.set_xlabel("记录时间（s）")
    save(fig, out / "六轴自由空间拟合.png")

    amplitude = np.linalg.norm(result["delta"], axis=1)
    hold = (role == "留出") & result["fresh"]
    candidates = [e for e in turns if e["complete"]]
    fig, (ax, zoom) = plt.subplots(2, 1, figsize=(11, 6.2), gridspec_kw={"height_ratios": [1, 1.2]})
    fig.subplots_adjust(left=0.09, right=0.97, top=0.82, bottom=0.10, hspace=0.55)
    fig.text(0.07, 0.958, "无接触轨迹中的新增反馈", fontsize=20, weight="bold")
    fig.text(
        0.07,
        0.90,
        f"只评价留出段。六轴新旧反馈差的模长达到 {level:.2f} N·m 并持续 30 ms 记为一次误提示。",
        fontsize=11,
    )
    line(ax, tr.t, np.where(hold, amplitude, np.nan), BLUE)
    for e in candidates:
        ax.axvspan(e["start"], e["end"], color="#e4d3ad", alpha=0.5, lw=0)
    for _, hit, _ in alarms:
        ax.plot(hit, level, "v", color=RED, ms=6)
    ax.axhline(level, ls="--", color=RED, lw=1)
    ax.set(xlabel="记录时间（s）", ylabel="反馈差模长（N·m）")
    style(ax)
    if candidates:
        e = max(candidates, key=lambda x: x["peak_change_nm"])
        a, b = e["start"], e["end"]
        use = (tr.t >= a - 0.15) & (tr.t <= b + 0.15)
        zoom.axvspan(a, b, color="#eee1c4", alpha=0.65, lw=0)
        line(zoom, tr.t[use], np.where(hold[use], amplitude[use], np.nan), BLUE, "新增反馈")
        zoom.axhline(level, ls="--", color=RED, lw=1)
        speed = zoom.twinx()
        line(speed, tr.rx[use], tr.v[use, e["axis"] - 1], GREEN, "从端速度")
        speed.spines["top"].set_visible(False)
        speed.set_ylabel(f"J{e['axis']} 速度（rad/s）", color=GREEN)
        zoom.set_title(f"完整换向中反馈差峰值最大的一段  ·  J{e['axis']}", loc="left")
    zoom.set(xlabel="记录时间（s）", ylabel="反馈差模长（N·m）")
    style(zoom)
    ax.legend(
        handles=[
            Patch(color="#eee1c4", label="可评价换向窗"),
            Line2D([], [], color=BLUE, label="反馈差模长"),
            Line2D([], [], color=RED, ls="--", label="幅值判据"),
            Line2D([], [], marker="v", lw=0, color=RED, label="误提示确认"),
        ],
        frameon=False,
        loc="upper right",
        ncol=4,
        fontsize=8,
    )
    save(fig, out / "快速换向与误提示.png")

    fig6, grid = plt.subplots(6, 3, figsize=(15, 11), sharex="col", sharey="row")
    fig6.subplots_adjust(left=0.072, right=0.985, bottom=0.065, top=0.90, wspace=0.12, hspace=0.20)
    fig6.text(0.07, 0.972, "三种材料全程的六轴主端反馈", fontsize=20, weight="bold")
    grouped_legend(fig6, [(GRAY, "-", "当前记录的反馈"), (GREEN, "-", "离线融合候选")], 0.955)
    for c, (name, (mat, res, panels)) in enumerate(items.items()):
        for j, axj in enumerate(grid[:, c]):
            line(axj, mat.t, mat.base[:, j], GRAY, width=1)
            line(axj, mat.t, np.where(res["fresh"], res["fused"][:, j], np.nan), GREEN, width=1)
            for row, _, _ in panels:
                axj.axvspan(row["start_s"], row["end_s"], color="#e7ded0", alpha=0.42, lw=0)
            if j == 0:
                axj.set_title(name)
            if c == 0:
                axj.set_ylabel(f"J{j + 1}（N·m）")
            if j == 5:
                axj.set_xlabel("记录时间（s）")
            style(axj)
        fig, paxes = plt.subplots(
            (len(panels) + 1) // 2, 2, figsize=(12, 1.99 * ((len(panels) + 1) // 2) + 1.65), squeeze=False
        )
        fig.subplots_adjust(
            left=0.08,
            right=0.98,
            bottom=0.12,
            top=0.75 if len(panels) == 4 else 0.80,
            hspace=0.68,
            wspace=0.18,
        )
        fig.text(0.07, 0.95, f"{name}逐次按压的关节反馈", fontsize=19, weight="bold")
        fig.text(
            0.07,
            0.887 if len(panels) == 4 else 0.903,
            "沿每次六轴加载方向投影。正值表示阻碍进给，横轴零点为进入末段加载窗。",
            fontsize=10,
        )
        grouped_legend(
            fig,
            [
                (GRAY, "-", "当前反馈"),
                (GREEN, "-", "融合候选"),
                (BLUE, ":", "新增阻力"),
                (RED, "--", "任一轴首达平台"),
                (GOLD, "-.", "新增阻力确认"),
            ],
            0.877 if len(panels) == 4 else 0.894,
        )
        for axp, (row, old, new) in zip(paxes.flat, panels):
            left, right = row["start_s"], row["end_s"]
            use = (mat.t >= left - 0.35) & (mat.t <= right)
            t = mat.t[use] - left
            axp.axvspan(-0.35, 0, color="#edf0f1", lw=0)
            line(axp, t, old[use], GRAY)
            line(axp, t, np.where(res["fresh"][use], new[use], np.nan), GREEN)
            line(axp, t, np.where(res["fresh"][use], new[use] - old[use], np.nan), BLUE, ls=":", width=1.3)
            axp.axhline(level, color="#93a9b5", lw=0.7, ls=":")
            for key, color, ls in (("sat_s", RED, "--"), ("extra_s", GOLD, "-.")):
                if np.isfinite(row[key]):
                    axp.axvline(row[key] - left, color=color, ls=ls, lw=1.4)
            axp.set(xlim=(-0.35, right - left), ylabel="力矩（N·m）", xlabel="距加载窗起点（s）")
            axp.set_title(
                f"{row['press']}    {left:.2f}–{right:.2f} s    {row['outcome']}", loc="left", fontsize=10
            )
            style(axp)
        for blank in list(paxes.flat)[len(panels) :]:
            blank.set_visible(False)
        save(fig, out / f"按压回放_{name}.png")
    save(fig6, out / "全程六轴反馈对照.png")
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 6.4), sharex="col", sharey="row")
    fig.subplots_adjust(left=0.095, right=0.98, top=0.77, bottom=0.12, hspace=0.13, wspace=0.15)
    fig.text(0.07, 0.95, "从端首次触及限幅时，主端增加了多少阻力", fontsize=19, weight="bold")
    grouped_legend(
        fig,
        [
            (GRAY, "-", "当前反馈"),
            (GREEN, "-", "融合候选"),
            (BLUE, ":", "新增阻力"),
            (GOLD, "-.", "新增阻力确认"),
            (RED, "--", "从端首达平台"),
        ],
        0.916,
    )
    for c, name in enumerate(("铁块", "硬海绵")):
        mat, res, panels = items[name]
        row, old, new = panels[0]
        left, right = row["start_s"], row["end_s"]
        rx, tau = raw_materials[name]
        use = (rx >= left - 0.1) & (rx <= min(right, row["sat_s"] + 0.5))
        line(axes[0, c], rx[use] - left, 100 * np.max(abs(tau[use]) / limits, axis=1), BLUE)
        axes[0, c].axhline(100, color="#8ca3b0", ls=":", lw=1)
        use = (mat.t >= left - 0.1) & (mat.t <= min(right, row["sat_s"] + 0.5))
        t = mat.t[use] - left
        line(axes[1, c], t, old[use], GRAY)
        line(axes[1, c], t, np.where(res["fresh"][use], new[use], np.nan), GREEN)
        line(axes[1, c], t, np.where(res["fresh"][use], (new - old)[use], np.nan), BLUE, ls=":")
        axes[1, c].axhline(level, color="#8ca3b0", ls=":", lw=1)
        for ax in axes[:, c]:
            ax.axvline(row["extra_s"] - left, color=GOLD, ls="-.", lw=1.5)
            ax.axvline(row["sat_s"] - left, color=RED, ls="--", lw=1.5)
            style(ax)
        timing = "提前" if row["lead_ms"] > 0 else "晚于平台"
        axes[0, c].set_title(
            f"{name}第一次按压    新增阻力{timing} {abs(row['lead_ms']):.0f} ms", loc="left", fontsize=11
        )
        axes[1, c].set_xlabel("距加载末段起点（s）")
    axes[0, 0].set_ylabel("六轴中最高的力矩\n限幅占比（%）")
    axes[1, 0].set_ylabel("沿加载方向的阻力矩（N·m）")
    save(fig, out / "限幅与新增阻力时序.png")
