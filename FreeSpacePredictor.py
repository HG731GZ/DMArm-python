"""按实际主端指令预测从端自由空间运动，输出与回包接收时刻配对的状态。"""

from bisect import bisect_right
from dataclasses import dataclass
from math import isfinite, pi, sqrt

import numpy as np

@dataclass
class FreeSpaceParameters:
    speed: tuple
    accel_positive: tuple
    accel_negative: tuple
    braking: tuple
    delay: tuple
    step_s: float

    def __post_init__(self):
        for name in ("speed", "accel_positive", "accel_negative", "braking", "delay"):
            values = np.asarray(getattr(self, name), dtype=float)
            if values.shape != (6,) or not np.isfinite(values).all():
                raise ValueError(f"自由空间预测 {name} 须按 J1～J6 填写六个有限值")
            if (values < 0).any() or (name != "delay" and (values == 0).any()):
                raise ValueError("预测速度、加速和制动上限须为正数，滞后须为非负数")
            setattr(self, name, values)
        if not isfinite(self.step_s) or self.step_s <= 0:
            raise ValueError("预测积分步长须为正的有限秒数")


class FreeSpacePredictor:
    def __init__(self, parameters):
        self.parameters = parameters
        p = parameters
        # 六轴小向量使用 Python 标量推进，避免长回包间隔后集中分配大量 NumPy 临时数组。
        self.limits = tuple(zip(p.speed.tolist(), p.accel_positive.tolist(), p.accel_negative.tolist(),
                               p.braking.tolist(), (p.delay / p.step_s).tolist()))
        self.reset()

    def reset(self):
        """由连接、发布或跟随会话结束触发；力反馈按钮不重置模型。"""
        self.origin = None
        self.tick = 0
        self.q = self.dq = self.previous_q = self.previous_dq = None

    def initialize(self, received_at, q, dq):
        """每个会话仅用一次实测状态确定初值，后续完全由指令推进。"""
        self.origin = received_at
        self.tick = 0
        self.q = list(map(float, q))
        self.dq = list(map(float, dq))
        self.previous_q = self.q.copy()
        self.previous_dq = self.dq.copy()

    def advance(self, received_at, commands):
        """命令按单调发送时刻升序排列，覆盖上一回包至本次回包所需的延迟历史。"""
        step = self.parameters.step_s
        times = [command.sent_at for command in commands]
        elapsed = received_at - self.origin
        while self.tick * step < elapsed:
            # 与离线模型一致：先在固定积分网格上零阶保持实际指令，再取各轴滞后值。
            self.previous_q, self.previous_dq = self.q, self.dq
            q, dq = [], []
            for axis, (speed, accel_positive, accel_negative, braking, delay_steps) in enumerate(self.limits):
                past = max(0., self.tick - delay_steps)
                left = int(past)
                fraction = past - left
                lo = max(0, bisect_right(times, self.origin + left * step) - 1)
                hi = max(0, bisect_right(times, self.origin + min(left + 1, self.tick) * step) - 1)
                q0, q1 = float(commands[lo].q[axis]), float(commands[hi].q[axis])
                target = q0 + fraction * (pi - (pi - (q1 - q0)) % (2 * pi))
                error = pi - (pi - (target - self.q[axis])) % (2 * pi)
                goal = (1. if error >= 0 else -1.) * min(speed, sqrt(2 * braking * abs(error)))
                velocity = self.dq[axis]
                change = goal - velocity
                speeding = velocity * change >= 0
                rate = (accel_positive if goal >= 0 else accel_negative) if speeding else braking
                next_dq = velocity + max(-rate * step, min(rate * step, change))
                # 反向先制动到零，下一积分步再按新方向起步。
                if velocity * next_dq < 0 and not speeding:
                    next_dq = 0.
                q.append(self.q[axis] + .5 * (velocity + next_dq) * step)
                dq.append(next_dq)
            self.q, self.dq = q, dq
            self.tick += 1
        if self.tick == 0:
            return np.array(self.q), np.array(self.dq)
        fraction = (elapsed - (self.tick - 1) * step) / step
        return (np.array([a + fraction * (b - a) for a, b in zip(self.previous_q, self.q)]),
                np.array([a + fraction * (b - a) for a, b in zip(self.previous_dq, self.dq)]))
