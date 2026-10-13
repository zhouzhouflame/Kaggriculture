"""任务链与出售决策的短场景强化学习实验。

启发式生成可行动作集合，Q-learning 学习集合内的选择。
奖励按模拟环境中实际兑现的收入记录，每笔收入计入一次。
"""
from __future__ import annotations

import itertools
import math
import random
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass


def summary(values):
    values = sorted(values)
    return {"n": len(values), "mean": statistics.mean(values),
            "median": statistics.median(values), "p10": values[int(.1 * (len(values) - 1))],
            "minimum": values[0], "maximum": values[-1]}


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


@dataclass(frozen=True)
class Chain:
    name: str
    xy: tuple[int, int]
    operations: int
    deadline: int
    value: int = 0
    required: bool = False
    wheat: int = 0


class MobileEnv:
    """单个流动工人、五条候选任务链、17-turn 决策时域。

    返仓选项合并运输、卸货与成功出售，将携货价值计入已兑现收入。
    官方环境接口按工人动作和市场订单分步执行这条任务链。
    """
    horizon = 17
    shed = (0, 0)
    RETURN, END = 5, 6

    def __init__(self, seed):
        rng = random.Random(seed)
        self.context = (rng.randrange(3), rng.randrange(3), rng.randrange(2))
        a, b, weed = self.context
        self.chains = [
            Chain("nearby FEED + CARE", (1, 1), 2, 7, required=True, wheat=1),
            Chain("deadline WATER", (3, 1), 1, 11, required=True),
            Chain("assist HARVEST near shed", (1, 2), 1, 17, 35 + 35*a),
            Chain("HARVEST farther batch", (4, 2), 1, 17, 65 + 50*b),
            Chain("DIG if needed + HARVEST", (2, 3), 1 + weed, 17, 75),
        ]
        self.t, self.pos, self.done, self.bag, self.cash, self.wheat = 0, self.shed, 0, 0, 0, 1
        self.ended = False
        self.trace = []

    def state(self):
        return (*self.context, self.t, *self.pos, self.done, self.bag, self.wheat)

    def _can_finish_required(self, pos, t, done, bag, wheat):
        pending = [i for i, c in enumerate(self.chains) if c.required and not done & (1 << i)]
        for order in itertools.permutations(pending):
            at, clock, feed, good = pos, t, wheat, True
            for i in order:
                c = self.chains[i]
                clock += manhattan(at, c.xy) + c.operations
                feed -= c.wheat
                if clock > c.deadline or feed < 0:
                    good = False
                    break
                at = c.xy
            if good and clock + (manhattan(at, self.shed) + 1 if bag else 0) <= self.horizon:
                return True
        return False

    def allowed(self):
        if self.ended:
            return []
        result = []
        for i, c in enumerate(self.chains):
            finish = self.t + manhattan(self.pos, c.xy) + c.operations
            if self.done & (1 << i) or finish > c.deadline or self.wheat < c.wheat:
                continue
            if self._can_finish_required(c.xy, finish, self.done | (1 << i),
                                         self.bag + c.value, self.wheat - c.wheat):
                result.append(i)
        if self.bag:
            t = self.t + manhattan(self.pos, self.shed) + 1
            if t <= self.horizon and self._can_finish_required(self.shed, t, self.done, 0, self.wheat):
                result.append(self.RETURN)
        if not self.bag and all(not c.required or self.done & (1 << i) for i, c in enumerate(self.chains)):
            result.append(self.END)
        return result

    def heuristic(self, allowed):
        required = [i for i in allowed if i < 5 and self.chains[i].required]
        if required:
            return min(required, key=lambda i: self.chains[i].deadline)
        harvests = [i for i in allowed if i < 5]
        if harvests:
            return max(harvests, key=lambda i: self.chains[i].value /
                       (manhattan(self.pos, self.chains[i].xy) + self.chains[i].operations))
        return self.RETURN if self.RETURN in allowed else self.END

    def step(self, action):
        assert action in self.allowed(), "Action must belong to the feasible candidate set"
        before, reward = self.t, 0
        if action < 5:
            c = self.chains[action]
            self.t += manhattan(self.pos, c.xy) + c.operations
            self.pos = c.xy
            self.bag += c.value
            self.wheat -= c.wheat
            # 短场景模拟器一次完成确定性任务链；环境接口逐步核验操作结果。
            self.done |= 1 << action
            name = c.name
        elif action == self.RETURN:
            self.t += manhattan(self.pos, self.shed) + 1
            self.pos = self.shed
            reward, self.bag = self.bag, 0
            self.cash += reward
            name = "RETURN + DROP + accepted SELL"
        else:
            self.ended = True
            name = "END: no further commitment"
        self.trace.append({"start": before, "finish": self.t, "chain": name,
                           "receipt": reward, "cash": self.cash, "bag_value": self.bag})
        return reward, max(1, self.t-before)

    def result(self):
        return {"cash": self.cash, "unrealized": self.bag,
                "missed_required": sum(c.required and not self.done & (1 << i)
                                       for i, c in enumerate(self.chains)),
                "elapsed_turns": self.t}


class SaleEnv:
    """八个市场 tick，使用单商品线性价格模型。

    每 tick 按入库、出售、需求消耗的顺序结算。
    tick t 的出售为 t+1 的入库预留容量；策略观测包含带噪声的对手供给信号。
    """
    horizon, capacity = 8, 18
    HOLD, MINIMUM, HALF, ALL = range(4)

    def __init__(self, seed):
        rng = random.Random(seed)
        self.demand = rng.choice([2, 6])
        self.signal = rng.randrange(2)
        self.arrival_class = rng.randrange(2)
        self.incoming = {2: 4 + self.arrival_class * 4, 5: 5}
        # 模拟器保存实际出货事件；策略通过 signal 观测公开供给信号。
        self._opponent_tick = rng.choice([3, 4, 6])
        self._opponent_qty = (rng.choice([10, 18, 24]) if rng.random() < (.8 if self.signal else .2) else 0)
        self.t, self.warehouse, self.market = 0, rng.choice([8, 12]), rng.choice([8, 16])
        self.cash, self.lost, self.feed_missing = 0, 0, 0
        self.ended, self.trace = False, []
        self._last_decision = None

    def protected(self):
        return 2 if self.t < 4 else 0

    def tradable(self):
        return max(0, self.warehouse - self.protected())

    def minimum(self):
        consume_before_arrival = 2 if self.t+1 == 4 else 0
        return min(self.tradable(), max(0, self.warehouse + self.incoming.get(self.t+1, 0)
                                        - consume_before_arrival - self.capacity))

    def quantities(self):
        n, m = self.tradable(), self.minimum()
        return [0, m, max(m, math.ceil(n/2)), n]

    def allowed(self):
        if self.ended:
            return []
        if self.t == self.horizon-1:
            return [self.ALL]
        # 每个 tick 按价格、供给或入库倒计时决策，并检查仓容要求。
        return [i for i, n in enumerate(self.quantities()) if n >= self.minimum()]

    def state(self):
        # 状态由当期可见库存、需求、供给信号和我方入库计划构成。
        return (self.t, self.warehouse, self.market // 4, self.demand,
                self.signal, self.arrival_class)

    def heuristic(self, allowed):
        a = self.ALL if self.signal or self.t >= 5 else self.MINIMUM
        return a if a in allowed else allowed[0]

    @staticmethod
    def price(stock):
        return max(1., 30. - .5*stock)

    def _sell(self, quantity):
        assert 0 <= quantity <= self.tradable(), "Sale quantity must fit unreserved warehouse stock"
        receipt = 0.
        for _ in range(quantity):
            receipt += self.price(self.market)
            self.market += 1
            self.warehouse -= 1
        self.cash += receipt
        return receipt

    def step(self, action):
        assert action in self.allowed()
        q = self.quantities()[action]
        protected, before, market_before = self.protected(), self.warehouse, self.market
        reward = self._sell(q)
        self.trace.append({"tick": self.t, "warehouse_before_sale": before,
                           "protected": protected, "market_before_sale": market_before,
                           "quantity": q, "receipt": reward, "warehouse_after_sale": self.warehouse,
                           "space_reserved_for_next_arrival": self.capacity-self.warehouse})
        if self.t == self._opponent_tick:
            self.market += self._opponent_qty
        self.market = max(0, self.market-self.demand)
        self.t += 1
        if self.t >= self.horizon:
            self.ended = True
        else:
            if self.t == 4:
                self.feed_missing += max(0, 2-self.warehouse)
                self.warehouse = max(0, self.warehouse-2)
            incoming = self.incoming.get(self.t, 0)
            accepted = min(incoming, self.capacity-self.warehouse)
            self.warehouse += accepted
            self.lost += incoming-accepted
        return reward, 1

    def result(self):
        return {"cash": self.cash, "lost_units": self.lost,
                "missing_feed": self.feed_missing, "unsold_units": self.warehouse}


def choose(q, state, allowed, fallback):
    if state not in q:
        return fallback, True
    row = q[state]
    # Deterministic heuristic breaks ties and handles unknown feasible actions.
    best = max(allowed, key=lambda a: (row.get(a, 0.), a == fallback))
    return best, False


def train(env_type, episodes, seed):
    q = defaultdict(dict)
    visits = defaultdict(int)
    rng = random.Random(seed)
    started = time.perf_counter()
    for episode in range(episodes):
        env = env_type(rng.randrange(1_000_000))
        epsilon = .30 - .25 * episode / max(1, episodes-1)
        while not env.ended:
            state, allowed = env.state(), env.allowed()
            assert allowed, "Safety verifier must preserve an executable continuation"
            baseline = env.heuristic(allowed)
            action = rng.choice(allowed) if rng.random() < epsilon else choose(q, state, allowed, baseline)[0]
            reward, duration = env.step(action)
            future = 0. if env.ended else max(q.get(env.state(), {}).get(a, 0.) for a in env.allowed())
            # 有限期现金目标取 gamma=1；使用折扣时按任务持续时间计算 gamma**duration。
            visits[(state, action)] += 1
            alpha = .35 / (1 + .002*visits[(state, action)])
            old = q[state].get(action, 0.)
            q[state][action] = old + alpha*(reward + future-old)
    return dict(q), {"episodes": episodes, "seconds": time.perf_counter()-started,
                    "states": len(q), "state_actions": len(visits), "gamma": 1.0, "seed": seed}


def run_episode(env_type, seed, q=None, mode="heuristic"):
    env, unknown, choices = env_type(seed), 0, 0
    while not env.ended:
        state, allowed = env.state(), env.allowed()
        if mode == "rl":
            action, unseen = choose(q, state, allowed, env.heuristic(allowed))
            unknown += unseen
        elif mode == "immediate":
            action = env.ALL
        else:
            action = env.heuristic(allowed)
        env.step(action)
        choices += 1
    result = env.result()
    result.update(unknown_states=unknown, decisions=choices)
    return result, env.trace


def evaluate(env_type, q, seeds):
    modes = ["heuristic", "rl"] if env_type is MobileEnv else ["immediate", "heuristic", "rl"]
    rows, aggregates = [], {}
    for seed in seeds:
        row = {"seed": seed}
        for mode in modes:
            row[mode] = run_episode(env_type, seed, q, mode)[0]
        rows.append(row)
    for mode in modes:
        aggregates[mode] = {"cash": summary([r[mode]["cash"] for r in rows])}
        for key in rows[0][mode]:
            if key != "cash":
                aggregates[mode][key+"_total"] = sum(r[mode][key] for r in rows)
    comparisons = {}
    for baseline in modes[:-1]:
        delta = [r["rl"]["cash"]-r[baseline]["cash"] for r in rows]
        comparisons["rl_minus_"+baseline] = {**summary(delta),
            "higher": sum(d>0 for d in delta), "equal": sum(d==0 for d in delta), "lower": sum(d<0 for d in delta)}
    return {"aggregates": aggregates, "comparisons": comparisons, "rows": rows,
            "example_seed": seeds[0], "example": {m: run_episode(env_type, seeds[0], q, m)[1] for m in modes}}
