"""作物合同、官方单格生物机制与技能反馈执行器。

通过 AST 从本地官方源码加载生物与交易函数，计算合同中的产出和批量收入。
"""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
import random
import statistics
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

ENGINE = (Path(__file__).resolve().parents[2] / "720_step_value_code" /
          "expansion_research/schedule_training_r3/sources/official_engine.py")


def engine_kernel():
    source = ENGINE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) or
             isinstance(n, ast.Assign) and all(isinstance(t, ast.Name) and t.id.isupper() for t in n.targets)]
    ns = {"math": math, "json": json, "random": random}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(ENGINE), "exec"), ns)
    ns["source_sha256"] = hashlib.sha256(source.encode()).hexdigest()
    return SimpleNamespace(**ns)


class ReservationLedger:
    def __init__(self):
        self.entries = {}

    def reserve(self, owner, item, n, available):
        already = sum(q for (who, what), q in self.entries.items() if what == item and who != owner)
        if n < 0 or already+n > available:
            return False
        self.entries[(owner, item)] = n
        return True

    def release(self, owner, item):
        self.entries.pop((owner, item), None)


class FeedbackExecutor:
    """单地块任务队列，根据下一次观测确认操作结果。

    调度层负责到达目标坐标；执行器检查前置状态、保留失败节点并推进已完成任务。
    """
    def __init__(self, tasks):
        self.tasks = list(tasks)
        self.pending = None
        self.events = []

    def next_action(self, tile, bag, seeds):
        while self.tasks:
            op = self.tasks[0]
            name = op[0]
            already = (
                name == "HARVEST" and (not isinstance(tile, dict) or tile.get("yield_units", 0) == 0)
                or name == "FEED" and isinstance(tile, dict) and tile.get("fed_today")
                or name == "CARE" and isinstance(tile, dict) and tile.get("cared_today")
                or name == "WATER" and isinstance(tile, dict) and tile.get("watered_today")
                or name == "PLANT" and isinstance(tile, dict) and tile.get("crop") == op[1]
            )
            if already:
                self.events.append({"op": name, "status": "ALREADY_DONE: skip this node only"})
                self.tasks.pop(0)
                continue
            if name == "PLANT" and isinstance(tile, dict) and tile.get("kind") == "WEED":
                self.pending = ("DIG", copy.deepcopy(tile))
                return ["DIG"]  # retain original PLANT and subsequent WATER
            if name == "PLANT" and (tile is not None or seeds.get(op[1], 0) < 1):
                self.events.append({"op": name, "status": "BLOCKED: retain commitment"})
                return None
            if name == "FEED" and bag.get("WHEAT", 0) < 1:
                self.events.append({"op": name, "status": "BLOCKED: wheat pickup/reassign needed"})
                return None
            self.pending = (name, copy.deepcopy(tile))
            return op
        return None

    def acknowledge(self, new_tile):
        if self.pending is None:
            return False
        name, before = self.pending
        tile = new_tile if isinstance(new_tile, dict) else {}
        succeeded = {
            "DIG": new_tile is None,
            "PLANT": tile.get("crop") == (self.tasks[0][1] if self.tasks and len(self.tasks[0]) > 1 else None) and bool(tile),
            "WATER": bool(tile.get("watered_today")),
            "FEED": bool(tile.get("fed_today")),
            "CARE": bool(tile.get("cared_today")),
            "HARVEST": bool(isinstance(before, dict) and before.get("yield_units", 0) > 0 and
                            (new_tile is None or tile.get("yield_units", 0) == 0)),
        }.get(name, False)
        self.events.append({"op": name, "status": "OBSERVED_DONE" if succeeded else "FAILED: retry or repair"})
        if succeeded and name != "DIG":
            self.tasks.pop(0)
        self.pending = None
        return succeeded


def sell_batch(E, item, market_stock, n):
    receipt = 0
    for _ in range(n):
        price = E.market_price(item, market_stock)
        receipt += price
        if price > 1:
            market_stock += 1
    return receipt, market_stock


def crop_contract(E, crop, start_day, end_day, visits, labor_by_day, distance_to_shed=3):
    """在资金充足、每日浇水、施肥用量为零的条件下计算作物合同。

    日期为 1..30；收入按三个恒定市场库存情景计算，劳动和运输按日核验。
    """
    if crop == "EMPTY":
        return {"crop": crop, "feasible": True, "reasons": [], "cash_net": 0., "score": 0.,
                "actions": [], "harvests": [], "labor": 0, "seed_cost": 0}
    cd = E.CROPS[crop]
    first = start_day + cd["first_yield_day"]
    if first > end_day:
        return {"crop": crop, "feasible": False, "reasons": ["首次成熟晚于终局"],
                "first_harvest_day": first, "actions": [], "harvests": []}
    farm = {"tiles": [[None]], "farmer": [0, 0], "hands": []}
    private = {"shed": {}, "seeds": {crop: 1}, "inventories": [{}]}
    actions, harvests = [], []
    nonongoing_harvest = min(end_day, start_day + (3 if crop in ("WHEAT", "CARROT") else cd["max_yield_day"]))
    for day in range(start_day, end_day+1):
        if day == start_day:
            E._apply_unit_action(farm, private, 0, ["PLANT", crop], 1, day-1, 24)
            actions.append({"day": day, "op": "PLANT"})
        if farm["tiles"][0][0] is None:
            break
        E._apply_unit_action(farm, private, 0, ["WATER"], 1, day-1, 24)
        actions.append({"day": day, "op": "WATER"})
        tile = farm["tiles"][0][0]
        if day >= first and tile["yield_units"] and (cd["ongoing"] or day >= nonongoing_harvest):
            before = private["inventories"][0].get(crop, 0)
            E._apply_unit_action(farm, private, 0, ["HARVEST"], 1, day-1, 24)
            qty = private["inventories"][0].get(crop, 0)-before
            actions.append({"day": day, "op": "HARVEST"})
            harvests.append({"day": day, "units": qty, "assumed_unload_hour": 14+distance_to_shed+1})
        if farm["tiles"][0][0] is None:
            break
        E._daily_refresh_plants(farm, day-1, 24)
    load = Counter(a["day"] for a in actions)
    for day in list(load):
        load[day] += 2  # illustrative marginal movement budget, explicitly priced
    for h in harvests:
        load[h["day"]] += distance_to_shed+1
    reasons = []
    for day, n in load.items():
        if day not in visits:
            reasons.append(f"Day{day}没有区域访问")
        elif labor_by_day.get(day, 0) < n:
            reasons.append(f"Day{day}劳动不足:需要{n},可用{labor_by_day.get(day, 0)}")
    if any(h["assumed_unload_hour"] > (22 if h["day"] == 30 else 23) for h in harvests):
        reasons.append("运输不能在兑现截止前完成")
    quantities = sum(h["units"] for h in harvests)
    receipts = [sell_batch(E, crop, E.MARKET_I0+offset, quantities)[0] for offset in (-40, 0, 40)]
    net = statistics.mean(receipts)-cd["seed"]
    return {"crop": crop, "feasible": not reasons, "reasons": reasons,
            "first_harvest_day": first, "actions": actions, "harvests": harvests,
            "seed_cost": cd["seed"], "labor_by_day": dict(load), "labor": sum(load.values()),
            "scenario_receipts": receipts, "cash_net": net,
            "score": net-3*sum(load.values()), "shadow_price_per_turn": 3,
            "post_harvest": "standing crop" if cd["ongoing"] else "empty tile"}


def ev_examples(E):
    short = {d: 12 for d in range(17, 21)}
    long = {d: 12 for d in range(17, 31)}
    terminal = {d: 12 for d in range(28, 31)}
    cases = {}
    for name, day, budget, dist in [("Day17_existing_short_visit", 17, short, 3),
                                   ("Day17_full_lifecycle_visit", 17, long, 3),
                                   ("Day28_near_shed", 28, terminal, 3),
                                   ("Day28_far_corner", 28, terminal, 9)]:
        contracts = [crop_contract(E, c, day, 30, set(budget), budget, dist)
                     for c in ["EMPTY", "WHEAT", "CARROT", "TOMATO", "STRAWBERRY"]]
        best = max((c for c in contracts if c["feasible"]), key=lambda c: c["score"])
        cases[name] = {"winner": best["crop"], "contracts": contracts}
    return cases
