# EV、局部决策学习与快照示例

示例代码围绕三个决策对象组织：作物生命周期合同、流动工人的完整任务链、仓库商品的出售数量。任务链与出售使用表格 Q-learning，启发式检查材料、期限、必要任务和仓容。

样本和示例的统一核验命令为 `python verify_examples.py`，检查文件校验值、观测与特征标签的对应关系、终局边界、EV 结果，以及材料预约和任务连续性。

## 1. 文件与运行环境

| 文件 | 作用 |
|---|---|
| [ev_demo.py](../examples/ev_demo.py) | 比较不同访问与终局运输条件下的作物合同 |
| [ev_core.py](../examples/ev_core.py) | 单格生物推演、逐件交易收入、合同核验、共享材料预约和反馈执行器 |
| [train_demo.py](../examples/train_demo.py) | 运行任务链或出售训练，保存评测与策略表 |
| [rl_core.py](../examples/rl_core.py) | 两个短场景环境、动作掩码、Q-learning 与配对评估 |
| [slice_snapshots.py](../examples/slice_snapshots.py) | 按比赛、日期、时刻、视角和区域读取快照 |

使用 Python 3.10 及以上版本，示例自身使用标准库。EV 示例默认读取同级 `720_step_value_code` 目录中的固定官方引擎源码，通过 AST 加载其中的生物与交易函数；也可以使用 `--engine` 指定源码路径。

以下命令从 `farm_agent_architecture` 目录执行。

## 2. 作物 EV

```sh
python examples/ev_demo.py
python examples/ev_demo.py --output outputs/ev.json
```

候选合同记录种植日期、维护和收获安排、按日劳动需求、运输时间、收入情景以及收获后状态。筛选顺序为：

1. 检查成熟是否赶得上终局。
2. 检查每个维护日是否存在区域访问和劳动容量。
3. 检查卸货能否在对应截止前完成。
4. 比较逐件出售收入、投入费用和劳动机会成本。

算例采用每日浇水、每日边际移动 2 turns、劳动机会成本 3/turn；市场库存设为基准 −40、基准、基准 +40 三个常量情景。近仓和远角落分别使用 3、9 turns 的到仓距离，收获后的卸货时刻按 `14 + 距离 + 1` 计算。

| 条件 | 选择结果 | 主要原因 |
|---|---|---|
| Day17，只承诺访问至 Day20 | 胡萝卜 | 长周期作物后续维护缺少访问；胡萝卜在可行短作物中得分更高 |
| Day17，后续完整访问可用 | 胡萝卜 | 番茄等候选可以维护，但该成本设置下劳动计价后收益较低 |
| Day28，近仓 | 胡萝卜 | 能在 Day30 成熟并完成运输，劳动计价分数为 2 |
| Day28，远角落 | 留空 | 最后一天劳动不足，预测卸货超过截止时间 |

同一模块中的 `FeedbackExecutor` 演示“发出动作、读取新状态、确认完成”的执行顺序；`ReservationLedger` 根据共享可用量为不同工人预约材料。

## 3. 流动工人的任务链学习

```sh
python examples/train_demo.py --mode workers --output outputs/workers.json
```

默认训练 18,000 个短场景，使用 400 个独立测试 seed。环境包含一个流动工人、五条候选链和 17-turn 时域；18 种结构参数组合改变收获价值与杂草成本。

候选覆盖动物喂养照料、到期浇水、近仓辅助收获、远处收获和除草收获。每次选择前，核验器计算移动、操作、剩余必要任务和最终返仓时间，生成可行动作集合。返仓选项合并运输、卸货与成功出售。

状态包含时间、位置、已完成任务、背包价值、饲料和可见任务参数。收入在兑现时计入奖励；有限期目标取折扣因子 1。未见状态按启发式选择，已有状态按 Q 值选择。

默认参数下，400 个测试样本中有 69 个收入提高、331 个持平；配对差均值为 1.1625，中位数为 0。两种方法均完成必要任务并兑现携货。

## 4. 出售 RL 训练

```sh
python examples/train_demo.py --mode sales --output outputs/sales.json
python examples/train_demo.py --mode sales --policy-output outputs/sales_policy.json
```

默认训练 30,000 个短场景，评估 400 个独立测试 seed。环境包含单商品和 8 个市场 tick，价格随市场库存线性下降。

动作集合为持有、最低腾仓量、约一半和全部可交易余量。安全层保护一次 2 单位生产投入，并根据下一批入库量确定出售下限。每 tick 按入库、出售和需求消耗的顺序执行，前一 tick 的出售为后续卸货预留空间。

对手实际出货时刻与数量由模拟器保存，策略观测包含带噪声的公开供给信号。终局按逐件交易结果清算剩余库存。配对评估为各策略使用相同初始状态和外部供给事件。

| 对照 | 配对差均值 | 中位数 | P10 | 收入提高 / 持平 / 降低 |
|---|---:|---:|---:|---:|
| Q-learning − 及时出售 | +46.785 | +39 | +26 | 400 / 0 / 0 |
| Q-learning − 信号启发式 | +32.3675 | +31 | −22 | 310 / 0 / 90 |

模型状态、训练参数、配对结果与作物合同算例汇总在 [local_experiment_summary.json](../data/local_experiment_summary.json)。

## 5. 保存与调整实验

`--episodes` 调整训练场数，`--test-seeds` 调整评估样本数。`--output` 保存逐样本结果与行动轨迹，`--policy-output` 保存可检查的 Q 表。输出文件写入 `outputs/`，由 `.gitignore` 管理。

```sh
python examples/train_demo.py --mode workers --episodes 2000 --test-seeds 50 --output outputs/workers_small.json
```

先固定环境与评估样本，再调整候选任务、状态编码或奖励定义，可以比较具体机制带来的变化。快照切片命令和字段示例见 [数据说明](../data/README.md)。
