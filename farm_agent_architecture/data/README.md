# 对局快照、特征与切片示例

这批数据来自 720 时刻估值实验使用的 841 场对局。每场保留 720 个时刻、双方各一个合法观测视角，并转换为 483 维特征。

| 数据层 | 组织方式 | 用途 |
|---|---|---|
| 原始对局 | 按 episode 保存完整记录 | 数据来源与时间顺序 |
| 观测快照 | `episode → step → seat → observation` | 盘面、资源、市场与工人状态 |
| 特征矩阵 | `float32[841, 720, 2, 483]` | 全时域估值训练与评测 |
| 终局标签 | `float64[841, 2]` | 各自终局现金与剩余净增长监督 |
| 数据划分 | 505 训练、168 开发、168 历史确认 | 按整场比赛保留分组 |

全部时刻共 1,211,040 条视角记录，训练集的非终局状态共 726,190 条。特征矩阵约 2.34 GB。公开示例保留 3 场、48 条观测及其对应特征和标签，样本文件合计约 0.98 MB。

## 1. 示例选择

每个既有分组按 episode ID 排序，取第一场；随后取固定时间点和双方视角。选择规则与对局结果、模型误差和最终收益独立。

| 分组 | Episode ID | 时间点数 | 视角数 | 记录数 |
|---|---|---:|---:|---:|
| 训练 | 112517734 | 8 | 2 | 16 |
| 开发 | 112516288 | 8 | 2 | 16 |
| 历史确认 | 112518940 | 8 | 2 | 16 |

| step | 展示日期 | hour | 切片用途 |
|---:|---:|---:|---|
| 0 | Day1 | 0 | 开局状态 |
| 263 | Day11 | 23 | 日切前的工人、携货和维护状态 |
| 264 | Day12 | 0 | 日切后的归仓、生产与人员状态 |
| 384 | Day17 | 0 | 中期经营盘面 |
| 385 | Day17 | 1 | 同一局的相邻回合 |
| 696 | Day30 | 0 | 终局日初库存和待收产物 |
| 718 | Day30 | 22 | 最后可决策时刻 |
| 719 | Day30 | 23 | 终局状态与实际现金 |

原始 `day` 从 0 开始，`step = day × 24 + hour`；`day_display` 在切片元数据中按 1–30 展示。

## 2. 文件与对应关系

| 文件 | 内容 |
|---|---|
| [observations.jsonl](samples/observations.jsonl) | 48 条原始观测，每行包含元数据和 observation |
| [features.jsonl](samples/features.jsonl) | 与观测一一对应的 483 维 float32 特征值，以 JSON 数组展示 |
| [feature_names.json](samples/feature_names.json) | 特征列名及固定顺序 |
| [labels.jsonl](samples/labels.jsonl) | 终局现金与相对当前现金的剩余净增长 |
| [configurations.json](samples/configurations.json) | 对应比赛的环境配置 |
| [manifest.json](samples/manifest.json) | 选择规则、来源哈希、样本数量与文件校验值 |

三个 JSONL 文件使用同一个 `sample_id` 连接，例如 `112517734:264:0` 表示该场比赛 step264 的 seat0 视角。

原始 replay 的 SHA256 与快照归档内容的 SHA256 分别记录。示例导出时核对快照归档校验值，并检查特征中的 `cash.own` 与观测现金一致、终局标签与终局状态一致。完整对局索引见 [REPLAYS.csv](../../720_step_value_code/REPLAYS.csv)。

## 3. 观测字段

```text
metadata
  sample_id / episode_id / split / step / seat / day_display

observation
  step / day / hour / player
  farms[0..1]
    money / farmer / hands / hires_today / unlocked_quadrants
    tiles[row][column]
  market
    inventory / prices
  town
    unlocked_shops
  private
    shed / seeds / inventories
```

`farms`、`market` 与 `town` 是当前公共状态；`private` 对应 `player` 指定一方的仓库、种子和工人背包。对双方进行离线评测时，分别读取各自的观测视角。

地块记录作物或动物类型、种植或放置日期、产物数量、当天维护状态、连续缺水或缺料状态及肥效期限。出生、移动与任务衔接依据工人当前坐标和背包核算。

元数据用于索引、分组和切片；特征向量按 `feature_names.json` 读取。终局标签保存在独立文件，训练时通过 `sample_id` 连接。

例如，`112517734:264:0` 的部分字段为：

```json
{
  "metadata": {
    "sample_id": "112517734:264:0",
    "split": "train",
    "step": 264,
    "seat": 0,
    "day_display": 12
  },
  "observation_fields": {
    "day": 11,
    "hour": 0,
    "player": 0,
    "own_cash": 3686.0,
    "farmer": [4, 4],
    "hands": [],
    "shed_WHEAT": 39,
    "shed_FERTILIZER": 17
  }
}
```

其中 `own_cash`、`shed_WHEAT` 等是为阅读展开的字段，分别来自 `farms[player].money` 与 `private.shed`；原始 JSONL 保持观测结构。

## 4. 四种切片方式

### 按时刻比较盘面

选定相同日期和 hour，读取不同比赛的合法观测，可以比较布局、成熟进度、仓库压力与工人分配。示例命令从 `farm_agent_architecture` 目录执行：

```sh
python examples/slice_snapshots.py --day 12 --seat 0 --summary
```

### 按连续回合检查任务衔接

在同一 episode、同一 seat 内选择相邻时间点，用于检查移动、收获、材料变化和任务完成状态。夜间切片 `263 → 264` 用于检查跨日衔接，中期切片 `384 → 385` 用于检查日内变化。

```sh
python examples/slice_snapshots.py --episode 112517734 --seat 0 --start-step 263 --end-step 264
python examples/slice_snapshots.py --episode 112517734 --seat 0 --start-step 384 --end-step 385
```

快照归档保存状态与终局标签。交互式 RL 训练由环境执行所选动作，另行记录动作、下一状态和实际奖励，形成训练转移。

### 按区域分析固定工人

区域切片使用 1-based 行列坐标：NW 为行1–5、列1–5，NE 为行1–5、列6–10，SW 为行6–10、列1–5，SE 为行6–10、列6–10。

```sh
python examples/slice_snapshots.py --episode 112517734 --day 17 --seat 0 --start-step 384 --end-step 384 --region NE
```

区域视图提取对应 25 格，同时保留日期、市场、商店、我方私有资源及全体工人位置。这样可以根据局部布局分析固定路线，并依据全局材料和位置计算跨区工人的衔接。

### 按比赛组织训练与评测

同一场比赛的双方视角、全部时刻和派生窗口保持在相同 split 中。教师交叉拟合进一步按 episode 划分折，使标注某场比赛的教师来自其他折。时间切片与空间切片继承原分组。

真实状态用于训练与常规评测；固定时钟下的跨时刻、跨日状态替换属于单独的诊断数据。跨日替换需要同步平移种植日期、放置日期、肥效到期日与生命周期时刻，以保持源状态的成熟进度和剩余寿命。

## 5. 读取示例

```python
import json
from pathlib import Path

base = Path("data/samples")
rows = [json.loads(line) for line in (base / "observations.jsonl").read_text(encoding="utf-8").splitlines()]
labels = {row["sample_id"]: row for row in map(json.loads, (base / "labels.jsonl").read_text(encoding="utf-8").splitlines())}

sample = next(row for row in rows if row["metadata"]["sample_id"] == "112517734:264:0")
obs = sample["observation"]
own_farm = obs["farms"][obs["player"]]
target = labels[sample["metadata"]["sample_id"]]["terminal_cash"]

print(own_farm["money"], obs["private"]["shed"], target)
```

切片样本将盘面阅读、特征检查和标签连接保留在一个小规模目录中，完整训练继续使用按整场分组的原始归档与特征矩阵。
