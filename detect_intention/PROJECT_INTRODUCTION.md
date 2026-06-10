# Detect Intention 项目介绍

## 1. 项目目标

`detect_intention` 是一个面向雷达目标意图识别与应对措施生成的小项目。它的目标是构建一条完整流程：

```text
雷达仿真数据生成
  -> 目标轨迹与意图标签构建
  -> Transformer 意图识别训练
  -> 模型评估与预测可视化
  -> 大模型生成态势解释和应对措施
```

本项目当前关注的不是传统目标检测本身，而是：

```text
给定雷达监测到的目标运动序列，判断目标意图，并给出应对措施。
```

当前支持 5 类目标意图：

| ID | 意图 | 含义 | 默认应对措施 |
| --- | --- | --- | --- |
| 1 | `benign_transit` | 正常通过、低风险航行 | `monitor` |
| 2 | `approach` | 接近雷达或警戒区域 | `increase_tracking_rate` |
| 3 | `retreat` | 远离雷达或警戒区域 | `monitor` |
| 4 | `loiter_patrol` | 徘徊、巡逻、局部反复运动 | `classify_and_shadow` |
| 5 | `intercept` | 朝保护点或关键区域逼近 | `alert_and_allocate_tracker` |

当前支持 4 类威胁等级：

| 等级 | 名称 | 说明 |
| --- | --- | --- |
| 1 | `low` | 低威胁 |
| 2 | `guarded` | 需要关注 |
| 3 | `elevated` | 威胁升高 |
| 4 | `high` | 高威胁 |

## 2. 总体流程

项目完整流程分为 6 步。

### 步骤 1：生成带意图标签的雷达数据集

脚本：

```text
detect_intention/generate_intention_fmcw_dataset.m
```

作用：

1. 生成干净的 FMCW-MIMO 雷达目标仿真数据；
2. 不生成干扰目标、不生成假目标、不生成宽带压制干扰；
3. 为每个真实目标生成轨迹、速度、雷达观测序列；
4. 为每个目标写入意图标签 `intentId / intentName`；
5. 为每个目标写入威胁等级 `threatLevel`；
6. 为每个目标写入默认应对措施 `recommendedActionName`。

推荐生成 compact 数据集：

```matlab
cd('E:\VSCodeProject\radarLLM\detect_intention')
generate_intention_fmcw_dataset(5000, [], 'compact')
```

compact 数据集保存位置：

```text
detect_intention/intention_dataset_compact/
```

compact 数据集参数：

```text
Nfast   = 64
Nchirp  = 16
Nrx     = 4
Nframes = 32
```

它比 full 数据集小很多，适合批量训练，同时仍然保留目标运动趋势和意图识别所需信息。

### 步骤 2：可视化数据集本身

脚本：

```text
detect_intention/visualize_intention_dataset.m
```

作用：

1. 读取生成的 `.mat` 场景；
2. 绘制目标运动轨迹；
3. 绘制保护点和保护区域；
4. 绘制 Range-Doppler 图；
5. 输出数据集复盘图。

运行：

```matlab
cd('E:\VSCodeProject\radarLLM\detect_intention')
visualize_intention_dataset('intention_dataset_compact')
```

输出位置：

```text
detect_intention/intention_dataset_compact/visualization/
```

### 步骤 3：训练 Transformer 意图识别模型

训练模块：

```text
detect_intention/train_intention/
```

核心脚本：

```text
detect_intention/train_intention/train.py
```

当前默认训练模式是：

```text
--input-mode track
```

也就是使用雷达监测得到的目标轨迹序列：

```text
[x, y, vx, vy]
```

作为 Transformer 输入，而不是直接使用原始 IQ 数据。这样做的原因是：目标意图主要由运动趋势决定，例如接近、远离、徘徊、逼近保护点等。轨迹序列比原始 IQ 更直接、更稳定，也更适合当前意图识别任务。

模型训练目标只保留 3 个核心 loss：

| Loss | 含义 | 是否核心 |
| --- | --- | --- |
| `loss_intent` | 意图分类损失 | 是 |
| `loss_threat` | 威胁等级分类损失 | 是 |
| `loss_state` | 未来轨迹回归损失 | 是 |

当前不再训练：

```text
objectness loss
targetClassId loss
```

原因是当前 `track` 模式输入已经是目标轨迹槽位，目标是否存在不是核心问题；目标类别更依赖散射特性和微多普勒，而不是轨迹趋势，因此不作为当前主任务。

推荐训练命令：

```powershell
conda activate radar

python detect_intention/train_intention/train.py `
  --epochs 40 `
  --scene-batch-size 128 `
  --device cuda `
  --lambda-intent 2.0 `
  --lambda-threat 0.5 `
  --lambda-state 0.5
```

训练输出位置：

```text
detect_intention/train_intention/runs/run_YYYYMMDD_HHMMSS/
```

### 步骤 4：评估训练好的模型

脚本：

```text
detect_intention/train_intention/evaluate.py
```

作用：

1. 加载训练好的 checkpoint；
2. 在验证集或训练集上进行推理；
3. 输出整体指标；
4. 输出每个目标的结构化预测结果。

推荐使用意图 loss 最优模型：

```text
best_by_intent.pt
```

运行：

```powershell
python detect_intention/train_intention/evaluate.py `
  --checkpoint detect_intention/train_intention/runs/run_20260604_005306/best_by_intent.pt `
  --dataset-dir detect_intention/intention_dataset_compact `
  --split val
```

生成文件：

```text
eval_metrics.json
predictions.jsonl
```

### 步骤 5：可视化模型预测轨迹

脚本：

```text
detect_intention/train_intention/visualize_model_predictions.py
```

作用：

1. 加载训练好的 best 模型；
2. 读取验证集场景；
3. 绘制目标历史轨迹；
4. 绘制真实未来轨迹；
5. 绘制模型预测未来轨迹；
6. 标注预测意图、真实意图、威胁等级和推荐动作；
7. 绘制保护点和保护区域。

运行：

```powershell
python detect_intention/train_intention/visualize_model_predictions.py `
  --checkpoint detect_intention/train_intention/runs/run_20260604_005306/best_by_intent.pt `
  --dataset-dir detect_intention/intention_dataset_compact `
  --split val `
  --max-scenes 8
```

输出位置：

```text
detect_intention/train_intention/runs/run_20260604_005306/prediction_visualizations/
```

### 步骤 6：调用大模型生成态势解释和应对措施

脚本：

```text
detect_intention/train_intention/llm_decision.py
```

作用：

1. 读取 `predictions.jsonl`；
2. 将 Transformer 的结构化预测结果整理成 prompt；
3. 调用 Qwen 或其他 OpenAI-compatible 大模型；
4. 让大模型根据目标运动趋势、威胁等级、保护区距离变化等信息生成最终解释；
5. 输出结构化决策结果。

推荐使用 Qwen-Turbo：

```powershell
$env:DASHSCOPE_API_KEY="你的API_KEY"

python detect_intention/train_intention/llm_decision.py `
  --provider qwen `
  --model qwen-turbo `
  --llm-task infer_intent `
  --predictions detect_intention/train_intention/runs/run_20260604_005306/predictions.jsonl `
  --output detect_intention/train_intention/runs/run_20260604_005306/llm_decisions_qwen_turbo.jsonl
```

如果中途中断或账户额度不足，可以恢复后续跑：

```powershell
python detect_intention/train_intention/llm_decision.py `
  --provider qwen `
  --model qwen-turbo `
  --llm-task infer_intent `
  --predictions detect_intention/train_intention/runs/run_20260604_005306/predictions.jsonl `
  --output detect_intention/train_intention/runs/run_20260604_005306/llm_decisions_qwen_turbo.jsonl `
  --resume
```

## 3. 代码文件说明

### 3.1 MATLAB 数据生成与可视化

#### `generate_intention_fmcw_dataset.m`

生成带意图标签的雷达仿真数据集。

主要输出：

```text
scene_000001.mat
scene_000002.mat
...
index.csv
```

每个 `.mat` 场景中包含：

| 字段 | 作用 |
| --- | --- |
| `iq` | FMCW-MIMO 雷达 IQ 数据 |
| `rdMap` | Range-Doppler 图 |
| `gt` | 目标真值，包括轨迹、速度、意图、威胁等级 |
| `meta` | 场景元信息 |
| `p` | 雷达参数和数据集参数 |

#### `visualize_intention_dataset.m`

对生成的数据集进行整体复盘和单场景可视化。

### 3.2 Python 训练模块

#### `train_intention/dataset.py`

负责读取 MATLAB `.mat` 文件，并构造 PyTorch 训练样本。

主要功能：

1. 读取 `iq`、`gt.pos`、`gt.vel`、`intentId`、`threatLevel`；
2. 构造滑动窗口；
3. 生成训练输入 `state_input`；
4. 生成监督标签 `state_label`、`intent_label`、`threat_label`。

#### `train_intention/model.py`

定义模型结构。

当前核心模型是：

```text
TrackIntentTransformer
```

输入：

```text
[batch, target_slot, time, 4]
```

其中 4 个状态量是：

```text
x, y, vx, vy
```

输出：

```text
future trajectory
intent logits
threat logits
```

#### `train_intention/train.py`

训练入口。

主要功能：

1. 加载数据集；
2. 自动读取数据集尺寸；
3. 构建 Transformer 模型；
4. 计算 `loss_intent / loss_threat / loss_state`；
5. 记录训练和验证指标；
6. 保存 checkpoint；
7. 执行早停。

#### `train_intention/evaluate.py`

模型评估脚本。

主要功能：

1. 加载 checkpoint；
2. 对验证集进行推理；
3. 生成总体评估指标；
4. 输出每个目标的结构化预测结果。

#### `train_intention/visualize_model_predictions.py`

模型预测可视化脚本。

主要功能：

1. 绘制观测轨迹；
2. 绘制真实未来轨迹；
3. 绘制预测未来轨迹；
4. 标注预测意图和威胁等级；
5. 标注应对措施。

#### `train_intention/llm_decision.py`

大模型决策脚本。

主要功能：

1. 读取 `predictions.jsonl`；
2. 将结构化结果转换为 prompt；
3. 支持规则模式、OpenAI-compatible 模式、Qwen 模式；
4. 支持 `qwen-turbo`；
5. 支持续跑 `--resume`；
6. 输出最终态势解释和应对措施。

#### `train_intention/plot_loss_curves.py`

训练曲线绘制脚本。

运行：

```powershell
python detect_intention/train_intention/plot_loss_curves.py `
  detect_intention/train_intention/runs/run_20260604_005306/metrics.jsonl
```

输出：

```text
loss_curves.png
```

#### `train_intention/analyze_run.py`

训练日志分析脚本。

用于快速查看最佳 epoch、验证集指标和泛化差距。

## 4. 输出文件说明

### 4.1 数据集文件

#### `intention_dataset_compact/index.csv`

数据集索引文件。

每一行对应一个场景，包含：

```text
scene_id
file
n_targets
intent_names
max_threat_level
recommended_actions
all_targets_visible
```

#### `intention_dataset_compact/scene_XXXXXX.mat`

单个雷达仿真场景。

包含完整雷达观测、目标轨迹、意图标签和威胁标签。

### 4.2 训练输出文件

训练输出目录示例：

```text
detect_intention/train_intention/runs/run_20260604_005306/
```

#### `metrics.jsonl`

训练日志文件。

每个 epoch 一行，包含：

```text
train.loss_total
train.loss_intent
train.loss_threat
train.loss_state
train.intent_accuracy
train.threat_accuracy
train.ade_m
val.loss_total
val.loss_intent
val.loss_threat
val.loss_state
val.intent_accuracy
val.threat_accuracy
val.ade_m
```

重点看：

```text
val.intent_accuracy
val.threat_accuracy
val.ade_m
val.loss_intent
```

#### `best_by_intent.pt`

验证集 `loss_intent` 最优的模型。

推荐用于后续评估和大模型推理。

#### `best_by_intent_accuracy.pt`

验证集意图准确率最高的模型。

#### `best_by_ade.pt`

未来轨迹预测误差最小的模型。

#### `last.pt`

最后一个 epoch 的模型。

### 4.3 评估输出文件

#### `eval_metrics.json`

整体评估指标。

主要字段：

| 字段 | 含义 |
| --- | --- |
| `intent_accuracy` | 意图分类准确率 |
| `threat_accuracy` | 威胁等级准确率 |
| `ade_m` | 平均轨迹误差 |
| `fde_m` | 最终点轨迹误差 |
| `loss_intent` | 意图分类损失 |
| `loss_threat` | 威胁等级损失 |
| `loss_state` | 轨迹回归损失 |

#### `predictions.jsonl`

逐目标预测结果。

每一行对应一个目标，包含：

```text
scene_file
slot
pred_intent_name
pred_intent_confidence
true_intent_name
pred_threat_name
pred_threat_confidence
true_threat_name
recommended_action_rule
observed_range_trend
observed_protected_distance_trend
predicted_range_trend
predicted_protected_distance_trend
```

该文件是大模型决策模块的输入。

### 4.4 可视化输出文件

#### `loss_curves.png`

训练曲线图。

展示：

```text
loss_total
loss_intent
loss_threat
loss_state
```

每条曲线同时包含训练集和验证集。

#### `prediction_visualizations/scene_XXXXXX_prediction.png`

模型预测轨迹可视化图。

图中包含：

```text
历史观测轨迹
真实未来轨迹
预测未来轨迹
预测意图
真实意图
威胁等级
推荐动作
保护点
保护区域
```

#### `prediction_visualization_summary.json`

预测可视化对应的结构化摘要。

### 4.5 大模型输出文件

#### `llm_decisions_qwen_turbo.jsonl`

Qwen-Turbo 输出的目标意图解释和应对措施。

每一行对应一个目标，包含：

```text
scene_file
slot
prompt
prediction
decision
```

其中 `decision` 是最终大模型决策结果：

```text
final_intent
priority
action
reason
follow_up
provider
```

字段解释：

| 字段 | 含义 |
| --- | --- |
| `final_intent` | 大模型判断的最终意图 |
| `priority` | 处置优先级 |
| `action` | 建议应对措施 |
| `reason` | 判断原因 |
| `follow_up` | 后续操作建议 |

## 5. 当前项目结论

当前实验结果表明：

1. 使用轨迹序列进行意图识别是有效的；
2. 当前模型在验证集上意图识别准确率已经接近 0.99；
3. 目标意图与运动趋势高度相关；
4. Transformer 适合承担数值识别任务；
5. 大模型更适合承担态势解释、决策表述和人机交互说明任务。

当前推荐系统定位：

```text
Transformer = 雷达目标意图识别器
Qwen/大模型 = 态势解释与应对措施生成器
```

这种分工比让大模型直接处理雷达张量更稳定，也更容易解释给专家。

## 6. 讲解建议

可以按照以下逻辑汇报：

1. 先说明项目目的：从雷达目标轨迹中识别意图，并自动生成应对措施；
2. 再说明数据集构建：仿真生成干净目标、轨迹、意图标签和威胁标签；
3. 再说明模型设计：使用 Transformer 处理目标轨迹时间序列；
4. 再说明训练目标：只训练意图、威胁和轨迹预测三个核心任务；
5. 再展示训练结果：意图准确率、威胁准确率、ADE/FDE；
6. 再展示预测图：历史轨迹、未来预测、保护区域、应对措施；
7. 最后说明大模型作用：不直接看雷达数据，而是基于结构化预测结果生成解释和处置建议。

一句话总结：

```text
本项目实现了从雷达目标运动序列到目标意图识别，再到大模型辅助决策解释的完整闭环。
```

