# FMCW-MIMO 雷达目标分类与轨迹回归实验报告

## 1. 实验目的

本实验基于 MATLAB 生成的 FMCW-MIMO 雷达仿真数据集，构建一个 raw IQ 输入的深度学习 baseline，用于完成两个任务：

1. 真实目标类别分类；
2. 真实目标未来状态回归，预测未来若干帧的 `[x, y, vx, vy]`。

当前 baseline 只预测真实目标，不预测干扰目标。干扰目标作为输入雷达回波中的背景/干扰存在，用于提升任务复杂度和模型鲁棒性。

## 2. 数据集说明

数据由 `radar_detect/generate_fmcw_traj_dataset.m` 生成。每个场景文件为 MATLAB v7.3 `.mat` 格式，主要包含：

- `iq`
- `rdMap`
- `gt`
- `meta`
- `p`

其中 raw IQ 数据的语义维度为：

```text
iq: [Nfast, Nchirp, Nrx, Nframes]
```

默认参数为：

```text
[128, 32, 8, 48]
```

含义如下：

```text
128: 每个 chirp 的 fast-time ADC 采样点
32 : 每帧 chirp 数
8  : 接收阵元 / 虚拟阵元数
48 : 连续时间帧数
```

标签字段中：

```text
gt.pos: [Nobject, Nframes, 2]，最后一维为 [x, y]
gt.vel: [Nobject, Nframes, 2]，最后一维为 [vx, vy]
```

## 3. 样本构造方式

训练时使用滑动窗口构造样本：

```text
Tin = 16
Tout = 8
stride = 1
```

因此每个 48 帧场景可生成：

```text
48 - 16 - 8 + 1 = 25
```

个训练样本。

输入使用 raw IQ，并拆分为 real/imag 两部分。最终输入张量格式为：

```text
X: [2*Nrx, Tin, Nchirp, Nfast]
```

默认：

```text
[16, 16, 32, 128]
```

通道顺序固定为：

```text
rx1_real, rx1_imag, rx2_real, rx2_imag, ..., rx8_real, rx8_imag
```

IQ 输入归一化使用：

```text
iq_norm = iq / 76.0
```

## 4. 任务定义

当前 baseline 只预测真实目标，干扰目标不作为输出对象。

最大真实目标数为：

```text
K = 4
```

分类类别数为：

```text
C = 5
```

类别定义：

```text
0: background / no-object
1: T1_slow_smooth
2: T2_uav_agile
3: T3_pedestrian
4: T4_fast_maneuver
```

模型输出为：

```text
class_logits: [B, 4, 5]
state_pred:   [B, 4, Tout, 4]
```

其中 `state_pred` 最后一维为：

```text
[x, y, vx, vy]
```

对于不足 4 个真实目标的场景，使用 padding slot：

```text
class_label = 0
state_label = 0
target_mask = 0
```

真实目标 slot：

```text
class_label = gt.targetClassId
target_mask = 1
```

回归损失只对 `target_mask = 1` 的真实目标 slot 计算。

## 5. 标签标准化

在服务器上使用 2000 个场景重新统计标签分布，统计命令为：

```bash
python radar_detect/analyze_dataset_stats.py \
  --dataset-dir radar_detect/fmcw_traj_dataset \
  --tin 16 \
  --tout 8 \
  --skip-input
```

数据集统计结果：

```text
scene count           : 2000
sliding windows       : 50000
target-window labels  : 123950
```

真实目标数量分布：

```text
1 个真实目标: 530 个场景
2 个真实目标: 490 个场景
3 个真实目标: 472 个场景
4 个真实目标: 508 个场景
```

干扰目标数量分布：

```text
0 个干扰目标: 685 个场景
1 个干扰目标: 636 个场景
2 个干扰目标: 679 个场景
```

标签统计值：

```text
x  mean = 45.8870227    std = 13.8679112
y  mean = -0.143020498  std = 9.48956181
vx mean = 6.75004707    std = 3.51767444
vy mean = -0.0552473279 std = 2.12857502
```

标准化方式：

```text
x_norm  = (x  - 45.8870227)    / 13.8679112
y_norm  = (y  - -0.143020498)  / 9.48956181
vx_norm = (vx - 6.75004707)    / 3.51767444
vy_norm = (vy - -0.0552473279) / 2.12857502
```

对应配置已更新到：

```text
radar_detect/train_baseline/config.py
```

## 6. 模型与损失函数

当前模型为 `FmcwBaseline3DCNN`，输入格式为：

```text
[B, 16, 16, 32, 128]
```

模型结构为 3D CNN encoder 加两个输出头：

1. 分类头：输出 `[B, 4, 5]`
2. 回归头：输出 `[B, 4, 8, 4]`

损失函数：

```text
loss = loss_cls + lambda_state * loss_state
```

其中：

```text
loss_cls   = CrossEntropyLoss
loss_state = SmoothL1Loss
lambda_state = 1.0
```

分类损失对所有 slot 计算，包括 background slot。

回归损失只对真实目标 slot 计算，并按照有效元素数量归一化：

```python
valid_count = target_mask.sum() * Tout * 4
loss_state = masked_loss.sum() / max(valid_count, 1)
```

## 7. 训练设置

训练/验证按 scene 划分，避免滑动窗口泄漏。

当前 2000 个 scene 的默认划分为：

```text
train scene: 1600
val scene  : 400
```

对应窗口数：

```text
train windows: 40000
val windows  : 10000
```

冒烟测试命令：

```bash
python radar_detect/train_baseline/train.py \
  --dataset-dir radar_detect/fmcw_traj_dataset \
  --epochs 1 \
  --batch-size 8 \
  --device cuda \
  --print-sample
```

正式训练尝试命令：

```bash
CUDA_VISIBLE_DEVICES=0 python radar_detect/train_baseline/train.py \
  --dataset-dir radar_detect/fmcw_traj_dataset \
  --epochs 20 \
  --batch-size 256 \
  --num-workers 0 \
  --device cuda \
  --print-sample
```

## 8. 实验结果

2000 scene 数据集上，1 epoch 训练输出如下：

```json
{
  "type": "epoch",
  "epoch": 1,
  "train": {
    "loss_total": 1.4781723503112794,
    "loss_cls": 1.1351380800247193,
    "loss_state": 0.3430342705488205,
    "slot_accuracy": 0.5140375,
    "target_accuracy": 0.2565389431505134,
    "position_mae_m": 7.741469310184424,
    "velocity_mae_mps": 1.9384464244618023,
    "ade_m": 12.08328941370054,
    "fde_m": 12.080453148967726
  },
  "val": {
    "loss_total": 1.2957396370887757,
    "loss_cls": 1.0369843923568725,
    "loss_state": 0.2587552393913269,
    "slot_accuracy": 0.566725,
    "target_accuracy": 0.2987772020725389,
    "position_mae_m": 6.0229479291056105,
    "velocity_mae_mps": 1.5422896062030693,
    "ade_m": 9.564573382599983,
    "fde_m": 9.63058073717819
  }
}
```

## 9. 结果分析

与最初 50 scene 数据集相比，2000 scene 数据集的训练表现明显更健康。

在 50 scene 实验中，虽然通过滑动窗口可获得 1250 个窗口样本，但这些样本高度相关。模型很快记住训练场景，表现为：

```text
train loss 快速下降
val loss 持续升高
```

进一步分析发现，验证集总 loss 变差主要来自分类分支，回归分支指标并没有同步恶化。这说明小数据集下分类头容易过拟合并产生过度自信预测。

扩展到 2000 scene 后，1 epoch 的验证指标并未崩溃，且验证集回归指标优于训练集：

```text
val loss_state      < train loss_state
val position_mae_m  < train position_mae_m
val ade_m           < train ade_m
```

这说明数据规模扩展有效缓解了小数据集下的严重过拟合问题，使模型具备更合理的泛化起点。

不过当前仅为 1 epoch 结果，尚不能说明模型已经收敛。后续应重点观察 20 epoch 或更长训练过程中的：

```text
loss_cls
loss_state
position_mae_m
velocity_mae_mps
ade_m
fde_m
target_accuracy
```

不应只看 `loss_total`，因为分类 CE loss 可能主导总损失。

## 10. 训练性能与工程问题

当前训练慢的主要瓶颈不是 GPU 算力，而是数据读取。

数据存储为 MATLAB v7.3 `.mat` 文件，本质为 HDF5。训练时每个 batch 需要：

1. 从磁盘读取 scene；
2. 解析 HDF5；
3. 切滑动窗口；
4. 拆分 real/imag；
5. 构造标签；
6. 送入 GPU。

这些步骤主要发生在 CPU 和磁盘 I/O 上。

实验中使用：

```text
--num-workers 8
```

时出现：

```text
DataLoader worker is killed by signal: Killed
```

该问题通常由内存压力或 HDF5 多进程读取导致。由于每个 worker 拥有独立 Dataset 和独立缓存，多 worker 会导致缓存重复，占用大量内存。2000 个 scene 总大小约 24GB，多 worker 同时缓存会造成内存快速膨胀。

因此当前更稳妥的训练方式是：

```text
--num-workers 0
```

即单进程读取，让 Dataset cache 在主进程中生效，避免多 worker 重复缓存。

## 11. 结论

1. 数据读取、维度转换、IQ real/imag 拆分、滑动窗口、标签标准化和 mask loss 等流程已经验证通过。

2. 原始 50 scene 数据集独立场景数过少，滑动窗口样本相关性强，导致模型严重过拟合。

3. 扩展到 2000 scene 后，训练表现明显改善，1 epoch 验证指标正常，没有出现小数据集下的验证集 loss 崩溃现象。

4. 当前训练瓶颈主要在 HDF5 `.mat` 数据读取和 CPU/I/O，而不是 GPU 计算能力。

5. 当前推荐训练配置为：

```bash
CUDA_VISIBLE_DEVICES=0 python radar_detect/train_baseline/train.py \
  --dataset-dir radar_detect/fmcw_traj_dataset \
  --epochs 20 \
  --batch-size 256 \
  --num-workers 0 \
  --device cuda \
  --print-sample
```

6. 后续评价模型时，应重点关注回归指标：

```text
position_mae_m
velocity_mae_mps
ade_m
fde_m
```

同时关注分类指标：

```text
target_accuracy
loss_cls
```

而不是只依赖 `loss_total`。

## 12. 后续工作

后续建议包括：

1. 完整训练 20 epoch，观察分项指标曲线；
2. 分别保存 `best_by_ade`、`best_by_position_mae`、`best_by_loss_state` checkpoint；
3. 尝试降低分类 loss 权重，例如：

```text
loss = 0.2 * loss_cls + loss_state
```

4. 尝试更小学习率，例如：

```text
lr = 3e-4
```

5. 将 `.mat` 数据预处理为更适合训练的格式，例如：

```text
.pt / .npy / zarr / lmdb
```

以减少 HDF5 读取开销，提高 GPU 利用率。

6. 后续可尝试更强的时序建模结构，例如：

```text
3D CNN + GRU
3D CNN + Transformer
```

7. 在 baseline 稳定后，可进一步扩展任务：

```text
K = 6
同时预测真实目标和干扰目标
```

