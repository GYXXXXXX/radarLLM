# Radar Baseline Handoff

## 项目路径

项目目录：

`/Users/guyuxuan/trae_projects/radarLLM513`

数据集目录：

`radar_detect/fmcw_traj_dataset`

训练工程目录：

`radar_detect/train_baseline`

## 数据集理解

数据由 MATLAB 脚本 `radar_detect/generate_fmcw_traj_dataset.m` 生成。

每个 `scene_xxxxxx.mat` 是 MATLAB v7.3 HDF5 格式，包含：

- `iq`
- `rdMap`
- `gt`
- `meta`
- `p`

其中 `iq` 的 MATLAB 语义维度为：

`[Nfast, Nchirp, Nrx, Nframes]`

默认：

`[128, 32, 8, 48]`

含义：

- `128`: 每个 chirp 的 fast-time ADC 采样点
- `32`: 每帧 chirp 数
- `8`: 接收阵元 / 虚拟阵元
- `48`: 连续时间帧

注意：用 `h5py` 读取 MATLAB v7.3 `.mat` 时，维度常和 MATLAB 显示顺序相反，代码必须修正为语义维度 `[128, 32, 8, 48]`。

`gt.pos` 语义维度：

`[Nobject, Nframes, 2]`

最后一维为 `[x, y]`。

`gt.vel` 语义维度：

`[Nobject, Nframes, 2]`

最后一维为 `[vx, vy]`。

## 当前任务定义

第一版 baseline 只预测真实目标，不预测干扰目标。

- 真实目标数量：每个 scene 至少 1 个，最多 4 个
- 干扰目标数量：0-2 个，但只作为输入背景/干扰，不作为输出对象
- 固定 slot baseline：
  - `K = 4`
  - `C = 5`
- 类别定义：
  - `0`: background / no-object
  - `1`: T1_slow_smooth
  - `2`: T2_uav_agile
  - `3`: T3_pedestrian
  - `4`: T4_fast_maneuver

输入输出：

- `Tin = 16`
- `Tout = 8`
- `stride = 1`
- 每个 48 帧 scene 可切出 `48 - 16 - 8 + 1 = 25` 个滑动窗口样本
- 当前 50 个 scene 共 1250 个窗口样本

输入 `X`：

`[2*Nrx, Tin, Nchirp, Nfast]`

默认：

`[16, 16, 32, 128]`

复数 IQ 拆 real/imag，通道顺序固定：

- `channel 0`: rx1 real
- `channel 1`: rx1 imag
- `channel 2`: rx2 real
- `channel 3`: rx2 imag
- ...
- `channel 14`: rx8 real
- `channel 15`: rx8 imag

IQ 输入归一化：

`iq_norm = iq / 76.0`

标签：

- `class_label`: `[K]`
- `state_label`: `[K, Tout, 4]`
- `target_mask`: `[K]`

`state_label` 最后一维为：

`[x, y, vx, vy]`

padding/background slot：

- `class_label = 0`
- `state_label = 0`
- `target_mask = 0`

真实目标 slot：

- `class_label = gt.targetClassId`
- `target_mask = 1`

不要直接用 `gt.targetObjectIndices` 作为 Python 索引，因为 MATLAB 是 1-based。当前代码使用：

```python
is_target = gt["isTarget"]
target_idx = np.where(is_target)[0]
```

`gt.targetClassId` 是类别标签 1-4，不需要减 1。

## 标签标准化

只对真实目标 slot 标准化，padding slot 保持 0。

统计值来自当前 50 个 scene：

```text
x_mean  = 44.1808581
x_std   = 13.3377469
y_mean  = -0.0426138478
y_std   = 9.38106178
vx_mean = 6.64884038
vx_std  = 3.46447131
vy_mean = -0.273117027
vy_std  = 2.27035932
```

标准化：

```text
x_norm  = (x  - x_mean)  / x_std
y_norm  = (y  - y_mean)  / y_std
vx_norm = (vx - vx_mean) / vx_std
vy_norm = (vy - vy_mean) / vy_std
```

## 当前已实现工程

目录：

`radar_detect/train_baseline`

文件：

- `dataset.py`
- `model.py`
- `train.py`
- `utils.py`
- `config.py`
- `requirements.txt`
- `README.md`

模型：

`FmcwBaseline3DCNN`

输入：

`[B, 16, 16, 32, 128]`

输出：

- `class_logits`: `[B, 4, 5]`
- `state_pred`: `[B, 4, 8, 4]`

Loss：

分类：

- `CrossEntropyLoss`
- 对所有 slot 计算，包括 background

回归：

- `SmoothL1Loss(reduction='none')`
- 只对 `target_mask = 1` 的 slot 计算
- 回归 loss 按有效元素数归一化：

```python
valid_count = target_mask.sum() * Tout * 4
loss_state = masked_loss.sum() / max(valid_count, 1)
```

总 loss：

```python
loss = loss_cls + lambda_state * loss_state
```

默认：

`lambda_state = 1.0`

## 训练/验证划分

按 scene 划分，避免滑窗数据泄漏。

默认：

- train: 前 40 个 scene
- val: 后 10 个 scene

窗口数：

- train: 1000
- val: 250

## 运行方式

安装依赖：

```bash
python3 -m pip install -r radar_detect/train_baseline/requirements.txt
```

编译检查：

```bash
python3 -m compileall radar_detect/train_baseline
```

冒烟实验：

```bash
python radar_detect/train_baseline/train.py \
  --dataset-dir radar_detect/fmcw_traj_dataset \
  --epochs 1 \
  --batch-size 2 \
  --device cpu \
  --print-sample
```

训练结果写到：

`radar_detect/train_baseline/runs/run_YYYYmmdd_HHMMSS/metrics.jsonl`

保存：

- `best.pt`
- `last.pt`

注意：之前 `--device auto` 自动选了 `mps`，导致 loss 变成 NaN。已修复为 `auto` 优先 CUDA，否则 CPU。MPS 需要显式 `--device mps`，且要确认 finite loss。

JSONL 也已修复为严格 JSON，非有限数会转成 `null`；训练中如果出现 non-finite loss 会直接报错，并向 JSONL 写入 error 记录。

## 已验证

MATLAB v7.3 读取维度检查通过：

```text
iq  (128, 32, 8, 48) complex64
pos (6, 48, 2)
vel (6, 48, 2)
isTarget 4
targetClassId [1, 2, 4, 3, 0, 0]
```

CPU 1 epoch 可正常运行。

一次 CPU 冒烟输出：

```json
{
  "type": "epoch",
  "epoch": 1,
  "train": {
    "loss_total": 1.4552906549572944,
    "loss_cls": 1.117898376405239,
    "loss_state": 0.3373922773450613,
    "slot_accuracy": 0.513,
    "target_accuracy": 0.3403703703703704,
    "position_mae_m": 7.60404237623568,
    "velocity_mae_mps": 2.020112740331226,
    "ade_m": 11.886468309826322,
    "fde_m": 11.85369223665308
  },
  "val": {
    "loss_total": 2.3267841191291807,
    "loss_cls": 1.9001061824560166,
    "loss_state": 0.4266779453754425,
    "slot_accuracy": 0.308,
    "target_accuracy": 0.07285714285714286,
    "position_mae_m": 8.088313677651541,
    "velocity_mae_mps": 2.298096582719258,
    "ade_m": 13.16447701045445,
    "fde_m": 13.3603900882176
  }
}
```

该输出说明训练已正常跑通，但 1 epoch 下模型还很弱，val 分类精度较低是正常现象。

## 后续建议

优先继续做：

1. 多训练几个 epoch，观察 train/val loss 是否下降。
2. 增加 `--epochs 20` 或 `50`。
3. 检查是否过拟合。
4. 后续可以考虑：
   - 更强的时序模块：CNN + GRU/Transformer
   - 使用 RD map 辅助输入
   - Hungarian matching 替代固定 slot
   - 扩展到 `K=6`，同时预测干扰目标
