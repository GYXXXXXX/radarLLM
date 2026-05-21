# FMCW-MIMO 雷达目标轨迹预测仿真数据集

本数据集是一个基于 **FMCW-MIMO 雷达** 的二维连续场景仿真数据集，主要用于：

1. 从雷达 IQ 数据学习目标运动规律；
2. 基于历史观测进行目标轨迹预测；
3. 区分真实目标与干扰物；
4. 识别不同目标类别。

当前版本每个场景直接生成连续 **48 帧**。数据生成阶段不固定前 32 帧和后 16 帧的划分；若用于轨迹预测，可在训练阶段自行切分，例如：

```matlab
inputFrames = 1:32;
predFrames  = 33:48;
```

## 1. 数据集基本设定

### 1.1 二维平面场景

- 雷达位置：`[0, 0]`（二维平面原点）
- 雷达朝向：`+x` 方向
- 目标/干扰物位置表示：`[x, y]`
- 坐标含义：
  - `x`：前向距离方向
  - `y`：横向方向

### 1.2 雷达有效探测区域

雷达有效探测区域为前向扇形区域：

- 最小距离：`5 m`
- 最大距离：`100 m`
- 视场角：`120°`（即前向 `±60°`）

物体被认为可见，需同时满足：

1. 位于雷达前方（`x > 0`）；
2. 距离满足 `5 m <= range <= 100 m`；
3. 方位角满足 `-60° <= azimuth <= 60°`。

在代码中，可见性由 `isVisible(p, pos)` 判断。

## 2. 场景设置

每个场景为连续时间序列，参数如下：

- 帧率：`10 Hz`
- 帧数：`48`
- 总时长：`4.8 s`

每个场景包含：

- 真实目标数量：`1 ~ 4`
- 干扰物数量：`0 ~ 2`
- 真实目标类别数：`4`
- 干扰物类别数：`2`

## 3. 目标类别设计

数据集中共有四类真实目标：

### 3.1 `T1_slow_smooth`

慢速平滑目标，特点：

1. 运动较平滑；
2. 机动幅度较小；
3. RCS 较强；
4. 多散射中心较稳定。

信号表现：Range-Doppler 峰值较稳定。

### 3.2 `T2_uav_agile`

无人机类目标，特点：

1. 运动更灵活；
2. 存在轻微横向机动；
3. RCS 较小；
4. 具有模拟转子微多普勒特征。

信号表现：除主体散射点外，还会出现由转子模拟产生的多普勒边带。

### 3.3 `T3_pedestrian`

行人类目标，特点：

1. 运动速度较低；
2. 机动幅度较小；
3. 存在模拟步态微多普勒；
4. 散射强度中等。

用于模拟行人步态导致的微多普勒变化。

### 3.4 `T4_fast_maneuver`

快速机动目标，特点：

1. 运动速度较快；
2. 轨迹弯曲和机动更明显；
3. 散射强度较高；
4. 多普勒变化更剧烈。

用于模拟高速运动目标。

## 4. 干扰物类别设计

数据集中包含两类干扰物：

### 4.1 `I1_false_reflector`

假目标/杂波反射体，特点：

1. 类似点目标回波；
2. 相位不稳定；
3. 多普勒存在随机扰动；
4. 轨迹可能带有随机抖动。

在 Range-Doppler 图中可能表现为类似目标峰值，但运动和相位稳定性较差。

### 4.2 `I2_wideband_jammer`

宽带干扰源，特点：

1. 不服从普通 FMCW 点目标模型；
2. 产生宽带干扰信号；
3. 可能在 Range-Doppler 图中形成污染、条纹或宽区域能量。

用于模拟非目标型雷达干扰。

## 5. 多目标关系设计

同一场景中的真实目标并非完全独立随机运动，而是存在场景级关系。当前支持：

1. `formation`
2. `crossing`
3. `leader_follower`
4. `converging`

### 5.1 `formation`

编队运动：多个目标保持一定相对位置，并沿相似方向运动。

### 5.2 `crossing`

交叉运动：多个目标从不同横向位置出发，在运动过程中发生横向交叉。

### 5.3 `leader_follower`

跟随运动：多个目标沿相似方向运动，并保持前后关系。

### 5.4 `converging`

汇聚运动：多个目标从不同初始位置出发，向相近区域汇聚。

## 6. 雷达信号模型

本数据集使用 FMCW-MIMO 雷达模型生成复数 IQ 数据。

### 6.1 雷达参数

默认参数如下：

| 参数 | 值 |
| --- | --- |
| 载频 `fc` | `77 GHz` |
| 带宽 `B` | `150 MHz` |
| chirp 时长 `Tc` | `25 us` |
| ADC 采样率 `Fs` | `10 MHz` |
| fast-time 采样点数 `Nfast` | `128` |
| 每帧 chirp 数 `Nchirp` | `32` |
| 接收阵元数 `Nrx` | `8` |
| 帧数 `Nframes` | `48` |

每个场景的 IQ 数据维度：

```text
iq: [128, 32, 8, 48]
```

维度含义：

1. 第 1 维：fast-time 采样点；
2. 第 2 维：chirp 序列；
3. 第 3 维：接收天线/虚拟阵元；
4. 第 4 维：时间帧。

### 6.2 为什么使用 MIMO

单通道 FMCW 雷达通常可较好获得：

1. 距离（range）；
2. 径向速度（radial velocity）。

但单接收通道二维角度信息不足，难以恢复二维平面轨迹。因此本数据集采用 `Nrx = 8` 接收阵元，目标角度信息编码在接收通道间的相位差中，使模型可从 IQ 中学习：

- `range`
- `radial velocity`
- `azimuth`
- `2D trajectory`

## 7. 数据文件结构

生成后目录结构如下：

```text
fmcw_traj_dataset/
├── index.csv
├── scene_000001.mat
├── scene_000002.mat
├── scene_000003.mat
├── ...
└── visualization/
    ├── overview_page_001.png
    ├── overview_page_002.png
    ├── scene_000001_detail.png
    ├── scene_000002_detail.png
    └── ...
```

- `index.csv`：保存所有场景索引信息；
- `scene_xxxxxx.mat`：每个文件保存一个完整雷达场景。

## 8. 单个场景文件内容

每个 `scene_xxxxxx.mat` 包含：

- `iq`
- `rdMap`
- `gt`
- `meta`
- `p`

### 8.1 `iq`

原始复数 IQ 数据，维度：

```text
[Nfast, Nchirp, Nrx, Nframes]
```

默认：

```text
[128, 32, 8, 48]
```

字段含义：

- `Nfast`：每个 chirp 的 fast-time 采样点
- `Nchirp`：每帧 chirp 数
- `Nrx`：接收阵元数量
- `Nframes`：场景帧数

示例：

```matlab
load('fmcw_traj_dataset/scene_000001.mat');
size(iq)
```

输出类似：

```text
ans =
   128    32     8    48
```

### 8.2 `rdMap`

`rdMap` 是 IQ 经二维 FFT 得到的 Range-Doppler 图，维度：

```text
[NrangeFFT, NdopplerFFT, Nframes]
```

默认：

```text
[128, 32, 48]
```

维度含义：

1. 第 1 维：距离 bin；
2. 第 2 维：多普勒/径向速度 bin；
3. 第 3 维：时间帧。

说明：

- `rdMap` 已对接收天线维度做能量融合；
- 因此角度信息弱于原始 `iq`；
- 若任务为二维轨迹预测，优先使用 `iq` 而非仅使用 `rdMap`。

### 8.3 `gt`

`gt` 是真值信息，包含轨迹、速度、类别、可见性等。主要字段：

```text
gt.numObjects
gt.objId
gt.isTarget
gt.targetClassId
gt.interferenceClassId
gt.name
gt.pos
gt.vel
gt.range
gt.azimuth
gt.radialVel
gt.visible
gt.targetObjectIndices
gt.interferenceObjectIndices
```

关键字段说明：

- `gt.numObjects`：场景内全部物体数（真实目标 + 干扰物）
- `gt.isTarget`：是否为真实目标（`true` 为目标，`false` 为干扰物）
- `gt.targetClassId`：真实目标类别编号（干扰物该字段为 `0`）
  - `1`: `T1_slow_smooth`
  - `2`: `T2_uav_agile`
  - `3`: `T3_pedestrian`
  - `4`: `T4_fast_maneuver`
- `gt.interferenceClassId`：干扰物类别编号（真实目标该字段为 `0`）
  - `1`: `I1_false_reflector`
  - `2`: `I2_wideband_jammer`
- `gt.pos`：二维位置真值，维度 `[Nobject, Nframes, 2]`（第三维为 `x,y`）
- `gt.vel`：二维速度真值，维度 `[Nobject, Nframes, 2]`（第三维为 `vx,vy`）
- `gt.range`：距离，维度 `[Nobject, Nframes]`，计算 `sqrt(x^2 + y^2)`
- `gt.azimuth`：方位角（弧度），计算 `atan2(y, x)`
- `gt.radialVel`：径向速度（`m/s`），即速度在径向方向的投影
- `gt.visible`：可见性 mask，维度 `[Nobject, Nframes]`

示例：读取第一个目标的 48 帧位置

```matlab
targetIdx = gt.targetObjectIndices(1);
xy = squeeze(gt.pos(targetIdx, :, :));  % xy: [48, 2]
```

在当前 clean 版本中：

- 真实目标在 48 帧内应满足 `visible = true`
- 干扰物不强制全程可见

### 8.4 `meta`

`meta` 保存场景元信息，常见字段：

```text
meta.sceneId
meta.nTargets
meta.nInterferers
meta.relationId
meta.relationName
meta.targetClassIds
meta.interferenceClassIds
meta.requireTargetVisibleAllFrames
meta.allTargetsVisibleAllFrames
```

其中 `meta.relationName` 表示场景关系类型，例如：

- `formation`
- `crossing`
- `leader_follower`
- `converging`

### 8.5 `p`

`p` 保存数据生成与雷达参数，常见字段：

```text
p.Nframes
p.frameRate
p.dt
p.rMin
p.rMax
p.fovDeg
p.Nfast
p.Nchirp
p.Nrx
p.fc
p.B
p.Tc
p.Fs
p.rangeAxis
p.velocityAxis
```

## 9. `index.csv` 说明

`index.csv` 保存全部场景索引，字段如下：

| 字段 | 含义 |
| --- | --- |
| `scene_id` | 场景编号 |
| `file` | 对应 `.mat` 文件名 |
| `n_targets` | 当前场景真实目标数量 |
| `n_interferers` | 当前场景干扰物数量 |
| `relation` | 当前场景目标运动关系 |
| `all_targets_visible` | 所有真实目标是否 48 帧全程可见 |

## 10. 可视化结果说明

运行：

```matlab
visualize_all_fmcw_scenes('fmcw_traj_dataset', 16, true, false);
```

会在 `fmcw_traj_dataset/visualization/` 下生成：

- `overview_page_xxx.png`
- `scene_xxxxxx_detail.png`

### 10.1 `overview_page_xxx.png`

分页展示所有场景完整轨迹，用于快速检查：

1. 目标数量是否合理；
2. 干扰物数量是否合理；
3. 目标是否在雷达视场内；
4. 目标关系是否合理；
5. 轨迹是否连续；
6. 是否存在明显异常轨迹。

### 10.2 `scene_xxxxxx_detail.png`

单场景详细图通常包括：

1. 左侧：完整 48 帧二维轨迹；
2. 右侧：第 1、24、48 帧的 Range-Doppler 图。

轨迹图标记说明：

- 绿色圆点：起点
- 黑色方块：终点
- 实线：真实目标
- 虚线/叉号：干扰物
- 虚线扇形：雷达有效视场

## 11. 轨迹预测任务使用方式

虽然生成阶段只保存完整 48 帧，但可在训练阶段构造预测任务。例如使用前 32 帧预测后 16 帧：

```matlab
load('fmcw_traj_dataset/scene_000001.mat');

inputFrames = 1:32;
predFrames  = 33:48;
targetIdx = gt.targetObjectIndices;

X_iq = iq(:, :, :, inputFrames);
X_rd = rdMap(:, :, inputFrames);
Y_pos = gt.pos(targetIdx, predFrames, :);
Y_vel = gt.vel(targetIdx, predFrames, :);
```

对应维度：

- `X_iq`: `[128, 32, 8, 32]`
- `X_rd`: `[128, 32, 32]`
- `Y_pos`: `[Ntarget, 16, 2]`
- `Y_vel`: `[Ntarget, 16, 2]`

## 12. 可支持的研究任务

### 12.1 从 IQ 数据预测未来轨迹

- 输入：前若干帧 IQ
- 输出：未来若干帧目标二维位置
- 示例：
  - 输入：`iq(:, :, :, 1:32)`
  - 输出：`gt.pos(targetIdx, 33:48, :)`

### 12.2 从 RD 图预测未来轨迹

- 输入：前若干帧 Range-Doppler 图
- 输出：未来若干帧目标二维位置
- 注意：RD 图角度信息较弱，二维轨迹预测通常不如多通道 IQ

### 12.3 目标/干扰物识别

- 输入：IQ 或 RD 图特征
- 输出：每个物体是真实目标还是干扰物
- 标签：`gt.isTarget`

### 12.4 目标类别识别

- 输入：目标对应 IQ/RD 特征
- 输出：目标类别编号
- 标签：`gt.targetClassId`
- 类别：
  - `1`: `T1_slow_smooth`
  - `2`: `T2_uav_agile`
  - `3`: `T3_pedestrian`
  - `4`: `T4_fast_maneuver`

### 12.5 多目标关系建模

- 输入：同一场景中多个目标雷达观测
- 输出：多目标未来轨迹
- 关系标签：`meta.relationName`
  - `formation`
  - `crossing`
  - `leader_follower`
  - `converging`

## 13. 当前版本设计原则

当前版本是一个 clean dataset，原则包括：

1. 所有真实目标在 48 帧内全程可见；
2. 真实目标轨迹连续、平滑；
3. 目标之间存在明确关系；
4. 干扰物可不全程可见；
5. 干扰物与真实目标在信号结构上有差异；
6. 生成阶段不绑定固定训练切分方式。

设计收益：

1. IQ 与轨迹标签严格对应；
2. 避免“目标不可见但仍强制预测”的矛盾；
3. 训练任务定义更清晰；
4. 支持灵活预测窗口配置。

## 14. 当前版本限制

当前仍为简化仿真数据集，与真实雷达环境相比限制包括：

1. 无复杂地物杂波模型；
2. 无真实多径传播；
3. 无遮挡与目标消失/重现；
4. 目标形状为简化多散射中心模型；
5. 干扰模型为简化模拟；
6. 暂未加入 CFAR 检测点、漏检、虚警等后处理结果。

可扩展方向：

1. 加入地面杂波；
2. 加入多径反射；
3. 加入目标遮挡；
4. 加入 CFAR 检测点；
5. 加入漏检与虚警；
6. 加入角度-距离-多普勒 RAD cube；
7. 加入点云格式输出；
8. 加入不同天气或噪声强度设置。

## 15. 推荐使用流程

1. 生成数据：

   ```matlab
   generate_fmcw_traj_dataset
   ```

   结果保存于 `fmcw_traj_dataset/`。

2. 可视化检查：

   ```matlab
   visualize_all_fmcw_scenes('fmcw_traj_dataset', 16, true, false);
   ```

   检查 `fmcw_traj_dataset/visualization/` 下图片。

3. 读取单场景：

   ```matlab
   load('fmcw_traj_dataset/scene_000001.mat');
   size(iq)
   size(rdMap)
   gt.numObjects
   meta
   ```

4. 构造训练样本（例如前 32 帧预测后 16 帧）：

   ```matlab
   inputFrames = 1:32;
   predFrames  = 33:48;
   targetIdx = gt.targetObjectIndices;
   X = iq(:, :, :, inputFrames);
   Y = gt.pos(targetIdx, predFrames, :);
   ```

## 16. 注意事项

- `iq` 为复数数据，训练深度学习模型时通常拆成实部与虚部：

  ```matlab
  X_real = real(X);
  X_imag = imag(X);
  ```

- 也可将实部/虚部拼接为两个通道；
- `rdMap` 为对数功率谱（近似 dB 表示），适合图像输入；
- 做二维轨迹预测时建议优先使用多通道 `iq`（角度信息在接收阵元相位差中）；
- clean 版本真实目标应 48 帧全可见，若发现：

  ```matlab
  any(~gt.visible(gt.targetObjectIndices, :), 'all')
  ```

  结果为 `true`，说明场景生成可能异常，需检查生成代码；
- 干扰物不要求全程可见，属于正常现象。

## 17. 示例检查代码

检查所有场景中的真实目标是否全程可见：

```matlab
datasetDir = 'fmcw_traj_dataset';
indexTable = readtable(fullfile(datasetDir, 'index.csv'));

for i = 1:height(indexTable)
    sceneFile = fullfile(datasetDir, indexTable.file{i});
    load(sceneFile, 'gt');

    targetIdx = gt.targetObjectIndices;
    flag = all(gt.visible(targetIdx, :), 'all');

    fprintf('Scene %d: all targets visible = %d\n', i, flag);
end
```

读取第一个场景并绘制目标轨迹：

```matlab
load('fmcw_traj_dataset/scene_000001.mat');

figure;
hold on;
grid on;
axis equal;

for k = gt.targetObjectIndices(:)'
    xy = squeeze(gt.pos(k, :, :));
    plot(xy(:, 1), xy(:, 2), '-o');
end

xlabel('x / m');
ylabel('y / m');
title('Target trajectories');
```

## 18. 总结

本数据集是面向雷达目标轨迹预测的 FMCW-MIMO 仿真数据集，核心特点：

1. 每个场景包含连续 48 帧；
2. 使用 FMCW-MIMO 复数 IQ 数据；
3. 提供 Range-Doppler 图；
4. 提供二维轨迹真值；
5. 包含 4 类真实目标；
6. 包含 2 类干扰物；
7. 支持多目标关系场景；
8. clean 版本保证真实目标 48 帧全程可见；
9. 训练阶段可灵活切分输入/预测帧。

适合作为雷达信号理解、目标识别、多目标轨迹预测和雷达场景建模的基础仿真数据集。
