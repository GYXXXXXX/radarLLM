# 认知雷达闭环仿真可视化系统功能讲解报告

## 1. 系统定位

本系统面向“海上认知雷达闭环”课题演示，目标是在同一个可视化界面中展示以下链路：

```text
海上目标与平台运动仿真
-> 三部雷达独立观测
-> 多雷达观测融合为统一世界坐标轨迹
-> TrackIntentTransformer 进行轨迹预测、意图识别、威胁判断
-> LLM 输出任务级态势建议
-> 约束控制器生成可执行雷达控制参数
-> WebSocket 推送前端实时可视化
```

当前系统不是纯静态页面，而是采用 Python 后端实时仿真服务驱动前端。前端主要负责展示，后端负责闭环状态演化、Transformer 推理、LLM 决策与雷达控制参数生成。

系统当前实现位置：

```text
cognitive_radar_loop/
├── index.html
├── app.js
├── styles.css
├── run_backend.py
└── backend/
    ├── realtime_loop.py
    └── test_qwen_llm.py
```

使用的模型权重：

```text
detect_intention/train_intention/runs/run_20260604_005306/best_by_intent.pt
```

该权重当前被后端识别为：

```text
model_type = track_intent_transformer
input_mode = track
Tin = 24
Tout = 8
```

因此，当前闭环后端调用的是轨迹输入模型，不是原始 IQ 输入模型。

## 2. 场景组成

### 2.1 平台与目标

系统中包含五类核心实体：

| 实体 | 标识 | 颜色/角色 | 功能 |
|---|---|---|---|
| 岸基雷达 | `shore` | 黄色 | 固定广域发现雷达，提供持续监视与目标指示 |
| 己方舰船 01 | `V-01` / `v01` | 绿色 | 移动平台，搭载舰 01 雷达，近距精跟 |
| 己方舰船 02 | `V-02` / `v02` | 青色 | 移动平台，搭载舰 02 雷达，从另一观测方位协同确认 |
| 红色目标 | `T-01` | 红色 | 动态目标，可被三雷达闭环跟踪 |
| 蓝色目标 | `T-02` | 蓝色 | 动态目标，可与 `T-01` 切换跟踪 |

### 2.2 运动路线逻辑

后端在 `ROUTES` 中定义了舰船与目标的循环航迹点。每个实体沿一组二维航路点运动：

| 实体 | 周期 | 说明 |
---|---:|---|
| `V-01` | 92 s | 从左下至中部/右上区域机动，形成近距跟踪视角 |
| `V-02` | 96 s | 从右下至右上区域机动，形成交叉观测视角 |
| `T-01` | 110 s | 在右上、中心、左下之间变化，模拟接近/盘旋/撤离态势 |
| `T-02` | 120 s | 在上方海域横向机动，模拟另一类目标轨迹 |

后端通过 `route_position(route, t)` 按时间插值计算当前位置，并用前后位置差估计速度与航向。舰载雷达原点直接绑定到对应舰船位置，因此 `V-01`、`V-02` 移动时，舰载雷达波束原点也随平台运动。

## 3. 系统总体架构

### 3.1 后端闭环服务

后端入口为：

```text
cognitive_radar_loop/run_backend.py
```

核心逻辑位于：

```text
cognitive_radar_loop/backend/realtime_loop.py
```

后端服务同时承担两项职责：

1. 静态页面服务：向浏览器提供 `index.html`、`app.js`、`styles.css`。
2. WebSocket 服务：通过 `/ws` 持续向前端推送闭环快照。

推荐启动方式：

```powershell
.\start_qwen_radarbackend.ps1
```

或直接运行：

```powershell
conda run --no-capture-output -n radar python cognitive_radar_loop/run_backend.py --port 5177
```

前端访问：

```text
http://localhost:5177
```

### 3.2 前端可视化

前端文件为：

```text
cognitive_radar_loop/index.html
cognitive_radar_loop/app.js
cognitive_radar_loop/styles.css
```

前端主要负责：

- 显示海域二维地图；
- 显示岸基雷达、两艘舰船、两个目标；
- 显示三部雷达波束、距离门、航迹、预测轨迹；
- 显示选中雷达的数据面板；
- 通过按钮切换观察雷达和跟踪目标；
- 通过 WebSocket 向后端发送控制命令；
- 在后端断开时显示 `DISCONNECTED` 状态。

目前前端设置为 `backendRequired: true`。也就是说，系统设计上要求连接后端闭环服务；如果后端未连接，前端不再继续本地假仿真，以避免“静态页面”和“真实后端页面”之间概念混淆。

## 4. 后端闭环运行流程

每个仿真周期由 `ClosedLoopEngine.step(dt)` 推进。核心流程如下：

```text
1. 根据仿真速度缩放 dt
2. 更新时间 sim_time 与周期 cycle
3. 更新 V-01 / V-02 / T-01 / T-02 的位置、速度、航向
4. 三部雷达分别对 T-01 / T-02 生成观测量
5. 将每个雷达的局部观测转换为统一世界坐标估计
6. 对同一目标的多雷达观测进行置信度加权融合
7. 更新每个目标的 track history
8. 周期性调用 TrackIntentTransformer
9. 周期性调用 LLM 或规则 fallback
10. 约束控制器计算三部雷达的波束参数
11. 生成 snapshot
12. 通过 WebSocket 推送给前端
```

其中 Transformer 推理间隔约为 0.35 s 仿真时间，LLM 决策间隔约为 6.0 s 仿真时间。

## 5. 三部雷达如何跟踪目标

### 5.1 观测生成

每部雷达对每个目标生成一组观测量：

```text
range
bearing
localBearing
rangeRate
doppler
confidence
estimated x/y
```

观测不是简单锁死在目标上，而是受以下因素影响：

| 因素 | 作用 |
|---|---|
| 目标与雷达真实距离 | 距离越远，观测置信度降低 |
| 目标是否落在波束中心附近 | 波束误差越小，置信度越高 |
| 目标是否为当前选中跟踪目标 | 选中目标获得更高观测权重 |
| `uncertainty` 感知扰动 | 扰动越大，测距/测角抖动越大 |
| `clutter` 海杂波 | 会降低显示 SNR |

观测函数的关键逻辑可以概括为：

```text
true_range  = distance(radar, target)
true_bearing = bearing(radar, target)
beam_error = |radar.beam_azimuth - true_bearing|
beam_score = 1 - beam_error / beam_width
range_score = 1 - true_range / max_range
confidence = base + selected_bonus + beam_score + range_score - uncertainty_penalty
```

因此，波束越对准目标、距离越合适、扰动越低，置信度越高。

### 5.2 局部观测到世界坐标

岸基雷达是固定坐标系，舰载雷达随舰船运动。为了让静态雷达模型和动态舰载雷达观测能共同使用，后端将所有雷达观测统一转换到世界坐标：

```text
mx = radar.x + cos(measured_bearing) * measured_range
my = radar.y + sin(measured_bearing) * measured_range
```

这样不论观测来自岸基、`V-01` 还是 `V-02`，最终都会进入统一的目标状态：

```text
[x, y, vx, vy]
```

这也是动态雷达复用 track 模型的关键：模型不直接吃“雷达自身坐标系下的方位距离”，而是吃统一世界坐标下的目标轨迹。

### 5.3 多雷达融合

同一目标在同一周期会得到三部雷达的观测。后端使用置信度加权平均得到融合位置：

```text
fused_x = sum(measured_x * confidence) / sum(confidence)
fused_y = sum(measured_y * confidence) / sum(confidence)
```

随后使用简化的 alpha-beta 滤波思想更新轨迹：

```text
pred_x = track.x + track.vx * dt
residual_x = fused_x - pred_x
track.x  = pred_x + alpha * residual_x
track.vx = track.vx + beta * residual_x / dt
```

当前 `alpha = 0.42`，`beta = 0.12`。这不是完整 EKF/UKF，但已经能表现“预测-校正”的跟踪闭环思想。

## 6. Transformer 感知与轨迹预测

### 6.1 两类模型的区别

`detect_intention/train_intention/model.py` 中存在两个模型：

| 模型 | 输入 | 用途 |
|---|---|---|
| `FmcwIntentTransformer` | `[B, C, T, H, W]` IQ / Range-Doppler 类张量 | 直接从 FMCW/IQ 时序中提取空间-时间特征 |
| `TrackIntentTransformer` | `[B, K, Tin, 4]` 目标轨迹状态 | 从已检测/融合的目标轨迹中预测未来轨迹、意图、威胁 |

当前 `best_by_intent.pt` 是 `TrackIntentTransformer`，因此后端使用的是第二类模型。

### 6.2 模型输入

模型输入张量形式：

```text
[B, K, Tin, 4]
```

各维含义：

| 维度 | 含义 |
|---|---|
| `B` | batch size，在线服务中为 1 |
| `K` | 最大目标槽位数，当前为 4 |
| `Tin` | 输入历史帧数，当前为 24 |
| `4` | 每帧状态 `[x, y, vx, vy]` |

当前系统实际使用 `T-01`、`T-02` 两个目标，分别放入前两个目标槽位。剩余槽位保留。

输入状态会使用训练配置中的 `STATE_MEAN`、`STATE_STD` 标准化：

```text
normalized = (state - STATE_MEAN) / STATE_STD
```

这样在线输入分布尽量与训练阶段保持一致。

### 6.3 模型输出

模型输出包括：

| 输出字段 | 含义 |
|---|---|
| `state_pred` | 未来 `Tout=8` 帧的预测轨迹，每帧为 `[x, y, vx, vy]` |
| `intent_logits` | 目标意图分类 logits |
| `threat_logits` | 威胁等级分类 logits |
| `objectness_logits` | 目标存在性评分 |
| `target_class_logits` | 目标类别分类 logits，当前前端没有重点展示 |
| `slot_features` | 目标槽位的 Transformer 表征 |

后端将 logits 转为概率：

```text
intent_prob = softmax(intent_logits)
threat_prob = softmax(threat_logits)
objectness = sigmoid(objectness_logits)
```

再映射为可读标签：

| 意图编号 | 标签 | 含义 |
|---:|---|---|
| 1 | `benign_transit` | 常规通过 |
| 2 | `approach` | 接近 |
| 3 | `retreat` | 远离/撤离 |
| 4 | `loiter_patrol` | 盘旋/巡逻 |
| 5 | `intercept` | 拦截/高关注机动 |

| 威胁编号 | 标签 | 含义 |
|---:|---|---|
| 1 | `low` | 低威胁 |
| 2 | `guarded` | 警戒 |
| 3 | `elevated` | 升高 |
| 4 | `high` | 高威胁 |

### 6.4 历史不足时的处理

模型需要至少 24 帧历史轨迹。如果系统刚启动、轨迹历史不足，后端会使用常速度预测 fallback：

```text
future_x = x + vx * dt
future_y = y + vy * dt
```

同时基于未来距离变化粗略判断 `approach`、`retreat` 或 `loiter_patrol`。等历史帧足够后，才切换到真正的 `TrackIntentTransformer` 推理。

## 7. LLM 态势推理层

### 7.1 LLM 的位置

LLM 位于 Transformer 之后、约束控制器之前：

```text
TrackIntentTransformer
-> LLM mission advisor
-> ConstraintController
```

这意味着 LLM 不直接输出波束角、波束宽度等底层控制量，而是输出任务级建议。底层可执行参数由约束控制器负责生成。

这样设计的原因是：

- LLM 适合做态势解释、任务级策略与多平台协同建议；
- 控制器适合做数值约束、平滑转向、参数限幅和安全边界；
- 两者解耦可以避免 LLM 直接生成不稳定或越界的控制参数。

### 7.2 LLM 输入

后端传给 LLM 的结构化状态包括：

| 字段 | 含义 |
|---|---|
| `target_id` | 当前目标编号，例如 `T-01` |
| `state` | 目标当前融合状态 `[x, y, vx, vy]` |
| `intent` | Transformer 判断的意图 |
| `threat` | Transformer 判断的威胁等级 |
| `intent_confidence` | 意图置信度 |
| `threat_confidence` | 威胁置信度 |
| `predicted_range_delta` | 预测末端相对岸基雷达的距离变化 |
| `predicted_track` | Transformer 预测的未来轨迹 |
| `selected_target` | 当前前端/后端选中的跟踪目标 |

其中 `predicted_range_delta < 0` 表示目标未来可能更接近岸基区域，`predicted_range_delta > 0` 表示目标未来可能远离。

### 7.3 LLM 输出

LLM 被要求输出严格 JSON，主要字段为：

| 字段 | 含义 |
|---|---|
| `target_id` | 目标编号 |
| `final_intent` | LLM 综合判断后的最终意图 |
| `priority` | 任务优先级，数值越高越紧急 |
| `action` | 任务级动作 |
| `reason` | 简短态势解释 |
| `radar_guidance` | 对岸基、舰 01、舰 02 雷达的任务建议 |
| `platform_guidance` | 对舰船平台运动/站位的建议 |

允许的 `action` 包括：

| action | 控制含义 |
|---|---|
| `monitor` | 常规监视，保持广域跟踪 |
| `increase_tracking_rate` | 提高跟踪频率，收窄波束并增加驻留 |
| `classify_and_shadow` | 分类识别并伴随监视 |
| `alert_and_allocate_tracker` | 高优先级告警并分配精跟资源 |

### 7.4 异步调用与 fallback

后端不会让 LLM 调用阻塞整个可视化循环。当前实现采用线程池异步调用：

```text
ThreadPoolExecutor(max_workers=2)
```

LLM 调用过程中，系统会暂时使用规则决策 fallback，页面上可能看到类似：

```text
waiting for decision
llm_pending
```

这表示 LLM 请求仍在进行或当前还没有返回新决策，并不等于系统停止工作。后端仍然会用缓存决策或规则决策继续生成控制参数。

如果 LLM 超时或返回格式异常，后端会自动进入规则 fallback，并在终端输出类似：

```text
[LLM] fallback for T-01: TimeoutError: ...
```

如果调用成功，会输出类似：

```text
[LLM] T-01 action=increase_tracking_rate intent=approach
```

### 7.5 当前 Qwen 接入方式

当前系统支持 OpenAI-compatible 接口。使用阿里云百炼/通义千问时，典型配置为：

```powershell
$env:LLM_PROVIDER="openai_compatible"
$env:LLM_API_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
$env:LLM_MODEL="qwen3.6-flash"
$env:DASHSCOPE_API_KEY="你的 API Key"
```

系统也支持使用 `LLM_API_KEY`。如果二者都存在，后端优先读取 `LLM_API_KEY`，否则读取 `DASHSCOPE_API_KEY`。

## 8. 约束控制器与波束控制

### 8.1 控制器输入

约束控制器 `ConstraintController.update()` 的输入包括：

| 输入 | 来源 |
|---|---|
| 雷达当前位置 | 岸基固定坐标或舰船当前位置 |
| 当前融合目标轨迹 | 多雷达融合结果 |
| Transformer 预测轨迹 | `track.prediction` |
| LLM 任务动作 | `track.llm_decision.action` |
| 意图置信度 | `track.intent_confidence` |
| 当前仿真步长 | `dt` |

控制器优先使用预测轨迹中的第 3 个未来点作为指向点：

```text
prediction = track.prediction[2]
```

因此波束不是只盯当前目标位置，而是带有一定前视量，体现“提前预知目标轨迹移动”的控制思想。

### 8.2 可执行控制参数

控制器生成以下参数：

| 字段 | 含义 |
|---|---|
| `beam_azimuth` | 雷达当前实际波束方位角 |
| `beam_width` | 当前实际波束宽度 |
| `dwell_ms` | 当前驻留时间 |
| `range_gate` | 当前距离门中心/范围显示值 |
| `desiredAzimuth` | 控制器期望方位角 |
| `desiredWidth` | 控制器期望波束宽度 |
| `desiredDwell` | 控制器期望驻留时间 |
| `mode` | 当前控制模式，对应 LLM action |
| `constraint` | 控制约束类型，当前为 `bounded_slew` |
| `platformGuidance` | LLM 给平台的任务建议 |

### 8.3 不同任务动作对应的控制策略

| LLM action | 期望波束宽度 | 期望驻留时间 | 含义 |
|---|---:|---:|---|
| `monitor` | 约 30° | 使用雷达基础驻留时间 | 常规广域监视 |
| `classify_and_shadow` | 约 20° | 约 135 ms | 保持伴随、增强分类稳定性 |
| `increase_tracking_rate` | 约 14° | 约 165 ms | 提高跟踪精度和刷新强度 |
| `alert_and_allocate_tracker` | 约 10° | 约 210 ms | 高优先级精跟 |

当模型置信度较低时，控制器会适当增大波束宽度并增加驻留时间，以提升重新捕获与稳定观测能力。

### 8.4 平滑转向

为避免切换目标时波束扇形突然跳变，控制器采用方位角速率限制：

```text
max_step = slew_rate * dt
beam_azimuth = slew_angle(current, desired, max_step)
```

同时波束宽度、驻留时间和距离门采用线性平滑：

```text
beam_width = lerp(current_width, desired_width, 0.18)
dwell_ms = lerp(current_dwell, desired_dwell, 0.18)
range_gate = lerp(current_gate, desired_gate, 0.2)
```

因此，在用户从 `T-01` 切换到 `T-02` 时，三部雷达会逐渐转向新目标，而不是瞬间跳变。

### 8.5 岸基与舰载雷达差异

控制器对岸基和舰载雷达采用不同参数边界：

| 雷达 | 波束宽度边界 | 转向速率逻辑 |
|---|---|---|
| 岸基雷达 | 12° - 62° | 较稳健，偏广域监视 |
| 舰载雷达 | 8° - 48° | 更快，偏近距精跟 |

舰载雷达的原点随舰船移动，因此同一个目标对 `V-01`、`V-02` 的方位角不同。前端上看到三部雷达波束方向不完全一致，是因为它们来自不同观测点，并不是同一个静态扇形的复制。

## 9. 前端界面字段解释

### 9.1 顶部状态栏

| 字段 | 含义 |
|---|---|
| `BACKEND` / `DISCONNECTED` | 是否连接到 Python 后端 WebSocket |
| `后端 track_intent_transformer` | 当前后端加载的模型类型 |
| `LLM qwen3.6-flash` | 当前配置的 LLM 模型；如果未配置则显示 `rule` |
| `周期` | 后端闭环推进次数 `cycle` |
| `时标` | 当前仿真时间 `simTime` |
| `闭环` | 前端显示的闭环刷新频率估计 |

### 9.2 左侧控制面板

| 控件 | 含义 |
|---|---|
| `岸基 / 舰01 / 舰02` | 只切换右侧数据面板和 IQ 视图的观察对象，不会关闭其他雷达 |
| `T-01 / T-02` | 切换三部雷达当前协同跟踪的目标 |
| `暂停 / 运行` | 控制仿真是否推进 |
| `复位` | 后端闭环状态重新初始化 |
| `仿真速度` | 调整后端仿真时间缩放 |
| `海杂波` | 影响 SNR 与显示噪声强度 |
| `感知扰动` | 影响测距/测角误差与置信度 |
| `波束` | 显示/隐藏三部雷达扇形波束 |
| `航迹` | 显示/隐藏平台和目标运动轨迹 |
| `距离门` | 显示/隐藏雷达距离门 |

### 9.3 地图区域

地图区域展示：

| 图形元素 | 含义 |
|---|---|
| 黄色固定图标 | 岸基雷达 |
| 绿色船形图标 | `V-01` 与舰 01 雷达 |
| 青色船形图标 | `V-02` 与舰 02 雷达 |
| 红色目标 | `T-01` |
| 蓝色目标 | `T-02` |
| 扇形区域 | 雷达当前波束覆盖区域 |
| 虚线圆/弧 | 距离门或探测范围辅助显示 |
| 实线轨迹 | 历史航迹 |
| 预测轨迹 | Transformer 输出的未来轨迹 |
| 方位连线 | 雷达对目标的观测/指示关系 |

### 9.4 右侧雷达数据面板

右侧面板显示当前选中的雷达数据。注意，切换 `岸基 / 舰01 / 舰02` 只是切换查看对象，不会停止其他雷达工作。

| 字段 | 后端/前端来源 | 含义 |
|---|---|---|
| `方位` | 后端 `beamAzimuth` | 当前雷达实际波束方位角 |
| `距离` | 后端 `perception.range` | 当前雷达对选中目标的测距结果 |
| `置信度` | 后端 `perception.confidence` | 当前雷达观测可信度 |
| `SNR` | 后端 `snr` | 信噪比显示值，受置信度和海杂波影响 |
| `波束宽度` | 后端 `beamWidth` | 当前雷达扇形宽度 |
| `驻留时间` | 后端 `dwell` | 当前波束驻留时间 |
| `估计方位` | 后端 `perception.bearing` | 观测得到的世界方位 |
| `估计距离` | 后端 `perception.range` | 观测得到的距离 |
| `多普勒` | 后端 `perception.doppler` | 由径向速度估算的多普勒/速度显示量 |
| `一致性` | 后端 `fusionScore` | 当前选中目标的意图/威胁综合置信度 |
| `意图` | 后端 `reasoning.intent` | Transformer 判断的意图标签 |
| `威胁` | 后端 `reasoning.threat` | Transformer 判断的威胁等级 |
| `态势说明` | 后端 `reasoning.narrative` | LLM 或 fallback 生成的解释 |
| `控制建议` | 后端 `reasoning.action` + `controller.constraint` | 任务级动作与控制约束 |

### 9.5 IQ / Range-Doppler 面板说明

当前前端中的 `IQ / Range-Doppler` 面板是浏览器侧的仿真热力图，用于可视化表达不同雷达视角下的距离-多普勒响应，不是后端真实传输的 MATLAB IQ 原始矩阵。

当前后端真实推送的是：

```text
range / bearing / doppler / confidence / track / prediction / reasoning / controller
```

如果后续要接入真实 MATLAB IQ 或 `.mat` 序列，需要在后端增加 IQ 数据读取、Range-Doppler 处理与 WebSocket 二进制/压缩推送，或在后端完成图像化后推送给前端。

### 9.6 底部状态卡片

底部三个雷达卡片分别显示：

| 字段 | 含义 |
|---|---|
| 雷达名称 | 岸基雷达、舰 01 雷达、舰 02 雷达 |
| `目标 T-01/T-02` | 当前协同跟踪目标 |
| `xxx° / yy°` | 当前波束方位角 / 波束宽度 |
| 置信度数字 | 当前雷达对目标观测的可信度 |
| 彩色进度条 | 置信度可视化 |

事件流显示最近的关键事件，例如：

```text
target handoff -> T-02
backend loop reset
```

## 10. WebSocket 数据快照结构

后端每次推送的 snapshot 大致包含：

```json
{
  "type": "snapshot",
  "backend": true,
  "simTime": 12.3,
  "cycle": 123,
  "speed": 1.0,
  "clutter": 0.26,
  "uncertainty": 0.22,
  "selectedTargetId": "T-01",
  "model": {},
  "llm": {},
  "fusionScore": 0.86,
  "entities": {},
  "radars": {},
  "tracks": {},
  "events": []
}
```

关键结构解释如下。

### 10.1 `entities`

```text
entities.ships.v01
entities.ships.v02
entities.targets.T-01
entities.targets.T-02
```

每个实体包含：

| 字段 | 含义 |
|---|---|
| `id` | 实体编号 |
| `x` / `y` | 世界坐标位置 |
| `vx` / `vy` | 世界坐标速度 |
| `heading` | 航向角 |
| `trail` | 历史轨迹点 |

### 10.2 `radars`

```text
radars.shore
radars.v01
radars.v02
```

每部雷达包含：

| 字段 | 含义 |
|---|---|
| `origin.x` / `origin.y` | 雷达原点 |
| `heading` | 雷达平台航向 |
| `beamAzimuth` | 当前波束方位角 |
| `beamWidth` | 当前波束宽度 |
| `dwell` | 驻留时间 |
| `rangeGate` | 距离门 |
| `snr` | 信噪比显示值 |
| `perception` | 当前观测结果 |
| `reasoning` | 当前态势推理结果 |
| `controller` | 当前控制器参数 |

### 10.3 `tracks`

```text
tracks.T-01
tracks.T-02
```

每个目标轨迹包含：

| 字段 | 含义 |
|---|---|
| `state` | 当前融合状态 `[x, y, vx, vy]` |
| `history` | 输入 Transformer 的历史轨迹窗口 |
| `prediction` | Transformer 或常速度 fallback 的未来轨迹 |
| `intent` | 当前意图标签 |
| `threat` | 当前威胁标签 |
| `intent_confidence` | 意图置信度 |
| `threat_confidence` | 威胁置信度 |
| `objectness` | 目标存在性评分 |
| `llm_decision` | LLM 或 fallback 决策结果 |

## 11. 当前系统实现边界

为了在汇报中保持表述准确，需要区分“已实现”和“仍是仿真/示意”的部分。

### 11.1 已实现

当前已经实现：

- Python 后端实时闭环服务；
- WebSocket 驱动前端；
- 岸基、`V-01`、`V-02` 三雷达同时工作；
- 舰载雷达随舰船移动；
- `T-01` / `T-02` 目标切换；
- 多雷达观测生成；
- 局部观测到世界坐标转换；
- 置信度加权融合与轨迹滤波；
- `TrackIntentTransformer` 在线推理；
- 异步 LLM 调用；
- LLM 超时/异常 fallback；
- LLM 任务级建议到控制器参数的转换；
- 波束方位、宽度、驻留时间、距离门平滑控制；
- 前端展示三部雷达协同跟踪、预测轨迹与控制状态。

### 11.2 当前仍为仿真或示意

当前仍为仿真/示意的部分：

- 后端没有直接读取真实 MATLAB IQ `.mat` 流；
- 前端 `IQ / Range-Doppler` 热力图不是后端真实 IQ 矩阵；
- 动态雷达观测模型是几何测量仿真，不是完整电磁传播/雷达信号处理链；
- 多雷达融合是简化 alpha-beta 滤波，不是严格 EKF/UKF/JPDA/MHT；
- LLM 输出对平台运动只有任务级建议，当前没有真正驱动舰船路线规划器；
- 前端二维坐标和距离单位用于仿真显示，不等价于真实海图坐标。

### 11.3 汇报时建议表述

推荐表述为：

```text
本系统实现了一个后端驱动的认知雷达闭环仿真框架。
当前后端以多雷达几何观测生成目标轨迹，调用已训练的 TrackIntentTransformer 进行轨迹预测和意图/威胁识别，再通过在线 LLM 生成任务级态势建议，最后由约束控制器转换为可执行的波束方位、波束宽度、驻留时间与距离门参数，并通过 WebSocket 驱动前端实时可视化。
```

不建议表述为：

```text
系统已经接入真实 MATLAB IQ 数据并直接完成端到端 IQ 感知。
```

更准确的说法是：

```text
当前系统已经为真实 IQ 接入预留了展示和闭环结构，但在线推理链路使用的是统一世界坐标下的 track 输入。
```

## 12. 与真实系统的扩展关系

如果要从当前仿真系统进一步扩展为真实系统，建议按以下路线推进。

### 12.1 接入真实 IQ 数据

目标：

```text
MATLAB / 雷达采集模块
-> IQ 数据
-> Range-Doppler / 检测
-> 目标量测
-> 世界坐标轨迹
-> TrackIntentTransformer 或 FmcwIntentTransformer
```

可选方案：

1. 保持当前 `TrackIntentTransformer`：先从 IQ 中检测目标，输出 `[range, bearing, doppler]`，再转换为 `[x, y, vx, vy]`。
2. 使用 `FmcwIntentTransformer`：将 IQ/RD 序列直接输入模型，但需要对应训练权重与在线预处理完全一致。

当前更稳妥的工程路线是第一种：先做检测与坐标统一，再复用 track 模型。

### 12.2 增强动态雷达坐标变换

真实动态雷达还需要考虑：

- 舰船 GPS/惯导位置；
- 舰船航向；
- 雷达安装偏角；
- 雷达坐标系到舰体坐标系转换；
- 舰体坐标系到世界坐标系转换；
- 时间同步与延迟补偿。

转换链路可写为：

```text
radar polar measurement
-> radar local Cartesian
-> ship body frame
-> world frame
```

### 12.3 增强 LLM 决策

当前 LLM 是任务级 advisor。后续可以扩展为：

- 输入更多上下文：禁航区、任务区域、规则库、武器/传感器状态；
- 输出结构化任务计划：监视、靠近、远离、交叉定位、重新捕获；
- 增加 JSON schema 校验；
- 增加安全规则层；
- 将 `platform_guidance` 接入舰船运动控制或路径规划模块。

推荐保持结构：

```text
LLM 只给任务建议
安全/约束控制器生成具体控制量
```

### 12.4 增强跟踪融合

后续可替换当前 alpha-beta 融合为：

- EKF：适用于非线性量测模型；
- UKF：适用于更复杂非线性运动；
- JPDA：适用于多目标关联不确定；
- MHT：适用于复杂航迹假设；
- IMM：适用于多机动模型切换。

## 13. 系统演示时的讲解顺序

建议汇报时按以下顺序展示：

1. 展示整体界面，说明三部雷达、两艘舰船、两个目标。
2. 切换 `岸基 / 舰01 / 舰02`，说明只是切换数据视图，三部雷达始终同时工作。
3. 切换 `T-01 / T-02`，观察三部雷达波束平滑转向，说明控制器有转向速率约束。
4. 展示右侧 `Transformer 感知`，解释方位、距离、多普勒、一致性。
5. 展示 `大模型态势推理`，解释意图、威胁、任务动作、自然语言原因。
6. 展示底部三雷达卡片，说明三部雷达对同一目标进行协同跟踪。
7. 打开终端，观察 `[LLM] queued ...` 与 `[LLM] action=...`，说明 LLM 在线调用成功。
8. 说明当前模型输入为 track，不是原始 IQ；IQ 面板当前用于可视化表达，后续可接真实 IQ。

## 14. 总结

当前系统已经形成一个完整的认知雷达闭环原型：

```text
动态场景仿真
-> 多雷达观测
-> 世界坐标融合
-> Transformer 轨迹预测与意图/威胁识别
-> LLM 任务级态势推理
-> 约束控制器生成波束参数
-> 前端实时可视化
```

其核心价值在于将“感知模型”“大模型推理”“控制约束”和“可视化验证”放入同一个在线闭环中。虽然当前 IQ 数据链路仍是仿真展示，尚未接入真实 MATLAB IQ 流，但系统已经具备真实数据接入所需的工程骨架：后端服务、模型加载、轨迹标准化、LLM 决策、控制参数生成和 WebSocket 实时展示。
