# 认知雷达闭环仿真可视化系统

本目录现在支持两种运行方式：

- 纯前端 fallback：只做浏览器内仿真演示。
- 后端真实闭环：Python 后端加载 `TrackIntentTransformer` checkpoint，通过 WebSocket 驱动前端。

## 推荐运行：后端真实闭环

从项目根目录运行：

```powershell
cd E:\VSCodeProject\radarLLM
conda run -n radar python cognitive_radar_loop/run_backend.py --port 5177
```

然后访问：

```text
http://localhost:5177
```

后端会提供静态页面和 `/ws` WebSocket。前端连上后，右上角会显示 `BACKEND T-01/T-02`。

## LLM 在线决策

默认不配置外部 LLM 时，后端使用规则决策 fallback，仍会输出任务级建议和控制器参数。

如需接入 OpenAI-compatible 接口：

```powershell
$env:LLM_PROVIDER="openai_compatible"
$env:LLM_API_URL="http://localhost:8000/v1"
$env:LLM_MODEL="your-model-name"
$env:LLM_API_KEY="your-key-if-needed"

conda run -n radar python cognitive_radar_loop/run_backend.py --port 5177
```

LLM 只输出任务级建议，例如：

```json
{
  "action": "increase_tracking_rate",
  "radar_guidance": {
    "shore": "continuous_designation",
    "v01": "increase tracking rate and narrow beam after lock",
    "v02": "keep_cross_bearing_track"
  }
}
```

底层波束角、波束宽度、驻留时间和距离门由约束控制器生成。

## 后端闭环流程

每个周期执行：

```text
更新 V-01 / V-02 / T-01 / T-02 运动状态
三雷达生成局部量测 range / bearing / doppler
动态雷达量测转换到世界坐标
多雷达融合得到目标 [x, y, vx, vy]
维护最近 24 帧 track buffer
TrackIntentTransformer 预测未来 8 帧轨迹、意图、威胁
LLM / 规则服务生成任务级建议
约束控制器生成可执行雷达参数
WebSocket 推送给前端显示
```

当前 checkpoint：

```text
detect_intention/train_intention/runs/run_20260604_005306/best_by_intent.pt
model_type = track_intent_transformer
input_mode = track
Tin = 24
Tout = 8
```

## 纯前端 fallback

如果只启动普通静态服务：

```powershell
python -m http.server 5177
```

前端无法连接 `/ws`，会自动使用浏览器内仿真逻辑。
