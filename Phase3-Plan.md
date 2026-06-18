# Phase 3 计划 — OneVL × Cosmos-Reason2 → AlpaSim Closed-Loop Integration

**Date:** 2026-06-12
**前置条件:** Phase 2 完成 (c-lang ADE 1.17m, visual decoder validated at 250 scenes)
**核心目标:** 将训练好的模型部署到 AlpaSim 闭环仿真，生成 demo 级别的自动驾驶视频

---

## 现有资产

| 资产 | 位置 | 状态 |
|------|------|:----:|
| **c-lang 模型** (LoRA adapter) | `<instance-id>:/opt/onevl-experiment/output/w8_clang_1000/` | ✅ ADE 1.17m |
| **c-full 模型** (LoRA + vis/lang decoders) | `<instance-id>:/opt/onevl-experiment/output/w9_step2_final/final_model/` | ✅ 0 errors |
| **Base model** | `<instance-id>:/opt/onevl-experiment/models/models--nvidia--Cosmos-Reason2-2B/` | ✅ |
| **AlpaSim instance** | `<alpasim-instance-id>` (g5.4xlarge, 4×A10G) | ⚠️ UI works, sim launch fails |
| **20 NuRec scenes** | AlpaSim EBS | ✅ Visible in UI |
| **1 reconstructed scene** | `pai_05500c90-386b-4f97-99f4-790d7711ef0d.usdz` | ✅ On EBS |

---

## 核心问题

Phase 2 证明了 latent CoT 方法在 **开环评估 (ADE/FDE)** 上有效。
Phase 3 要回答：**"这个模型能在 AlpaSim 闭环仿真中安全驾驶吗？"**

具体子问题：
1. 模型输出的 8-waypoint 轨迹能否转换为 AlpaSim 的 action space？
2. 闭环下 (每步预测依赖上一步执行结果) 是否有误差累积问题？
3. 与 Alpamayo-R1 (10B, 专业 AV 策略) 相比表现如何？
4. 能否生成有说服力的 demo 视频？

---

## 方案选项

### 方案 A: 直接集成 c-lang 作为 AlpaSim Policy (推荐)

**思路:** 将 c-lang 模型包装为 AlpaSim 的 driving policy service，和现有的 A15/AR1/VaVAM 策略并列。

**优点:**
- c-lang ADE 1.17m 足够好（AlpaSim 的 VaVAM policy 也只是 waypoint follower）
- 不需要视觉解码器（推理时已丢弃）
- 2B 参数，L4/A10G 24GB 足够跑推理
- 模型输出格式 (8 waypoints, 4s) 和 AlpaSim 的 planning horizon 兼容

**挑战:**
- AlpaSim 用 gRPC 接口，需要写一个 policy server wrapper
- 需要处理 camera input → model → trajectory → vehicle control 的实时 loop
- AlpaSim 期望的 action format 可能不完全是 8-waypoint (可能是 speed + steering 或 acceleration)

### 方案 B: AlpaSim 闭环 + World Model Demo (可选加分项)

**思路:** 除了方案 A，额外展示视觉解码器的 "future frame prediction" 能力。

**优点:**
- Demo 更有冲击力："模型不仅能开车，还能预测未来会看到什么"
- 对标 NVIDIA Cosmos World Foundation Model 的 narrative
- W9 visual decoder 已经训练好

**挑战:**
- 需要 Emu3 decoder (反向：token IDs → image)，可能有额外依赖
- 预测质量可能不够好 (vis_loss 3.95 ≈ 50% accuracy per token)
- 增加集成复杂度

### 方案 C: PDM-Score Evaluation Only (最小方案)

**思路:** 不做 AlpaSim 集成，只做 NAVSIM PDM-score 标准评估。

**优点:** 快，1-2 天，无新基础设施
**缺点:** 没有视觉 demo，只是数字

---

## 推荐方案: A + B (渐进式)

先做 A（闭环驾驶），验证后再加 B（世界模型可视化）。

---

## 执行计划

### W10: AlpaSim Policy Service (3-4 days)

| Day | Task | Deliverable |
|:---:|------|-------------|
| 1 | 修复 AlpaSim sim launch (`run-alpasim.sh` debugging) | 能用内置策略跑仿真 |
| 2 | 研究 AlpaSim policy interface (gRPC proto, action space) | 接口文档 |
| 3 | 写 `onevl_policy_server.py`: 加载 c-lang model + LoRA, 接收 camera frame, 输出 trajectory | Working server |
| 4 | 集成到 AlpaSim，首次闭环测试 | 车能动（哪怕不完美） |

### W11: 调优 + Demo (2-3 days)

| Day | Task | Deliverable |
|:---:|------|-------------|
| 5 | Trajectory → vehicle control 映射调优 (speed profile, steering) | 平滑驾驶 |
| 6 | 多场景测试 (直道、弯道、交叉口) + 指标收集 | 成功率统计 |
| 7 | 录制 demo 视频 (AlpaSim 渲染 + overlay) | 30s-1min demo video |

### W12 (可选): World Model Visualization

| Day | Task | Deliverable |
|:---:|------|-------------|
| 8 | Emu3 decoder: token IDs → future frame image | 生成的未来帧 |
| 9 | Side-by-side visualization: GT future vs predicted future | 对比图/视频 |
| 10 | 集成到 demo: 驾驶时实时显示"模型想象的下一秒" | 完整 demo |

---

## 技术细节

### AlpaSim Policy Interface (研究中)

基于 June 10 session 的调试，AlpaSim 的架构是：

```
NRE (Neural Reconstruction Engine)
  → 渲染当前场景 camera frames
  → 发送给 Policy (via gRPC)
  → Policy 返回 action (trajectory/control)
  → Physics engine 执行
  → 循环
```

Policy gRPC 接口 (待确认，需要读 AlpaSim 源码):
```protobuf
// 猜测的接口 (需要验证)
message PolicyRequest {
  bytes camera_image = 1;       // 前方 camera PNG/JPEG
  VehicleState ego_state = 2;   // 当前速度、位置、朝向
  NavigationCommand nav = 3;    // 目的地/方向指令
}

message PolicyResponse {
  repeated Waypoint trajectory = 1;  // 未来 N 秒的 waypoints
  // 或者:
  float throttle = 2;
  float steering = 3;
  float brake = 4;
}
```

### Model Inference Pipeline

```python
# onevl_policy_server.py (概念)
class OneVLPolicy:
    def __init__(self):
        self.model = load_cosmos_reason2_with_lora("w9_step2_final/final_model/")
        self.processor = AutoProcessor.from_pretrained(...)
    
    def predict(self, camera_frame, ego_state, nav_command):
        """
        Input: 1920x1080 camera image + vehicle state
        Output: 8 waypoints (x, y, heading) × 0.5s intervals = 4s horizon
        """
        # Format input like training data
        prompt = f"Command: {nav_command}. Velocity: {ego_state.velocity}."
        inputs = self.processor(text=prompt, images=[camera_frame], ...)
        
        # Generate trajectory
        with torch.no_grad():
            output = self.model.generate(inputs, max_new_tokens=100)
        
        # Parse "[x, y, h], [x, y, h], ..." from output text
        trajectory = parse_trajectory(output)
        return trajectory
```

### 资源需求

| 组件 | 实例 | GPU 需求 | 预计费用 |
|------|------|:--------:|:--------:|
| Policy inference (c-lang 2B) | g6.xlarge (L4 24GB) 或 AlpaSim 的 A10G | 4-6 GB | 共用 |
| AlpaSim rendering (NRE) | g5.4xlarge (已有) | 需要 2-3 GPU | $2.02/hr |
| Total (sim running) | — | — | ~$2-3/hr |

---

## 风险评估

| 风险 | 可能性 | 影响 | 缓解 |
|------|:------:|:----:|------|
| AlpaSim policy interface 不兼容 (action space 不是 waypoints) | 中 | 阻塞 | 先读 AlpaSim proto 定义 |
| 闭环误差累积 (每步小错导致越来越偏) | 高 | 质量差 | PID controller 做 trajectory following |
| AlpaSim sim launch 仍然失败 (Jun 10 未解决) | 高 | 阻塞 Day 1 | 这是第一优先级 |
| 2B 模型推理太慢 (>500ms/frame) | 低 | 不流畅 | batch_size=1, bf16, 实际应该 <100ms |
| NRE 渲染需要更多 GPU | 中 | 需要升级实例 | 用 pre-built scenes (不需要 NRE) |

---

## 成功标准

| 级别 | 标准 | 价值 |
|:----:|------|------|
| **最低** | 模型在 AlpaSim 中驾驶一个直道场景不撞车 (30s) | 证明模型能闭环 |
| **良好** | 3+ 场景，包括弯道和交叉口，完成率 >80% | 可以 demo |
| **优秀** | 对标 VaVAM/A15 策略的闭环指标 + 世界模型可视化 | 会议级别 demo |

---

## 预估

| 项目 | 值 |
|------|:---:|
| 总工期 | 5-7 工作日 |
| GPU 费用 | ~$30-50 (AlpaSim $2/hr × ~15hr) |
| 最大风险 | AlpaSim sim launch debugging (Day 1) |
| 最有价值产出 | 30s demo video showing autonomous driving powered by latent CoT |

---

## 与 Phase 1-2 的对比

| Phase | 问题 | 方法 | 产出 |
|:-----:|------|------|------|
| 1 | Latent tokens 能学会吗？ | 100 samples, 5 variants ablation | 核心发现: decoder 是必须的 |
| 2 | 能扩展到真实数据吗？ | 1000 samples, real NAVSIM, visual decoder | ADE 1.17m, 方法验证 |
| **3** | **能实际驾驶吗？** | **AlpaSim 闭环仿真** | **Demo video, 闭环指标** |

---

## 依赖项 (开始前需要确认)

- [ ] AlpaSim sim launch 修复 (Jun 10 还坏着)
- [ ] AlpaSim policy gRPC interface 文档/proto 文件
- [ ] c-lang model + LoRA adapter 可从 S3 下载 (已设置上传)
- [ ] AlpaSim instance 有足够 GPU 同时跑 NRE + policy inference
- [ ] 确认要用哪些 NuRec scenes 做 demo (建议: 1 直道 + 1 弯道 + 1 交叉口)
