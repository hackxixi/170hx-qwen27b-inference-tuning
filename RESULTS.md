中文 | [English](RESULTS.en.md)

# 全部变体总表

每行一个测过的变体。数字都是**相对该行基线**的变化，不同基线之间不能直接比（基线定义见 [reports/00](reports/00-methodology.md) §2）：

| 基线代号 | 含义 |
|---|---|
| A | 2026-09-24 的生产配置：vLLM 0.28、`-fast`（int4 输出头）、MTP k=4、全词表起草、KV bf16 |
| P | 2026-09-25 的生产配置：A 换成 int8 输出头，测试卡上测 |
| off | 与 P 同配置，另一节点上关掉混合开关 |
| H8 | 带真实流量的 int8 输出头生产副本（同时段 A/B） |
| shuffle | 同一对副本上的 litellm simple-shuffle |

列说明：
- **单流**：c1 每路 decode tok/s 中位，英 / 中。
- **c8**：8 并发聚合 tok/s，英 / 中。
- **long32k**：33K token 输入的 decode tok/s。
- **开发场景**：devbench 聚合 tok/s，c1 / c8（括号内为 c1 单轮 decode 中位）。噪声带：c1 ±0.5%，c8 ±5%。
- **精度**：gate 为贪心 17 条逐 token 全同数；tf 为 teacher forcing top1 一致率，噪声底 99.51%（P 轮）/ 99.45%（A 轮）。
- 通用负载噪声带 ±2%。「—」表示没测。

## 上线（3 个方向）

| 变体 | 基线 | 单流 英 / 中 | c8 英 / 中 | long32k | 开发场景 c1 / c8 | 精度 | 结论 | 报告 |
|---|---|---|---|---|---|---|---|---|
| **int8 输出头**（D2） | A | −4.2% / −0.1% | −2.0% / −4.4% | +1.3%（long4k −7.2%） | −1.5% / −1.4%（decode −0.2%；c8 在噪声内） | gate 0/17，分歧点 top1 概率 0.17–0.60；tf 97.63%（噪声底 99.45%） | **上线** | [02](reports/02-int8-lm-head.md)、[09](reports/09-precision-theory.md) |
| **组合版**：vLLM 0.30 + int4 起草头（C1），测试卡 | P | +7.4% / +8.3% | +4.9% / +5.9% | +7.6%（long4k +8.7%） | +2.3% / +3.5%（decode +9.7%） | gate 2/17，分歧差基线侧 ≤0.25、C1 侧最大 0.375，probe（基线 : 变体）4 : 11 | **上线** | [04](reports/04-decode-six-items.md) |
| 组合版，生产同时段 A/B | H8 | **+12.2% / +11.4%** | **+8.3% / +5.7%** | — | — | 浸泡 319 请求 0 错误 0 Xid | **上线**（7 副本） | [06](reports/06-canary-and-rollout.md) |
| **网关会话粘滞**（litellm session_affinity） | shuffle | — | — | — | 后续轮 TTFT 均值 −85% / −44%，会话耗时中位 −63% / −40%（两轮） | 不改变输出 | **上线** | [07](reports/07-session-affinity-routing.md) |

## 否决（7 个方向）

| 变体 | 基线 | 单流 英 / 中 | c8 英 / 中 | long32k | 开发场景 c1 / c8 | 精度 | 结论 | 报告 |
|---|---|---|---|---|---|---|---|---|
| 170hx-fullstack 原方案（B） | A | +11.3% / −19.8% | −19.6% / −32.6% | −25.8% | +4.4% / −24.1%（decode −4.7%） | 理论量化保真度略高（1–2 dB）；换了微调模型，评测差异 <1σ | 否决（在本仓库负载与硬件下） | [01](reports/01-three-way-comparison.md)、[03](reports/03-dev-workload.md)、[09](reports/09-precision-theory.md) |
| 170hx-fullstack 推理栈 + HyperQwen 模型（C′） | A | +18.0% / −12.2% | −19.3% / −28.9% | −23.0% | +10.0% / −15.2%（decode +6.8%） | 与 A 同权重，多 fp8 KV | 否决（在本仓库负载与硬件下） | [01](reports/01-three-way-comparison.md)、[03](reports/03-dev-workload.md) |
| 截断草稿词表 32k（V32） | P | −7.5% / −1.9% | −10.2% / −1.8% | −10.7% | −8.2% / −2.8%（decode −17.0%） | gate 2/17，分歧 ≤0.125 | 否决 | [04](reports/04-decode-six-items.md) |
| 截断草稿词表 48k（V48） | P | −4.0% / +0.4% | −5.8% / +1.6% | −4.4% | −4.6% / +4.1%（decode −9.1%） | gate 1/17，分歧 ≤0.125 | 否决 | [04](reports/04-decode-six-items.md) |
| 截断草稿词表 64k（V64） | P | −4.2% / +0.9% | −3.3% / +1.7% | −3.0% | −1.5% / +3.3%（decode −6.8%） | gate 1/17，分歧 ≤0.125 | 否决 | [04](reports/04-decode-six-items.md) |
| KV int8（KI8） | P | −4.5% / −3.8% | −5.0% / +0.2% | −19.3%（TTFT 16→38 s） | −53.3% / −56.5%（decode −25.6%） | tf 99.46%，long4k 99.80% | 否决 | [08](reports/08-kv-cache-quantization.md) |
| KV fp8（KF8） | P | −58.5% / −55.4% | −35.7% / −24.8% | −44.7% | −38.0% / −22.6%（decode −45.4%） | tf 99.20%，long4k 98.44%（低于噪声底） | 否决 | [08](reports/08-kv-cache-quantization.md) |
| cpuset 绑 NUMA（CS） | P | −0.6% / −0.9% | −0.5% / +0.6% | −0.1% | +0.1% / +2.9% | gate 17/17 | 否决（噪声内，无收益） | [04](reports/04-decode-six-items.md) |
| cpuset + nice −10（CSN） | P | −0.7% / −0.6% | −0.9% / +0.9% | +0.1% | +0.0% / +5.4% | gate 17/17 | 否决（噪声内，无收益） | [04](reports/04-decode-six-items.md) |
| MTP + suffix，never（只开管线不覆盖） | off | 聚合 −5.4% / −3.7% | — | — | −3.9% / —（c4 −1.8%） | gate 17/17 | 诊断用 | [05](reports/05-mtp-suffix-hybrid.md) |
| MTP + suffix，τ=4 | off | 聚合 +7.1% / +10.1% | +5.0% / +3.9% | — | **−11.7%** / −2.4% | gate 1/17，分歧 ≤0.125；tf 99.52% | 否决 | [05](reports/05-mtp-suffix-hybrid.md) |
| MTP + suffix，ema | off | 聚合 +10.4% / 0.0% | +5.9% / +1.6% | — | **−11.9%** / −5.3% | gate 1/17，1 处 0.25；tf 99.50% | 否决 | [05](reports/05-mtp-suffix-hybrid.md) |
| DFlash2 on vLLM sm80 | — | 推算 +10~18% / −10~12% | 推算 −15~−30% | — | — | 触发 Xid 31，上游未修 | 否决（未上卡） | [10](reports/10-dflash2-and-hybrid-research.md) |

## 其他测过的子项

| 变体 | 基线 | 单流 英 / 中 | c8 英 / 中 | long32k | 开发场景 c1 / c8 | 精度 | 结论 | 报告 |
|---|---|---|---|---|---|---|---|---|
| 基座原样（D1：lm_head int8 + MTP int8） | A | −2.5% / −0.6% | −3.4% / −3.8% | −2.3%（long4k −8.7%） | — | gate 0/17，分歧点与 D2 基本相同 | 选了 D2（MTP 保持 int4） | [02](reports/02-int8-lm-head.md) |
| int4 起草头单独（D3） | P | +2.7% / +0.9% | +0.9% / +2.6% | +2.3% | +0.1% / +7.7%（decode +1.2%） | gate 2/17，分歧 ≤0.125；tf 99.48% | 并入组合版（单独看多数档在噪声内） | [04](reports/04-decode-six-items.md) |
| vLLM 0.29 单独（V029） | P | +1.4% / +1.6% | +2.0% / +2.1% | +1.4% | +0.7% / +4.1%（decode +5.5%） | gate 2/17，分歧 ≤0.125 | 收益小（大半在噪声内） | [04](reports/04-decode-six-items.md) |
| vLLM 0.30 单独（V030） | P | +2.2% / +2.6% | +0.9% / +1.7% | +3.1% | −2.8% / +5.2%（decode −2.9%） | gate 0/17，分歧 ≤0.25 | 作为组合版底座 | [04](reports/04-decode-six-items.md) |
| 基线收尾复测（P3） | P | −0.5% / −0.9% | −0.1% / +1.8% | +0.5% | +0.1% / +4.2% | gate 17/17；tf 99.51% | 噪声带来源 | [00](reports/00-methodology.md) |
| 全词表起草 vs 上游 40k 草稿词表（2026-09-18） | 40k 词表 | 中文 68 → 127 tok/s | 中文 210 → 419 | — | — | 不改变输出 | 采用（已含在基线里） | [01](reports/01-three-way-comparison.md) §4 |
| `DRAFT_TOKENS` 6 vs 4（2026-09-18） | k=4 | 持平 | k=4 高 16–23% | — | — | 不改变输出 | 采用 k=4 | [01](reports/01-three-way-comparison.md) §4 |
| TP=2 跨两卡（2026-09-18） | 单卡 | 24–36 tok/s（慢 6–9 倍） | 42 tok/s | — | — | — | 否决 | [01](reports/01-three-way-comparison.md) §4 |
| 功耗墙 250 W → 300 W | — | — | — | — | — | — | 未测（唯一剩下的硬件杠杆） | [04](reports/04-decode-six-items.md) |

## 汇总
- 上线的 3 个方向：int8 输出头、组合版（vLLM 0.30 + int4 起草头）、网关会话粘滞。
- 否决的 7 个方向：170hx-fullstack（两种形态）、截断草稿词表、KV fp8、KV int8、cpuset / nice、MTP + suffix 混合、DFlash2 on vLLM sm80。
- 贯穿所有测试的瓶颈：满载时 95% 以上的时间顶着 250 W 功耗墙（[00](reports/00-methodology.md) §6）。
