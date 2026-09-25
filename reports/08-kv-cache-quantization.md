# 08 KV cache 量化：int8 与 fp8（2026-09-25，均否决）

**结论：** 两种 KV 量化在这张卡、这个模型上都是负收益，不上线。
- **KV int8**（`int8_per_token_head` + TRITON_ATTN）：通用负载短输入慢 1–5%，long32k 慢 19%，首字延迟从 16 s 涨到 38 s；开发场景聚合吞吐掉 53–59%。精度落在噪声底以内。
- **KV fp8**（`fp8_e4m3` + FLASHINFER）：通用负载慢 25–59%，开发场景慢 23–38%；tf 一致率 99.20%，低于噪声底 99.51%。
- 换来的只有 KV 池容量：556k token 扩到约 1.02–1.06M token。这个模型只有 16 层全注意力层需要 KV，bf16 下已经能放 55 万 token，容量不是瓶颈。

测试方法、基线和噪声带的定义见 [00](00-methodology.md)。

## 1. 测试条件

| | 基线 P | KV int8（代号 KI8） | KV fp8（代号 KF8） |
|---|---|---|---|
| 镜像 | HyperQwen 684e927（vLLM 0.28） | 同 | 同 |
| 模型 | `-fast-head8`（int8 输出头） | 同 | 同 |
| 投机解码 | MTP k=4，全词表起草 | 同 | 同 |
| 注意力后端 | FLASH_ATTN（FA2）+ split-KV verify 补丁 | TRITON_ATTN + split-KV verify（int8 路径） | FLASHINFER |
| `--kv-cache-dtype` | bf16 | `int8_per_token_head` | `fp8` |
| CUDA graph | FULL_AND_PIECEWISE | FULL_AND_PIECEWISE | **退化为 PIECEWISE**（见 §4） |
| GPU KV 池（启动日志） | 556,584 token | 1,021,105 token | 1,063,761 token |

- 同一张测试卡、同一套负载，除 KV 相关参数外，其余启动参数和基线一致。
- 精度用独立的 teacher forcing 容器测（`--max-num-batched-tokens` 调小，以便返回全词表 prompt_logprobs），代号 TFP / TFI / TFK 分别对应基线、KV int8、KV fp8。

## 2. 通用负载（abbench）

「速度」列：c1 / c4 / long 为每路 decode tok/s 中位数，c8 为聚合 tok/s。「副列」：c1 / c4 / c8 为聚合 tok/s，long 为 TTFT（秒）。两轮取中位数。

### 2.1 KV int8

| 档位 | 速度 P → KI8 | 变化 | 副列 P → KI8 | 变化 | 接受长度 |
|---|---:|---:|---:|---:|---|
| c1 英 | 156.2 → 149.2 | −4.5% | 148.5 → 142.8 | −3.8% | 3.43 → 3.39 |
| c1 中 | 126.3 → 121.5 | −3.8% | 123.3 → 116.8 | −5.3% | 2.85 → 2.80 |
| c4 英 | 113.9 → 110.8 | −2.7% | 372.4 → 307.3 | −17.5% | 3.33 → 3.31 |
| c4 中 | 94.8 → 93.9 | −1.0% | 325.4 → 320.3 | −1.6% | 2.86 → 2.87 |
| c8 英（聚合） | 626.3 → 594.9 | −5.0% | 每路 93.2 → 89.4 | −4.1% | 3.45 → 3.42 |
| c8 中（聚合） | 511.4 → 512.4 | +0.2% | 每路 76.6 → 76.0 | −0.8% | 2.83 → 2.86 |
| long4k | 144.8 → 136.8 | −5.5% | TTFT 1.20 → 1.46 s | +22% | 3.37 → 3.35 |
| long32k | 123.1 → 99.3 | **−19.3%** | TTFT **16.05 → 38.24 s** | **+138%** | 3.20 → 3.23 |

### 2.2 KV fp8

| 档位 | 速度 P → KF8 | 变化 | 副列 P → KF8 | 变化 | 接受长度 |
|---|---:|---:|---:|---:|---|
| c1 英 | 156.2 → 64.8 | **−58.5%** | 148.5 → 63.2 | −57.4% | 3.43 → 3.42 |
| c1 中 | 126.3 → 56.3 | **−55.4%** | 123.3 → 54.2 | −56.0% | 2.85 → 2.83 |
| c4 英 | 113.9 → 64.0 | −43.8% | 372.4 → 194.6 | −47.7% | 3.33 → 3.32 |
| c4 中 | 94.8 → 52.3 | −44.9% | 325.4 → 181.3 | −44.3% | 2.86 → 2.88 |
| c8 英（聚合） | 626.3 → 403.0 | −35.7% | 每路 93.2 → 59.6 | −36.1% | 3.45 → 3.40 |
| c8 中（聚合） | 511.4 → 384.5 | −24.8% | 每路 76.6 → 56.6 | −26.1% | 2.83 → 2.83 |
| long4k | 144.8 → 68.2 | −52.9% | TTFT 1.20 → 1.47 s | +22% | 3.37 → 3.30 |
| long32k | 123.1 → 68.0 | −44.7% | TTFT 16.05 → 17.20 s | +7% | 3.20 → 3.17 |

接受长度两种变体都和基线持平，变慢不是因为草稿变差，而是每一轮前向本身变慢了。

## 3. 开发场景（devbench）

负载见 [03](03-dev-workload.md)。decode 中位是非工具轮的单轮 decode tok/s；TTFT 后续轮是会话第 2 轮起的中位数。

### 3.1 KV int8

| 档位 | 聚合 tok/s | decode 中位 | TTFT 首轮 s | TTFT 后续轮 s | 会话耗时中位 s | 接受 |
|---|---:|---:|---:|---:|---:|---|
| c1 | 83.3 → 38.9（−53.3%） | 121.9 → 90.7（−25.6%） | 2.28 → 6.96（×3.1） | 0.75 → 2.86（×3.8） | 14.3 → 31.9（×2.2） | 3.50 → 3.39 |
| c4 | 131.8 → 54.3（−58.8%） | 59.0 → 24.1（−59.1%） | 9.59 → 28.66（×3.0） | 0.94 → 3.06（×3.3） | 40.2 → 112.8（×2.8） | 3.48 → 3.47 |
| c8 | 130.4 → 56.8（−56.5%） | 60.9 → 25.0（−59.0%） | 23.08 → 70.33（×3.0） | 1.32 → 5.02（×3.8） | 56.6 → 156.4（×2.8） | 3.43 → 3.47 |

### 3.2 KV fp8

| 档位 | 聚合 tok/s | decode 中位 | TTFT 首轮 s | TTFT 后续轮 s | 会话耗时中位 s | 接受 |
|---|---:|---:|---:|---:|---:|---|
| c1 | 83.3 → 51.6（−38.0%） | 121.9 → 66.6（−45.4%） | 2.28 → 2.67（+17%） | 0.75 → 1.26（+68%） | 14.3 → 23.7（+66%） | 3.50 → 3.58 |
| c4 | 131.8 → 89.0（−32.4%） | 59.0 → 40.0（−32.3%） | 9.59 → 11.29（+18%） | 0.94 → 1.35（+44%） | 40.2 → 53.2（+32%） | 3.48 → 3.54 |
| c8 | 130.4 → 101.0（−22.6%） | 60.9 → 36.9（−39.4%） | 23.08 → 23.20（+1%） | 1.32 → 2.06（+56%） | 56.6 → 69.8（+23%） | 3.43 → 3.48 |

开发场景的噪声带：c1 ±0.5%，c4 / c8 聚合 ±5%（见 [00](00-methodology.md)）。两种变体的降幅都远超噪声带。

## 4. 精度

| 指标 | 基线自比（噪声底） | KV int8 | KV fp8 |
|---|---:|---:|---:|
| tf top1 一致率，16 条短 prompt（13,255 token） | 99.51%（13190） | 99.46%（13184） | **99.20%**（13149） |
| tf top1 一致率，long4k（512 token） | 99.80%（511） | 99.80%（511） | **98.44%**（504） |
| 贪心 17 条逐 token 全同 | 17/17（P vs P3） | 0/17 | 0/17 |
| 贪心分歧点 top1-top2 差值 | — | 全部 ≤0.125 nats | 最大 0.375 nats |
| probe（分歧点重选，基线 : 变体） | — | 10 : 7，差距中位 0.000 | 7 : 10，差距中位 0.125 |
| long32k 贪心文本首个分歧字符（两轮） | 两轮全同 | 第 659 / 第 9 字 | 第 263 / 第 167 字 |

- **KV int8 与噪声底持平**：tf 差 0.05 个百分点，分歧点都在 bf16 分辨率（0.125 nats）以内，属于数值平局。long32k 有一轮在第 9 个字就分叉了，但该处同样是近平局；分叉早只说明长输入下累积的数值差异更早碰上一个平局点。
- **KV fp8 有可测损失**：短 prompt tf 低于噪声底 0.31 个百分点，long4k 低 1.36 个百分点（8 处不一致 vs 1 处），分歧点出现 0.375 nats 的非平局。这和 [09](09-precision-theory.md) 的推算一致：fp8 e4m3 对小于约 0.02 的值落进非规格化区间，误差翻倍，长上下文下更明显。
- long32k 没做 tf：返回全词表 prompt_logprobs 显存不够，改用贪心文本比对。作为参照，D3、vLLM 0.29 / 0.30、组合版在 long32k 上的首分歧位置在第 286–659 字之间。

## 5. 原因

### 5.1 KV fp8 为什么慢一半
1. **sm80 上能跑 fp8 KV 的后端只剩 FlashInfer。**
   - FLASH_ATTN（FA2）在 sm80 上不支持 fp8 KV cache，vLLM 选后端时直接把它排除了（启动日志的候选列表里没有它）。
   - TRITON_ATTN 出现在 vLLM 的候选列表里（启动日志：`potential backends: ['FLASHINFER', 'TRITON_ATTN']`），但本轮没有实测 TRITON + fp8。间接证据是 HyperQwen 的 `triton-spec-attn-fp8-kv.patch` 标注 fp8 路径要求 sm89 及以上（Triton 的 fp8e4nv 类型在 Ampere 上不可用）。
2. **投机解码下 CUDA graph 退化。** FlashInfer 对 spec decode 只支持单 token decode 的 FULL graph（`AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE`）。启动日志：`CUDAGraphMode.FULL_AND_PIECEWISE is not supported with spec-decode for attention backend FlashInferBackend ...; setting cudagraph_mode=PIECEWISE`。MTP k=4 每轮验证 5 个 token，走不了 FULL graph，每一步都要逐段 launch，单流下损失最大（−55~−59%），并发越高、GPU 越忙，launch 开销占比越小（c8 −25~−36%）。
3. **split-KV verify 补丁在 sm80 上没有 fp8 路径。** HyperQwen 的 split-KV verify 内核支持 bf16（FA 路径）和 `int8_per_token_head`（Triton 路径），fp8 分支要求 sm89+，也不挂在 FlashInfer 上。所以 KF8 的验证注意力走的是 FlashInfer 原生实现，长上下文不做 KV 切分，long32k decode 只有 68 tok/s。
4. GPU 采样也印证了这一点：KF8 的功率中位 233 W，SW Power Cap 只在 65% 的采样中触发（基线 98%），说明 GPU 在等 launch，没吃满。

### 5.2 KV int8 为什么预填充慢一倍
1. `int8_per_token_head` 只有 TRITON_ATTN 支持，预填充也跟着走 Triton 的 unified attention。它在 sm80 上的预填充效率明显低于 FA2：long32k TTFT 16 s → 38 s，开发场景首轮 TTFT 约 3 倍。
2. decode 本身损失不大（c1 −4%，c8 聚合 −5% ~ +0%），split-KV verify 有 int8 路径，graph 也保持 FULL。
3. 开发场景每轮都要预填充新增的几千 token（工具结果、代码文件），预填充一慢，聚合吞吐就掉一半以上。
4. KI8 的 GPU 采样：SW Power Cap 在 86% 的采样中触发，功率中位 244 W，SM 时钟中位 1410 MHz，同样说明 GPU 有空转。

### 5.3 为什么容量不是瓶颈
- Qwen3.8-27B 是混合结构：64 层里 48 层是线性注意力（GDN），状态大小固定，与上下文长度无关；只有 16 层全注意力层需要 KV cache。
- 所以 bf16 KV 在单卡 64 GB、`gpu_memory_utilization=0.9` 下已经能放 556,584 token，是 `max_model_len` 262,144 的 2.12 倍（启动日志 `Maximum concurrency ... 2.12x`）。
- 当前负载最长的是开发场景，单会话 23K 前缀加上尾部最多约 2 万 token，8 路并发也远用不满 55 万 token。KV 池翻倍换不来吞吐。
- 组合版（vLLM 0.30）的 KV 池是 528,901 token，比 0.28 略少，同样够用。

## 6. 对照：170hx-fullstack 的 fp8 KV
[170hx-qwen3.8-27b-fullstack](https://github.com/ChinaBoy0618/170hx-qwen3.8-27b-fullstack) 在同款 170HX 上用 SGLang + fp8_e4m3 KV，在本仓库负载与硬件下没有出现这种减半（见 [01](01-three-way-comparison.md)，英文单流 177–188 tok/s）。这说明 fp8 KV 慢不是硬件本身的问题，而是 vLLM 在 sm80 上的实现限制：可用后端只有 FlashInfer，而 FlashInfer 在投机解码下只能跑 PIECEWISE graph。

## 7. 过程记录
1. **KV int8**：`int8_per_token_head` 只有 TRITON_ATTN 实现，所以启动参数同时指定 `--attention-backend TRITON_ATTN`。启动日志确认候选后端只有 `['TRITON_ATTN']`，graph 模式保持 FULL_AND_PIECEWISE，KV 池 1,021,105 token（最大并发 3.90 个 262k 请求）。HyperQwen 的 split-KV verify 补丁有 int8 路径，验证注意力照常走切分。
2. **KV fp8**：先看 vLLM 的后端选择，sm80 上 fp8 KV 的候选是 `['FLASHINFER', 'TRITON_ATTN']`，于是显式指定 FLASHINFER。启动时出现一条警告：
   `CUDAGraphMode.FULL_AND_PIECEWISE is not supported with spec-decode for attention backend FlashInferBackend (support: AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE); setting cudagraph_mode=PIECEWISE`
   同时 FlashInfer 报告 `kv_cache_dtype=torch.float8_e4m3fn, arch=sm80`，KV 池 1,063,761 token（4.06 倍）。测速结果单流减半，与这条警告对得上：投机解码每轮验证 5 个 token，不是「单 token decode」，FlashInfer 不支持对它做 FULL graph，只能逐段 launch。
3. **稳定性**：两项在 c4 / c8 下都跑完，无错误、无 Xid。
4. **精度容器**：tf 要返回全词表 prompt_logprobs，常规参数下 long4k 那条会显存不足（[02](02-int8-lm-head.md) §6 遇到过）。这次另起 TFP / TFI / TFK 三个专用容器，`--max-num-batched-tokens 2048`，只跑 tf（16 条短 prompt + long4k），全部成功。
5. **判定**：两项速度都远超噪声带地变慢，KV 容量又不是瓶颈（§5.3），直接否决，没有再尝试 TRITON + fp8 等组合。

## 8. 复现
- 启动参数：KI8 加 `--attention-backend TRITON_ATTN --kv-cache-dtype int8_per_token_head`；KF8 加 `--attention-backend FLASHINFER --kv-cache-dtype fp8`。其余与基线相同。
- 测速与精度脚本：[`bench/abbench.py`](../bench/abbench.py)、[`bench/devbench.py`](../bench/devbench.py)、[`bench/gate.py`](../bench/gate.py)、[`bench/probe.py`](../bench/probe.py)、[`bench/tf.py`](../bench/tf.py)。
- 全量原始表：[data/decode-analysis-all.md](data/decode-analysis-all.md)（代号 KI8 / KF8 / TFI / TFK）；逐轮数据见 [data/raw-decode-abbench.md](data/raw-decode-abbench.md)、[data/raw-decode-devbench.md](data/raw-decode-devbench.md)；GPU 采样见 [data/gpu-sampling.md](data/gpu-sampling.md)；KV 池见 [data/kv-pool.md](data/kv-pool.md)。
