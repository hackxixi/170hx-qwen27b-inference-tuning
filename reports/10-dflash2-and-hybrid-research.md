# 10 调研：DFlash2 on vLLM sm80、MTP 混合起草、草稿词表上限（2026-09-25）

这份报告记录三项没有（或没法）直接上卡、靠调研和推算下结论的事项：
1. DFlash2 在 vLLM + sm80 上为什么会触发 Xid 31，现在有没有修法；
2. 就算修好了，vLLM + DFlash2 相对现役 MTP 值不值得换；
3. MTP 与 suffix / n-gram 混合起草的现有实现，以及我们为什么选了其中一种做法（实测结果见 [05](05-mtp-suffix-hybrid.md)）；
4. 截断草稿词表的理论收益上限。

**结论：** DFlash2 在 vLLM sm80 上上游未修，保底用 `SPEC=mtp`；即使修好，按显存读取量推算，只有英文单流会快 10–18%，中文和并发都会变慢，所以不换。混合起草选了 TRT-LLM 式覆盖法，实测负收益。截断草稿词表的收益上限只有约 2–4%，而实测截断带来的接受率损失更大。

## 1. DFlash2 在 vLLM sm80 上触发 Xid 31

### 1.1 现象
- HyperQwen 在 sm80 上的默认配置是 `SPEC=dflash2`。在 CMP 170HX（GA100，sm80，64 GB）上，持续负载几分钟内就会报 `Xid 31 MMU Fault ... FAULT_PDE ACCESS_TYPE_VIRT_READ`，EngineCore 退出，fault 地址每次不同。同一台机器上 `SPEC=mtp` 能稳定跑 9 小时以上。
- 本仓库的部署从一开始就用 `SPEC=mtp`，没有在生产上复现过这个问题。

### 1.2 相关 issue 与 PR（状态截至 2026-09-25）

| 链接 | 内容 | 状态 |
|---|---|---|
| [HyperQwen #98](https://github.com/syv-ai/HyperQwen/issues/98) | sm80 上 `SPEC=dflash2` 任意 k 都会反复触发运行时 Xid 31；`SPEC=mtp` 稳定 | open |
| [HyperQwen #72](https://github.com/syv-ai/HyperQwen/issues/72) | DFlash2 在 170HX 上的确定性累积越界：约 1.1 万 decode 步 / 第 24 个请求时引擎退出 | open |
| [HyperQwen #91](https://github.com/syv-ai/HyperQwen/issues/91) | 把 `_spec_attn_partial` 里的 block id 转成 int64（`blk = blk.to(tl.int64)`），针对 #86 的 int32 溢出 | 已合入 |
| [vllm#55279](https://github.com/vllm-project/vllm/issues/55279) | #72 的上游镜像报告 | open；评论里的最终结论是 int64 转换 |
| [vllm#56736](https://github.com/vllm-project/vllm/issues/56736) | 混合 Mamba/GDN + 投机解码的 Xid 31，报告者用同款 170HX | open |
| [wtdcode/vllm-backport PR#82](https://github.com/wtdcode/vllm-backport/pull/82) | sm8x 上 spec decode 验证的 split-KV 内核，附带 KV gather 边界 mask 修复 | open |

### 1.3 根因
1. **出问题的内核**：HyperQwen 的补丁 `patches/spec-decode-attn.patch` 加了一个 Triton 内核 `_spec_attn_partial`，用于多 token 验证时切分 KV 做注意力（split-KV verify）。它按 block table 拼出 K/V 的地址：`k_ptr + blk * stride_kb + slot * stride_ks + kvh * stride_kh + d`。
2. **int64 转换不够**：#86 发现 `blk * stride_kb` 在大 KV 池上会溢出 int32，#91 把 `blk` 转成 int64 并合入。但 #72 的维护者随后在 3090 上实测了溢出阈值，指出 170HX 的 KV 池到不了这个阈值；而且 `SPEC=mtp` 也走同一个内核却能跑 9 小时。所以 int64 不能解释 sm80 上的故障。vllm#55279 的最后结论停在 int64 转换上，这不够。
3. **缺的是组合地址的边界检查**：这个内核只对各个下标做了 mask（`k_ok = pos < kv_len`），没有检查拼出来的整条地址是否落在 KV cache 范围内。一组「自洽但过期」的参数（例如 block table 里残留的旧 block id）就能让地址走出缓存区。
4. **同款硬件上的确认**：vllm#56736 的报告者在同款 170HX 上做了单变量 A/B（rev `ce409399e4`，4 请求长上下文循环，单流），结果见其[更正评论](https://github.com/vllm-project/vllm/issues/56736#issuecomment-5683158869)：

   | 组 | 结果 |
   |---|---|
   | 不修 | 2/2 次都停在第 23 个请求，Xid 31 |
   | 只加 #56756（mamba 侧快照） | 2/2 次同一请求、同一 fault 地址 |
   | 关掉 split-KV verify（`SPEC_ATTN=0`） | 40 个请求无故障 |
   | 只给验证内核加边界 mask | **150 个请求无故障** |

   修法是对组合后的偏移做边界检查：`k_lim = nblocks * stride_kb`，mask `(k_off >= 0) & (k_off < k_lim)`。PR#82 正文另报：修之前连续 5 次在第 14 轮以内崩、停在同一位置；修之后 ≥25 轮 / 150 个请求 0 次崩溃。
5. **需要说明的推断**：vllm#56736 修的是报告者自己分支里的 `spec_decode_attn.py`，不是 HyperQwen 这份补丁。我们把两者联系起来，是因为它们结构相同（同样的 split-KV 验证、同样只 mask 下标、同样已做 int64 转换），故障特征也一致（FAULT_PDE / VIRT_READ、地址漂移、关掉该内核就不崩）。这是类比推断，没有在 HyperQwen 的补丁上做过同样的 A/B。
6. **为什么 SGLang 上能跑通**：170hx-fullstack 在 SGLang 上跑 DFlash2，在本仓库负载与硬件下测试的 252 个请求 0 错误、0 Xid（见 [01](01-three-way-comparison.md)）。SGLang 不走这个 Triton 内核，推测这就是它不受影响的原因（未验证）。

### 1.4 我们的做法
- 保底方案：`SPEC=mtp`，`DRAFT_TOKENS=4`，`MTP_DRAFT_VOCAB=0`（全词表起草）。
- 上游合入边界 mask 之前不在 vLLM 上用 DFlash2。

## 2. 推算：vLLM + DFlash2（block 8）相对 MTP k=4

假设 Xid 31 修好了，vLLM 上跑 DFlash2 值不值得换？没法实测，只能按显存读取量推算。

### 2.1 方法
- 这是一个按显存读取量的粗估，只作量级参考：假设每一轮（一次起草 + 一次验证）的耗时正比于目标模型权重和草稿权重的读取量。
- 需要说明：170HX 满载时实际顶着 250 W 功耗墙（见 [00](00-methodology.md) §6），不是纯带宽受限。而且这个模型会高估权重读取量的影响：按它估算，int8 输出头（[02](02-int8-lm-head.md)）应慢约 15%，int4 起草头（D3）应快 13–15%，实测分别只有 0–5% 和 1–3%。所以下面的数字只能看方向，DFlash2 不值得换的判断主要依据 [01](01-three-way-comparison.md) 在 SGLang 上的实测。
- 每 token 的读取量 = 每轮读取量 / 每轮平均产出的 token 数（接受长度，含验证补上的那 1 个）。
- 每轮读取量与接受长度：

  | | 每轮读取量 | 接受长度 英 | 接受长度 中 | 每 token 读取 英 | 每 token 读取 中 |
  |---|---:|---:|---:|---:|---:|
  | MTP k=4（验证 5 个 token） | 约 16.5 GB | 3.4 | 2.8 | 4.85 GB | 5.89 GB |
  | DFlash2 block 8（验证 9 个 token） | 约 17.3 GB | 4.0 | 2.6 | 4.33 GB | 6.65 GB |

  接受长度取自 [01](01-three-way-comparison.md) 的实测（DFlash2 的数值来自 SGLang 上同一个草稿模型，本负载下）。
- 单流估计：英文 4.85 / 4.33 ≈ +12%，中文 5.89 / 6.65 ≈ −11%。

### 2.2 结果

| 场景 | DFlash2 相对 MTP k=4 |
|---|---:|
| 英文单流 | +10% ~ +18% |
| 中文单流 | −10% ~ −12% |
| 4 并发 | −10% ~ −20% |
| 8 并发 | −15% ~ −30% |

- 单流是按读取量的粗估。并发下每轮要验证的 token 数是 batch × (k+1)：DFlash2 每路验证 9 个，MTP 每路验证 5 个。170HX 的算力弱（功耗墙 250 W 下 SM 时钟中位约 1350 MHz），批一大验证的计算量就成了主要开销，block 8 的验证越来越重。并发档的区间是在单流估计上叠加这一项的经验估计。
- 这和 01 在本仓库负载与硬件下、SGLang 上的实测趋势一致：DFlash2 只在英文单流领先，中文和并发都落后。

### 2.3 结论
不换。生产流量以中文和并发为主，DFlash2 的收益只在英文单流。

## 3. MTP 与 suffix / n-gram 混合起草：现有实现

开发场景里有大量「照抄上下文」的输出（改代码、工具调用参数），suffix / n-gram 草稿在这类输出上能一次猜中很长一段。调研了现有实现，看看怎么和 MTP 组合。

| 实现 | 做法 | 公开数据 |
|---|---|---|
| TensorRT-LLM `use_sa_spec` | 模型草稿（MTP / EAGLE）每步照跑；suffix 自动机匹配长度 ≥ `sa_spec_threshold`（默认 4）时，用 suffix 草稿覆盖模型草稿；K 固定 | Baseten 报告：代码编辑任务接受长度 +34%，agent 编码场景最多 +40% |
| SGLang [PR#27628](https://github.com/sgl-project/sglang/pull/27628)（`HYBRID_SUFFIX_MTP`） | suffix 与 MTP 混合 | SWE-bench 并发 2 为 3.20×；并发 16 反而不如纯 suffix。已关闭，未合入 |
| flozi00/vllm-suffix-hybrid（GitHub） | vLLM 上的 suffix 混合实验 | — |
| [ArcticInference](https://github.com/snowflakedb/ArcticInference) | suffix 只和自家的 MLP speculator（arctic 草稿）组合；suffix 分数 ≥ 投机长度时整行覆盖，否则用 MLP 草稿 | — |
| SuffixDecoding 论文（[arXiv 2411.04975](https://arxiv.org/abs/2411.04975)） | 纯 suffix 与 hybrid 对比 | AgenticSQL：纯 suffix 5.3×，hybrid 4.1×，EAGLE-3 1.6× |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | `--spec-type` 可以并列多种草稿来源 | — |
| vLLM 上游 [#46977](https://github.com/vllm-project/vllm/issues/46977) | 混合起草的需求讨论 | open，未实现 |

- ArcticInference 的覆盖规则已在其源码里核对过（`model_runner.py`：非纯 suffix 模式下 `min_score = n_predict`，`score >= min_score` 才用 suffix 草稿）。表中其余公开数据来自各项目的文档、PR 和论文，本仓库没有复现。
- 共同点：公开的正收益都出现在「输出大段照抄上下文」的负载上；并发一高，混合的优势就变小甚至反转。

**我们的选择**：TRT-LLM 式覆盖法。MTP 每步照跑，K 固定为 4，满足阈值的行用 suffix 草稿原地覆盖。理由：改动最小（不改调度和验证），概率采样下的正确性容易保证（覆盖行的 draft_probs 改成 one-hot），而且能直接用 ArcticInference 的 suffix C++ 扩展。第二步「放开 K」的前提是第一步收益 ≥10%。实测见 [05](05-mtp-suffix-hybrid.md)：开发场景单会话慢 12%，第二步没有评估。

## 4. 草稿词表的理论上限
- 截断草稿词表省下的是起草头（lm_head 的草稿副本）的读取量，接受长度则会因覆盖缺口下降。
- 现成的参照是 D3（int4 起草头，见 [04](04-decode-six-items.md)）：它把每次起草读的输出头从 int8 换成 int4，起草头读取量减半，接受长度不变，实测只快 1–3%（c1 +0.9 ~ +2.7%，c8 聚合 +0.9 ~ +2.6%，长上下文 +2 ~ +3%）。
- 由此推算，**整个起草头的读取开销上限约占 2–4%**。截断词表就算不损失接受长度，最多也只能省下这么多。
- 实测截断词表让接受长度下降 0.1–0.4（32k 档英文 3.43 → 3.04），速度损失 3–12%，远大于这 2–4% 的上限。所以截断词表在这台机器上不可能有净收益，与实测一致。
