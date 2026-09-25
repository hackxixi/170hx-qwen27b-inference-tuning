# 00 测试方法

本仓库所有报告共用的测法、基线、噪声带和正确性判据。各报告只写与此不同的地方。

## 1. 硬件与隔离
- 单张 CMP 170HX（GA100，sm_80，64 GB，PCIe Gen2 x4，无 P2P，功耗墙 250 W）。每个变体独占一张卡，测试容器单独起，不进生产网关。
- 测试卡是从生产里临时摘下来的一个副本，测完恢复。同一轮对比里的所有变体都在同一张卡上跑，避免卡间差异。
- 宿主内存紧张时，另起一个守护脚本在可用内存低于阈值时杀掉测试容器，防止影响同机的生产副本。
- 测试期间曾因外部供电中断，受影响的轮次整体作废重测。

## 2. 基线
不同阶段的基线不同，各报告的「相对变化」都只对本报告的基线算，**不能跨报告直接比较绝对值**。

| 阶段 | 报告 | 基线 | 说明 |
|---|---|---|---|
| 三方对比、int8 输出头、开发场景 | [01](01-three-way-comparison.md)、[02](02-int8-lm-head.md)、[03](03-dev-workload.md) | **A**：当时的生产配置，HyperQwen 684e927（vLLM 0.28）、`-fast` 模型（**int4 输出头**）、MTP k=4、全词表起草、KV bf16 | 2026-09-24 |
| decode 提速各项 | [04](04-decode-six-items.md)、[08](08-kv-cache-quantization.md) | **P**：int8 输出头上线后的生产配置，即 A 换成 `-fast-head8` | 2026-09-25，测试卡 |
| MTP + suffix 混合 | [05](05-mtp-suffix-hybrid.md) | **off**：同一节点、同一镜像、关掉混合开关 | 与 P 同配置，但在另一节点上测 |
| 生产 A/B | [06](06-canary-and-rollout.md) | 另一个跑 int8 输出头的生产副本 | 两边同时接真实流量 |
| 会话粘滞 | [07](07-session-affinity-routing.md) | 同一对副本上的 simple-shuffle | — |

例：03 里 A 的开发场景 c1 聚合是 88.1 tok/s，04 里 P 的是 83.3 tok/s。两者是不同节点、不同日期、不同模型（int4 / int8 输出头），不能拿来算变化。

## 3. 负载

### 3.1 通用负载 abbench（[`bench/abbench.py`](../bench/abbench.py)）
- prompt：英文组 8 条、中文组 8 条（英文组里含 2 条中文提问，用来覆盖中英混合）。英文组 4 条是写代码（Python / Go / Rust / SQL），其余是说明文、总结、故事、需求文档。
- 设置：贪心（temperature 0）、关思考（`enable_thinking: false`）、`max_tokens=900`、流式。
- 档位：
  - **c1**：8 条依次单流，看每路 decode tok/s 的中位数；
  - **c4**：取前 4 条同时发，看每路中位与聚合；
  - **c8**：8 条同时发，看聚合 tok/s；
  - **long4k / long32k**：约 4,179 / 33,203 token 的中文长文档加一句总结要求，单流，`max_tokens=512`，看 TTFT 和 decode tok/s。
- 每档预热后跑 2 轮，取中位数。每条请求加确定性前缀 `[r<轮>-<序号>]`，两轮不同，避免第二轮命中前缀缓存。
- decode tok/s = (输出 token 数 − 1) / (最后一个 token 时间 − 首 token 时间)。
- 接受长度 = 每轮验证平均产出的 token 数（含补上的那 1 个），取自服务端 `/metrics` 的 spec decode 计数差值。

### 3.2 开发场景 devbench（[`bench/devbench.py`](../bench/devbench.py)）
- 约 23K token 的 Claude Code 前缀（system prompt + 14 个工具定义 + 首条消息里的上下文块），8 条多轮会话共 35 轮。详见 [03](03-dev-workload.md)。
- 先在基线上录一次（record），之后所有配置都回放同一份历史，保证输入完全相同。
- 档位是**会话并发** c1 / c4 / c8；看聚合 tok/s、非工具轮的单轮 decode 中位、首轮与后续轮 TTFT、会话耗时中位、前缀命中率、工具调用正确率。

### 3.3 abbench 的 16 条 prompt 原文
这些 prompt 是我们自己写的。每条实际发送时前面加 `[r<轮>-<序号>] `，例如 `[r1-0] Write a Python function ...`。c4 只用每组的前 4 条。

英文组：

| 编号 | 类型 | prompt |
|---|---|---|
| en#0 | 代码（Python） | Write a Python function that parses an ISO-8601 duration string into seconds, with tests. |
| en#1 | 说明文（网络） | Explain in detail how TCP congestion control works, covering slow start, AIMD and fast recovery. |
| en#2 | 中文长文 | 写一篇约800字的文章，介绍长江流域的地理、气候和主要城市。 |
| en#3 | 代码（Go） | Implement a thread-safe LRU cache in Go with generics and explain the design. |
| en#4 | 总结 | Summarize the causes and consequences of the 2008 financial crisis in a structured way. |
| en#5 | 代码（Rust，中文提问） | 用 Rust 写一个命令行工具，统计目录下各类文件的行数，并解释关键代码。 |
| en#6 | 说明文（Transformer） | Describe how a transformer decoder works, step by step, including KV caching. |
| en#7 | 代码（SQL） | Write a SQL schema for a library system and five non-trivial example queries. |

中文组：

| 编号 | 类型 | prompt |
|---|---|---|
| zh#0 | 中文长文 | 写一篇约800字的文章，介绍长江流域的地理、气候和主要城市。 |
| zh#1 | 说明文（网络） | 请详细解释 TCP 拥塞控制的原理，包括慢启动、AIMD 和快速恢复。 |
| zh#2 | 计划文档 | 用中文写一份新员工入职培训计划，分周列出目标、内容和考核方式。 |
| zh#3 | 对比分析 | 请比较 Python、Go 和 Rust 在后端开发中的优缺点，用中文详细回答。 |
| zh#4 | 故事 | 讲一个关于人工智能和一位老木匠的中文短篇故事，约600字。 |
| zh#5 | 总结 | 请用中文总结 2008 年金融危机的起因、过程和影响，分点论述。 |
| zh#6 | 需求文档 | 以产品经理的视角，用中文写一份智能家居 App 的需求文档大纲并解释每一部分。 |
| zh#7 | 说明文（Transformer） | 请用中文解释 Transformer 解码器的工作原理，包括 KV 缓存。 |

长输入：
- long4k：一篇中文长文档（按 vLLM tokenizer 计 4,179 token）后接一句「请用中文分点总结上面材料的主要内容，约 400 字。」，前缀同样是 `[r<轮>-0] `，`max_tokens=512`。
- long32k：同样的格式，文档约 33,203 token。
- 文档本身不随仓库分发，换成任意同长度的中文材料即可复现。

**一个需要知道的缓存细节**：gate（见 §5）的 long4k 条目和 abbench 第 1 轮用的是同一个前缀 `[r1-0] `，所以 abbench long4k 第 1 轮能命中 gate 留下的 3,456 token 缓存，TTFT 只有约 0.4 s；第 2 轮（前缀 `[r2-0] `）完全不命中，TTFT 约 2.0 s。各报告里 long4k 的「TTFT 中位」是这两轮的均值（约 1.2 s），**真实的冷启动 TTFT 看第 2 轮的约 2.0 s**。01 那一轮没有先跑 gate，两轮都是约 2.0 s。long32k 两轮都只命中开头 3,456 token，不受影响。

### 3.4 devbench 的负载组成
- **共享前缀**（所有会话相同，约 23.0K token）：
  - Claude Code 的 system prompt（约 2.7 万字符）；
  - 14 个工具定义（Anthropic 格式转成 OpenAI function 格式）：Agent、Bash、Edit、Glob、Grep、Read、Skill、TaskCreate、TaskGet、TaskList、TaskUpdate、ToolSearch、Workflow、Write；
  - 首条 user 消息开头的上下文块（约 1.2 万字符）：可延迟加载的工具名列表与 system-reminder 块。
  - 前缀取自 170hx-fullstack 仓库里用于预热的 Claude Code 请求样本，不随本仓库分发。
- **会话**：

  | 会话 | 语言 | 轮数 | 步骤序列 | 任务类型 |
  |---|---|---:|---|---|
  | s1-py-bugfix | 英 | 3 | user → user → user | 给出一段 Python 模块，定位并修复 bug，再追问 |
  | s2-tool-grep-read | 英 | 7 | user → tool → tool → tool → user → tool → tool | 要求用工具在仓库里查某个配置项的处理逻辑，工具结果（Grep / Read 输出）回填 |
  | s3-rust-feature | 英 | 3 | user → user → user | 给出一段 Rust 代码，按要求加功能 |
  | s4-py-unittests | 英 | 3 | user → user → user | 给出一段 Python 配置代码，写 pytest 单测 |
  | s5-zh-bash | 中 | 3 | user → user → user | 给出一段 shell 启动脚本，加参数，不许调用工具 |
  | s6-tool-bash-test | 英 | 6 | user → tool → tool → tool → user → tool | 单测失败排查：跑测试、读代码、修复，工具结果回填 |
  | s7-review-patch | 英 | 4 | user → user → user → user | 给出一个补丁，做代码评审，不许调用工具 |
  | s8-zh-tool-rust | 中 | 6 | user → tool → tool → tool → user → user | 中文，用工具查找 Rust 代码并解释一个并发问题 |

  - 「user」步骤是固定的用户消息；「tool」步骤是按工具名准备好的模拟工具结果，录制时按基线实际发出的工具调用回填（调用了 Read 就回填对应文件内容，调用了 Bash 就回填预先准备的命令输出）。
  - 会话里引用的源码和补丁取自两个开源仓库（170hx-fullstack 与 HyperQwen），生成脚本 [`bench/gen_sessions.py`](../bench/gen_sessions.py) 需要这两个仓库的 checkout，本仓库不收第三方源码，也不分发生成出来的 `workload.json`。

## 4. 噪声带
- 定义：同一配置在一轮测试的开场（P）和收尾（P3）各测一次，两者之差就是这一轮的噪声带。开场和收尾之间隔了约 10 小时、十几个变体。
- 实测（P3 相对 P）：

  | 负载 | P3 相对 P | 定为噪声带 |
  |---|---|---|
  | 通用负载各档 | −0.9% ~ +1.8%（c1 英 −0.5%，c1 中 −0.9%，c8 中聚合 +1.8%） | **±2%** |
  | 开发场景 c1 聚合 / decode 中位 | +0.1% / −0.2% | **±0.5%** |
  | 开发场景 c4 / c8 聚合 | −2.6% / +4.2% | **±5%** |
  | 开发场景 c4 / c8 单轮 decode 中位 | +4.7% / −4.7%；CS 复测时 ±3~8%，个别到 ±11% | **不作判据** |
  | 贪心 17 条 | 17/17 逐 token 全同 | — |
  | tf top1 一致率（自比） | 99.51% | 噪声底 |

- 并发档的单轮 decode 中位受「哪几轮恰好同时在跑」影响很大，波动可到 ±11%，所以只看聚合吞吐。例如 D3 在开发场景 c4 聚合 −6.9%、c8 聚合 +7.7%，方向相反、幅度都在 ±5~8%，只能算噪声。
- cpuset 变体（CS）的全部输出与 P 逐 token 相同，等于又做了一次 P 复测：通用负载 ±1%，开发场景并发档 ±3~8%，与上面的噪声带一致。
- 02 那一轮（int8 输出头）用的是另一组 A 自比，tf 噪声底是 99.45%。

## 5. 正确性硬门
每个变体上线前必须过三道检查。

1. **gate：贪心逐 token 比对**（[`bench/gate.py`](../bench/gate.py)）
   - 16 条短 prompt（`max_tokens` 900）加 1 条 long4k（512），贪心生成，记录每个位置的 token 和 top-2 logprobs。
   - 与基线逐 token 比对，统计「全同」条数，以及每条首个分歧点上双方的 top1-top2 差值。
   - 判据：分歧点差值 ≤0.125 nats（bf16 logprob 的分辨率，即真平局）视为数值级分歧；出现 ≥0.25 的分歧要进一步看 probe 和 tf。
2. **probe：分歧点重选**（[`bench/probe.py`](../bench/probe.py)）
   - 取首个分歧点之前的前缀，发给基线走一次 prefill 路径，贪心取 1 个 token，看它选基线的还是变体的。
   - 真平局下两边的选择应接近对半，基线自己换一条计算路径也会翻。例如 D3 是 9 : 6，差距中位 0.000。
3. **tf：teacher forcing**（[`bench/tf.py`](../bench/tf.py)）
   - 把基线的贪心输出（prompt + 输出 token）整段喂给变体，`prompt_logprobs=1`，逐位置比较变体的 argmax 与基线 token，算 top1 一致率。
   - 共 16 条、13,255 个输出 token（02 那一轮是 13,513 个），另测 long4k 的 512 个 token。
   - 判据：落在基线自比的噪声底以内（99.51% 或 99.45%）视为等价。
   - tf 需要返回全词表 prompt_logprobs，显存开销大：long4k 那条要单独起小 `max-num-batched-tokens` 的容器，long32k 做不了，改用贪心文本首个分歧字符位置比较。

## 6. GPU 采样
- 测试全程用 `nvidia-smi --query-gpu` 每 500 ms 采一次：功率、SM 时钟、显存时钟、throttle 原因位。
- 只统计「忙碌样本」（GPU 利用率 >50%），算功率中位数 / P90 / 峰值、SM 时钟中位数 / 最小值、显存时钟、温度，以及 throttle 原因位里 `0x4`（SW Power Cap）出现的比例。
- 全部变体的采样表见 [data/gpu-sampling.md](data/gpu-sampling.md)。功率峰值（290–360 W）是个别采样点的瞬时值，P90 只有 254–263 W，中位 246–249 W，贴着 250 W 的墙。
- 典型结果（基线 P）：2,690 个样本，功率中位 248 W（墙 250 W）、最大 338 W，SM 时钟中位 1350 MHz（上限 1695），显存 1728 MHz 满频，98% 的样本触发 SW Power Cap。这说明 decode 受功耗墙限制，CPU 侧的绑核、提优先级不会有收益（见 [04](04-decode-six-items.md)）。

## 7. 上线流程
独立测试卡 → 单副本金丝雀（接真实流量）浸泡 → 同时段生产 A/B → 滚动上线（逐个副本自检，失败即停）。见 [06](06-canary-and-rollout.md)。

## 8. 时间线
只列日期和先后顺序。

**2026-09-18**
1. 每张卡起一个独立的 HyperQwen 副本（vLLM 0.28、MTP），经 litellm 网关对外。
2. 试 TP=2 跨两卡：比单卡慢 6–9 倍，放弃，定为单卡单副本。
3. 发现上游默认 40k 截断草稿词表对中文几乎无覆盖，改成全词表起草（`MTP_DRAFT_VOCAB=0`），中文单流 68 → 127 tok/s。
4. `DRAFT_TOKENS` 4 对 6：8 并发 k=4 高 16–23%，定 k=4。
5. 两副本浸泡 30 分钟（各 4 路多轮、200–12k 上下文）：3,246 请求，0 错误，0 Xid。
6. 网关从 least-busy 改为 simple-shuffle（原因见 [07](07-session-affinity-routing.md) §1）。

**2026-09-24**
1. 三方对比（[01](01-three-way-comparison.md)）：A 测完 → B 测完 → C（HyperQwen 模型直接上 SGLang）两种投机方式都起不来 → 改做 C′ 并测完。决定不迁移。
2. 理论精度推算（[09](09-precision-theory.md)），结论是现役最大的短板是 int4 输出头。
3. 当晚测输出头 int4 → int8（[02](02-int8-lm-head.md)）：A → D1 → D2，外加 D2 自复现。D2 的代价在 5% 以内，决定上线；D3（int4 起草头）补丁备好但没测。

**2026-09-25**
1. int8 输出头滚动上线到全部 7 个副本（[02](02-int8-lm-head.md) §5）。
2. 开发场景负载（[03](03-dev-workload.md)）：第一次录制撞上 400 报错，修掉后录制，再依次测 A、D2、B、C′。
3. decode 各项（[04](04-decode-six-items.md)、[08](08-kv-cache-quantization.md)）在测试卡上跑：
   - 第一次开场的基线 P 跑到一半遇上外部供电中断，整轮作废；换一张测试卡从 P 重跑。
   - 顺序：P → D3 → V32 → V48 → V64 → V029 → V030 → CS → CSN → C1 → KI8 → KF8 → tf 专用容器（TFP / TFI / TFK）→ P3。
   - 同时在另一对副本上做了会话粘滞的两轮对照（[07](07-session-affinity-routing.md)），第 2 轮的 shuffle 组因供电中断在副本重启后重跑。
4. MTP + suffix 混合（[05](05-mtp-suffix-hybrid.md)）在另一节点的测试卡上跑：off → never → t4 → ema → tf。开发场景单会话 −12%，否决。
5. 网关开启会话粘滞（[07](07-session-affinity-routing.md)）。
6. 组合版（vLLM 0.30 + int4 起草头）单副本金丝雀 → 浸泡约 80 分钟，其间做 3 轮同时段 A/B → 滚动上线到全部 7 个副本（[06](06-canary-and-rollout.md)）。

## 9. 原始数据索引
都在 [`reports/data/`](data/)，由 `bench/` 下的汇总脚本从原始 json / csv 生成（原始 json 含完整输出文本，不随仓库分发）：

| 文件 | 内容 |
|---|---|
| [raw-threeway.md](data/raw-threeway.md) | 三方对比：每档两轮与中位、每路 / 聚合 / TTFT / 输出 token / 接受长度 / 接受率；c1 逐条 |
| [raw-head8.md](data/raw-head8.md) | 输出头 int4 → int8：同上 |
| [raw-decode-abbench.md](data/raw-decode-abbench.md) | decode 各项（含 KV 两项与 P3）的通用负载：同上 |
| [raw-decode-devbench.md](data/raw-decode-devbench.md) | decode 各项的开发场景：c1 / c4 / c8 汇总、冷启动、逐会话 |
| [raw-dev-4configs.md](data/raw-dev-4configs.md) | 开发场景四套配置：c1 / c4 / c8 汇总、冷启动、逐会话 |
| [raw-canary-ab.md](data/raw-canary-ab.md) | 生产同时段 A/B：3 轮 × 两遍 |
| [gpu-sampling.md](data/gpu-sampling.md) | decode 各项测试期间的 GPU 采样 |
| [route-summary.md](data/route-summary.md) | 会话粘滞两轮对照 |
| [head8-tf.txt](data/head8-tf.txt) | int8 输出头 teacher forcing 逐条计数 |
| [kv-pool.md](data/kv-pool.md) | 各变体 KV 池容量 |
| [threeway-code-split.md](data/threeway-code-split.md) | 三方对比代码题 / 非代码题拆分 |
| [threeway-analysis.md](data/threeway-analysis.md)、[head8-analysis.md](data/head8-analysis.md)、[dev-analysis.md](data/dev-analysis.md)、[decode-analysis-all.md](data/decode-analysis-all.md)、[suffix-hybrid-summary.txt](data/suffix-hybrid-summary.txt)、[canary-ab-bench.log](data/canary-ab-bench.log) | 当时各轮汇总脚本的原样输出 |
