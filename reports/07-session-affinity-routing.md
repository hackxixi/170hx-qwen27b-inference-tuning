# 07 网关会话粘滞路由（2026-09-25，已上线）

**结论：** litellm 默认的 simple-shuffle 让多轮会话的每一轮随机落到不同副本，前缀缓存白白失效。打开 `session_affinity` 以后，同一会话固定在一个副本上。两副本实测：后续轮 TTFT 均值 9.46 → 1.45 s、4.37 → 2.45 s，会话耗时中位 100 → 37 s、89 → 53 s，换副本次数从 13–16/27 降到 0/27。推算到生产的 7 副本，每个会话约少等 7.3 s。不改变模型输出，已上线。

## 1. 问题
- 生产是 7 个单卡副本，前面是 litellm 网关，`routing_strategy: simple-shuffle`。
- 为什么是 simple-shuffle：最初用的是 least-busy，实测有缺陷。并列时它总是选列表里的第一个副本，而「正在处理的请求数」要等请求真正发出才加一，并发突发时整批请求会压到同一个副本上；当时实测 82% 的流量去了第一个副本。改成 simple-shuffle 后分布均匀了，代价就是这份报告要解决的问题：多轮会话每一轮都可能换副本。
- vLLM 的前缀缓存只在副本内有效。开发场景（Claude Code 这类 agent 客户端）每轮都带着之前的全部消息，只要换了副本，就要把会话尾部整个重新预填充一遍。23K 的共享 system + tools 前缀在每个副本上都是热的，损失的是会话自身的那部分。

## 2. 测试方法
- **网关**：临时起一个 litellm 1.97.0 容器，只挂两个测试副本（同一套模型与参数），两份配置分别是 shuffle 和 sticky，见 [`deploy/litellm/`](../deploy/litellm/)（`*-2replica-test.yaml`）。
- **负载**：回放 [03](03-dev-workload.md) 开发场景录下的 transcript，8 条会话共 35 轮，每轮输入是 transcript 中这一轮之前的全部消息，`max_tokens` 1500，贪心。
- **并发**：4 个 worker 线程，每个线程顺序跑完一条会话再取下一条。实测「会话耗时之和 / 墙钟」为 3.2–3.4，与 4 并发加上尾部排空相符。
- **会话 id**：sticky 组每条会话带 `x-litellm-session-id`；shuffle 组不带。
- **记录**：每轮的 TTFT、`cached_tokens`、服务副本（响应头 `x-litellm-model-id`）。
- **两轮，顺序相反**：第 1 轮先 shuffle 后 sticky，第 2 轮先 sticky 后 shuffle，排除「后跑的一方吃到前一方留下的缓存」。第 2 轮的 shuffle 曾因外部供电中断，在副本重启后重跑。
- 共 4 组 × 35 轮 = 140 个请求，0 错误。
- 脚本：[`bench/route/routebench.py`](../bench/route/routebench.py)、[`bench/route/an_route.py`](../bench/route/an_route.py)。

## 3. 结果

| 轮次 | 路由 | 换副本次数 | 后续轮前缀命中率（token 加权） | 后续轮 TTFT 中位 s | 后续轮 TTFT 均值 s | 后续轮 TTFT 最大 s | 首轮 TTFT 中位 s | 会话耗时中位 s | 墙钟 s | 错误 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 第 1 轮 | shuffle | 13 / 27 | 0.882 | 1.73 | **9.46** | 44.5 | 21.74 | **100.2** | 222 | 0 |
| 第 1 轮 | sticky | 0 / 27 | 0.944 | 0.95 | **1.45** | 5.7 | 7.58 | **36.7** | 96 | 0 |
| 第 2 轮 | shuffle | 16 / 27 | 0.885 | 2.46 | **4.37** | 17.3 | 16.48 | **89.0** | 193 | 0 |
| 第 2 轮 | sticky | 0 / 27 | 0.944 | 1.07 | **2.45** | 19.0 | 6.76 | **53.3** | 131 | 0 |

相对 shuffle 的变化：

| 指标 | 第 1 轮 | 第 2 轮 |
|---|---:|---:|
| 后续轮 TTFT 均值 | 9.46 → 1.45 s（−85%） | 4.37 → 2.45 s（−44%） |
| 后续轮 TTFT 中位 | 1.73 → 0.95 s（−45%） | 2.46 → 1.07 s（−57%） |
| 会话耗时中位 | 100.2 → 36.7 s（−63%） | 89.0 → 53.3 s（−40%） |
| 墙钟 | 222 → 96 s（−57%） | 193 → 131 s（−32%） |
| 后续轮前缀命中率 | 0.882 → 0.944 | 0.885 → 0.944 |

- 「27」是 8 条会话里相邻两轮之间的切换机会数（35 − 8）。两副本下随机分发的期望换副本次数是 13.5，实测 13 和 16。
- 命中率 0.88 → 0.94 看着差得不多，因为分母里 23K 的共享前缀两边都能命中；差别集中在会话尾部，而尾部正好是每轮要预填充的部分。
- shuffle 的均值远高于中位数：长会话一旦换副本，要重算上万 token，个别轮 TTFT 到 17–45 s。
- 首轮 TTFT 也有差异。首轮本身两边都要预填充会话首条消息，差异应来自排队：shuffle 下后续轮的长预填充更多，同时到达的首轮请求排在它们后面（推测）。

## 4. 推算到 7 副本
- 输入（取自基线的开发场景 c1）：后续轮会话尾部（共享前缀之外、需要重算的部分）中位 4,454 token、最大 20,564 token；单卡预填充约 1,781 tok/s；假设 23K 共享前缀在每个副本上都是热的。
- sticky：后续轮只需预填充本轮新增部分，TTFT 期望约 1.17 s。
- shuffle：7 个副本里有 6/7 的概率落到没有这条会话缓存的副本，要额外重算会话尾部，4,454 / 1,781 ≈ 2.5 s，于是期望约 1.17 + 6/7 × 2.5 ≈ 3.3 s（原始推算记录为 3.35 s）。
- 每轮差约 2.2 s，每条会话平均 3.4 个后续轮，合计**每会话多等约 7.3 s**。真实的长会话尾部比测试负载长得多，差距会随之放大。

## 5. 配置
并入 litellm `config.yaml` 的 `router_settings`，其余不变（片段见 [`deploy/litellm/session-affinity.snippet.yaml`](../deploy/litellm/session-affinity.snippet.yaml)）：

    router_settings:
      routing_strategy: simple-shuffle
      optional_pre_call_checks: ["session_affinity"]
      deployment_affinity_ttl_seconds: 3600

- `optional_pre_call_checks: ["session_affinity"]` 是全局写法，已实测。litellm 也支持按模型组配置（`model_group_affinity_config`），本轮没有实测。
- `deployment_affinity_ttl_seconds: 3600`：会话到副本的映射保留 1 小时，超时后重新随机分配。

## 6. 上线后的实测结论

**小结**：会话 id 的三种来源都生效——`x-litellm-session-id`；`x-*-session-id`（值要像 UUID、至少 8 位）；Anthropic 的 `metadata.user_id`。OpenAI 接口和 Anthropic 接口都已实测生效。逐项如下：

- `x-litellm-session-id`：生效。
- `x-<客户端>-session-id`：生效，例如 Claude Code 自动带的 `x-claude-code-session-id`。值要像 UUID，长度至少 8 位；像 `test-E` 这样的短值不会被识别，请求仍按 shuffle 分发。
- OpenAI 接口（`/v1/chat/completions`）和 Anthropic 接口（`/v1/messages`）都生效。
- Anthropic 请求体里的 `metadata.user_id`：**生效**。复测时带同一个 `metadata.user_id` 的 8 个请求全部落到同一个副本；不带任何会话 id 时 6 个请求分到了 6 个不同副本。
  - 这一项先后测过两次，第一次误判为「不生效」：Anthropic 接口的响应里没有 `x-litellm-model-api-base` 响应头，第一次按这个头判断落在哪个副本，拿不到值。改看 `x-litellm-model-id` 后确认生效。
- 不带会话 id 的请求仍按 simple-shuffle 分发，行为和改配置前一样。

### 6.1 上线过程
1. 在网关 `config.yaml` 的 `router_settings` 里加上 §5 的两行，改前备份原文件。
2. 重启网关，约 31 秒恢复服务。
3. 按 §6 逐项实测会话 id 的识别情况，判断依据是响应头 `x-litellm-model-id`（它标明这次请求由哪个副本处理）。
4. 回滚：用备份的 `config.yaml` 替换后重启网关。

## 7. 没有采用的方案
- **`deployment_affinity`**：按 API key 粘滞，而不是按会话。生产上多个客户端共用一个 key，所有流量会压到同一个副本上，其余副本空转。
- **`prompt_caching` 预检**：依赖请求里的 `cache_control` 标记来算缓存键，而键会随轮次变化（每轮追加消息后前缀哈希就变了），对「同一会话粘在同一副本」这个目标不适用；OpenAI 格式的客户端也不带 `cache_control`。

## 8. 对输出的影响
路由只决定请求落到哪个副本，所有副本跑同一套模型与参数，不改变输出。可能的副作用是负载不均：一个很长的会话会一直占着同一个副本。本轮没有专门测量上线后的副本负载分布。
