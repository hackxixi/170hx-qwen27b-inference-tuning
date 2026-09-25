# bench

全部只依赖 Python 标准库，走 OpenAI 兼容接口。`<key>` 填 `-` 表示不带鉴权。

| 脚本 | 用途 |
|---|---|
| `abbench.py <url> <model> <key> <mode> <tag> <outdir>` | 通用测速。mode 可选 c1 / c4 / c8 / long4k / long32k / warm；中英各 8 条 prompt，贪心、关思考、max_tokens=900，每档 2 轮 |
| `gate.py <url> <model> <key> <tag> <outdir>` | 贪心输出逐 token 记录，含 top-2 logprobs（16 条短 prompt 加 1 条 long4k） |
| `tf.py <url> <model> <key> <ref_gate.json> <tag> <outdir>` | teacher forcing：把参考输出喂给被测服务，统计 top1 一致率 |
| `probe.py <url> <model> <key> <A_gate.json> <B_gate.json> <tag> <outdir>` | 在首个分歧点，用 prefill 路径让服务端重选一个 token，判断分歧是不是数值平局 |
| `gen_sessions.py` | 生成开发场景负载 `workload.json`，需要外部输入，见文件头注释 |
| `devbench.py <url> <model> <key> <mode> <tag> <outdir>` | 开发场景回放。mode 可选 smoke / record / pin / cold / c1 / c4 / c8；需要同目录下的 `workload.json`，record 模式会产出 `transcript.json` |
| `route/routebench.py` | 经过 litellm 回放会话，对比 shuffle 和 session_affinity；`an_route.py` 负责汇总 |
| `analyze_*.py` | 各轮结果的汇总脚本，输出即 `reports/data/` 下的表 |
| `raw_tables.py` | 把 abbench / devbench 的原始 json 和 nvidia-smi 采样整理成逐轮表（`reports/data/raw-*.md`、`gpu-sampling.md`） |

补充说明：
- long4k / long32k 需要自备 `long4k.txt`、`long32k.txt`，长度分别约 4.2k 和 33k token，放在脚本同目录。
- 汇总脚本里的代号（A、B、Cp、D2、P、C1 等）含义见各份报告。
- 贪心下判断「等价」的标准：分歧点上 top1-top2 差值 ≤0.125 nats（bf16 logprob 的分辨率），probe 选择大致对半，tf 一致率落在同配置自比的噪声底之内。
