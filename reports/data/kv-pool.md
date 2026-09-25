# 各变体 GPU KV 池容量（启动日志）

| 代号 | KV | KV 池 token | 相对 max_model_len 262,144 |
|---|---|---:|---:|
| P | 基线 vLLM 0.28 bf16 | 556,584 | 2.12x |
| C1 | 组合版 vLLM 0.30 bf16 | 528,901 | 2.02x |
| KI8 | int8_per_token_head | 1,021,105 | 3.90x |
| KF8 | fp8 | 1,063,761 | 4.06x |
