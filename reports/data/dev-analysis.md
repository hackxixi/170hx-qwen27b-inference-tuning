# 开发场景原始汇总（analyze_dev.py 输出）

代号：A = 现役（int4 输出头），D2 = int8 输出头，B = 170hx-fullstack 原方案，Cp = 推理栈 + HyperQwen 模型。

| 配置 | 档位 | 聚合 tok/s | 单轮 decode 中位 | TTFT 会话首轮 | TTFT 后续轮 | 命中率 首/后 | 冷 TTFT (pt/hit) | 会话 e2e 中位 s | 接受长度 | 截断 | 错误 |
|---|---|---:|---:|---:|---:|---|---|---:|---:|---:|---:|
| A | c1 | 88.1 | 120.1 | 2.27 | 0.73 | 0.86/0.97 | 13.51 (34528/12528) | 13.8 | 3.50 | 3 | 0 |
| A | c4 | 132.2 | 57.5 | 10.71 | 0.91 | 0.86/0.97 |  | 41.6 | 3.55 | 4 | 0 |
| A | c8 | 137.9 | 48.8 | 22.83 | 1.21 | 0.86/0.97 |  | 55.3 | 3.39 | 1 | 0 |
| D2 | c1 | 86.8 | 119.9 | 2.29 | 0.74 | 0.86/0.97 | 13.39 (34528/12528) | 14.6 | 3.51 | 3 | 0 |
| D2 | c4 | 125.4 | 64.2 | 10.81 | 0.93 | 0.86/0.97 |  | 42.3 | 3.46 | 3 | 0 |
| D2 | c8 | 136.0 | 55.8 | 23.05 | 1.34 | 0.86/0.97 |  | 56.0 | 3.40 | 3 | 0 |
| B | c1 | 92.0 | 114.4 | 2.18 | 0.51 | 0.88/0.99 | 14.25 (34892/13632) | 18.5 | 3.82 | 3 | 0 |
| B | c4 | 127.9 | 44.6 | 10.77 | 0.62 | 0.88/0.99 |  | 46.1 | 3.89 | 4 | 0 |
| B | c8 | 104.6 | 32.7 | 11.29 | 3.43 | 0.88/0.97 |  | 77.4 | 3.76 | 3 | 0 |
| Cp | c1 | 96.9 | 128.3 | 2.17 | 0.51 | 0.88/0.99 | 14.02 (34892/13632) | 13.4 | 3.80 | 5 | 0 |
| Cp | c4 | 124.4 | 60.5 | 8.76 | 0.72 | 0.88/0.99 |  | 41.9 | 3.96 | 5 | 0 |
| Cp | c8 | 117.0 | 49.1 | 17.44 | 2.05 | 0.88/0.97 |  | 71.0 | 3.79 | 4 | 0 |

工具调用： {"A": {"calls": 82, "name_ok": 82, "json_ok": 79, "turns_with_calls": 74, "a_tool_turns": 72, "agree": 72, "leak": 0, "extra_calls_vs_A": 2}, "D2": {"calls": 81, "name_ok": 81, "json_ok": 78, "turns_with_calls": 72, "a_tool_turns": 72, "agree": 72, "leak": 0, "extra_calls_vs_A": 0}, "B": {"calls": 82, "name_ok": 82, "json_ok": 82, "turns_with_calls": 72, "a_tool_turns": 72, "agree": 68, "leak": 0, "extra_calls_vs_A": 4}, "Cp": {"calls": 83, "name_ok": 83, "json_ok": 81, "turns_with_calls": 75, "a_tool_turns": 72, "agree": 71, "leak": 0, "extra_calls_vs_A": 4}}
代码块语法： {"A": {"blocks": 30, "ok": 27, "trunc": 3, "by": {"diff": [3, 3], "python": [11, 11], "bash": [16, 13]}}, "D2": {"blocks": 31, "ok": 26, "trunc": 2, "by": {"diff": [2, 2], "python": [15, 14], "bash": [14, 10]}}, "B": {"blocks": 19, "ok": 16, "trunc": 5, "by": {"diff": [3, 3], "python": [9, 9], "bash": [6, 3], "rust": [1, 1]}}, "Cp": {"blocks": 28, "ok": 23, "trunc": 5, "by": {"diff": [4, 4], "python": [10, 10], "bash": [14, 9]}}}
质量： {"A": {"ok": 45}, "D2": {"ok": 45}, "B": {"ok": 55}, "Cp": {"ok": 51}}
prompt_tokens 与 A 的差（min/med/max, 非零个数/总数）： {'D2': (0, 0, 0, 0, 105), 'B': (365, 365, 365, 105, 105), 'Cp': (365, 365, 365, 105, 105)}
pin A {"prefix_only": {"prompt_tokens": 23021, "cached_tokens": 22464, "ttft": 0.5117702484130859}, "pin_http": 404, "pin_body": "{\"detail\":\"Not Found\"}"}
pin D2 {"prefix_only": {"prompt_tokens": 23021, "cached_tokens": 0, "ttft": 11.36372184753418}, "pin_http": 404, "pin_body": "{\"detail\":\"Not Found\"}"}
pin B {"prefix_only": {"prompt_tokens": 23386, "cached_tokens": null, "ttft": 12.382720708847046}, "pin_http": 200, "pin_body": "{\"success\":true,\"pinned_nodes\":4,\"message\":\"\"}"}
pin Cp {"prefix_only": {"prompt_tokens": 23386, "cached_tokens": null, "ttft": 12.043039083480835}, "pin_http": 200, "pin_body": "{\"success\":true,\"pinned_nodes\":4,\"message\":\"\"}"}
