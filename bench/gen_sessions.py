# gen_sessions.py — 生成开发场景负载 workload.json
# 需要两个外部输入（不随本仓库分发）：
#   FULLSTACK_REPO  170hx-qwen3.8-27b-fullstack 的 checkout（取其 cc-warm 前缀与若干源码做会话素材）
#   HYPERQWEN_REPO  syv-ai/HyperQwen 的 checkout
# 用法: FULLSTACK_REPO=... HYPERQWEN_REPO=... python3 gen_sessions.py [out.json]
# 前缀：170hx-fs cc-warm 的 Claude Code system + 14 个工具（Anthropic→OpenAI 格式）+ 首条 user 的 deferred-tools/system-reminder 块
# 会话：8 条，每条是「输入步骤」序列；user 步骤必发，tool 步骤给出按工具名的模拟结果（录制时按 A 实际调用回填）
# 首条 user 消息里的 {NONCE} 放在共享前缀之后：不同档位会话内容不命中，system+tools 前缀照常命中
import json, os, re, sys
FS, HQ = os.environ["FULLSTACK_REPO"], os.environ["HYPERQWEN_REPO"]
OUT = sys.argv[1] if len(sys.argv) > 1 else "workload.json"

w = json.load(open(FS + "/sglang/cc-warm/warm-cc-prefix.json"))
system = "\n".join(b["text"] for b in w["system"])
tools = [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                           "parameters": {k: v for k, v in t["input_schema"].items() if k != "$schema"}}}
         for t in w["tools"]]
m0, m1 = w["messages"]
ctx = m0["content"] + "\n" + "".join(b["text"] for b in m1["content"] if b.get("type") == "text" and b["text"] != "你好")
TOOL_NAMES = [t["function"]["name"] for t in tools]


def rd(p): return open(p).read()
def catn(txt, start=1):  # Claude Code Read 工具的输出格式
    return "\n".join(f"{i:>6}\t{l}" for i, l in enumerate(txt.splitlines(), start))
def grep(root, pat, glob=None):
    out = []
    for dp, _, fs in os.walk(root):
        if "/.git" in dp: continue
        for f in sorted(fs):
            if glob and not re.search(glob, f): continue
            p = os.path.join(dp, f)
            try: lines = open(p, errors="strict").read().splitlines()
            except Exception: continue
            for i, l in enumerate(lines, 1):
                if re.search(pat, l): out.append(f"{p.replace(root, '/work/repo')}:{i}:{l[:200]}")
    return "\n".join(out[:60])


def tr(**kw):  # 工具结果表：键=工具名，_default 兜底
    d = {"Edit": "The file has been updated successfully.", "Write": "File created successfully.",
         "_default": "OK"}
    d.update(kw); return d


EVICT = rd(FS + "/sglang/patches/0008-v6-scaled-evictor/lru_file_evictor.py")
EVICT_BUG = EVICT.replace("target = max(0, int(self.max_size_bytes * self.eviction_ratio) - needed_bytes)",
                          "target = max(0, int(self.max_size_bytes * self.eviction_ratio) + needed_bytes)")
assert EVICT_BUG != EVICT
START = rd(HQ + "/single-user/start_qwen.sh")
KVCFG = rd(HQ + "/kvarn/files/vllm/model_executor/layers/quantization/kvarn/config.py")
P2 = rd(FS + "/gateway/src/policies/power_of_two.rs")
CB = rd(FS + "/gateway/src/core/circuit_breaker.rs")
CH = rd(FS + "/gateway/src/policies/consistent_hashing.rs")
PIN = rd(FS + "/sglang/patches/0005-ttl-tier4-pin.patch")
CARGO_FAIL = """   Compiling sgl-model-gateway v0.3.1 (/work/repo/gateway)
    Finished `test` profile [unoptimized + debuginfo] target(s) in 41.87s
     Running unittests src/lib.rs (target/debug/deps/sgl_model_gateway-4f1c2a9e0b7d3c11)

running 23 tests
test policies::consistent_hashing::tests::test_empty_workers ... ok
test policies::consistent_hashing::tests::test_same_key_same_worker ... ok
test policies::consistent_hashing::tests::test_distribution_is_balanced ... FAILED
test policies::consistent_hashing::tests::test_worker_removal_minimal_remap ... ok
test policies::power_of_two::tests::test_power_of_two_selection ... ok
test policies::power_of_two::tests::test_power_of_two_with_cached_loads ... ok

failures:

---- policies::consistent_hashing::tests::test_distribution_is_balanced stdout ----
thread 'policies::consistent_hashing::tests::test_distribution_is_balanced' panicked at src/policies/consistent_hashing.rs:489:9:
worker w3 got 612 of 4000 keys, expected between 800 and 1200 (virtual nodes = 1)
note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace

failures:
    policies::consistent_hashing::tests::test_distribution_is_balanced

test result: FAILED. 22 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.31s

error: test failed, to rerun pass `--lib`"""
CARGO_OK = """    Finished `test` profile [unoptimized + debuginfo] target(s) in 3.12s
     Running unittests src/lib.rs (target/debug/deps/sgl_model_gateway-4f1c2a9e0b7d3c11)

running 23 tests
test result: ok. 23 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.29s"""

sessions = [
 {"id": "s1-py-bugfix", "lang": "en", "steps": [
   {"user": "{NONCE} We run this LRU file evictor for the SGLang HiCache L3 tier. Since last week the cache directory keeps growing past "
            "max_size_bytes and never gets trimmed back to the eviction ratio, especially right after large (~78 MB) mamba state writes. "
            "Here is the module:\n\n```python\n" + EVICT_BUG + "\n```\n\nFind the bug and show the fix as a unified diff. Don't use tools, the file is above."},
   {"user": "Good. Now write pytest unit tests for _evict_locked that would have caught this bug. Use tmp_path and small fake entries; "
            "no real GPU or SGLang imports — stub whatever you need."},
   {"user": "Also add a `stats()` method that returns a dict with total_bytes, entry_count, max_size_bytes and the current eviction target. "
            "Output the complete method with type hints and a docstring."},
 ]},
 {"id": "s2-tool-grep-read", "lang": "en", "steps": [
   {"user": "{NONCE} In the HyperQwen repo (/work/repo), find where MTP_DRAFT_VOCAB is handled and explain exactly what it changes at runtime. "
            "Use the tools to search, don't guess."},
   {"tool": tr(Grep=grep(HQ, "MTP_DRAFT_VOCAB"), Glob="/work/repo/single-user/start_qwen.sh\n/work/repo/patches/qwen3_5-mtp-draft-vocab.patch",
               Bash=grep(HQ, "MTP_DRAFT_VOCAB"), Read=catn(rd(HQ + "/patches/qwen3_5-mtp-draft-vocab.patch")))},
   {"tool": tr(Read=catn(rd(HQ + "/patches/qwen3_5-mtp-draft-vocab.patch")), Grep=grep(HQ + "/single-user", "DRAFT"),
               Bash=catn(rd(HQ + "/patches/qwen3_5-mtp-draft-vocab.patch")))},
   {"tool": tr(Read=catn(START), Grep=grep(HQ + "/single-user", "VOCAB"), Bash="(no output)")},
   {"user": "Add validation in start_qwen.sh so the script exits with a clear error if MTP_DRAFT_VOCAB is set to anything other than 0 or 1. "
            "Make the edit."},
   {"tool": tr(Read=catn(START))},
   {"tool": tr()},
 ]},
 {"id": "s3-rust-feature", "lang": "en", "steps": [
   {"user": "{NONCE} Here is our power-of-two-choices load balancing policy (Rust, sgl-model-gateway):\n\n```rust\n" + P2 + "\n```\n\n"
            "Add support for weighted workers: each worker may expose a `weight()` (f32, default 1.0) and the policy should compare "
            "load/weight instead of raw load. Show the complete modified `select_worker` function and any new helper."},
   {"user": "Now write Rust unit tests for the weighted behaviour (add them to the existing tests module). Cover weight=0, equal loads, and a heavily weighted worker."},
   {"user": "Briefly: is the cached-load path racy under concurrent select_worker calls? Answer in <=8 bullet points."},
 ]},
 {"id": "s4-py-unittests", "lang": "en", "steps": [
   {"user": "{NONCE} Write a pytest test module for the following quantization config code. Mock out vllm imports so the tests run "
            "without a GPU. Focus on config parsing (from_config), get_quant_method dispatch, and validation errors.\n\n```python\n" + KVCFG + "\n```"},
   {"user": "Running your tests gives:\n\n```\nE   ModuleNotFoundError: No module named 'vllm.model_executor.layers.quantization.base_config'\n```\n\n"
            "The mocking happens after the import. Fix the test module so the stubs are installed before importing the code under test. Show the full corrected file."},
   {"user": "Add parametrized cases for invalid bit-widths and group sizes."},
 ]},
 {"id": "s5-zh-bash", "lang": "zh", "steps": [
   {"user": "{NONCE} 下面是我们单卡推理服务的启动脚本。请给它加一个 --dry-run 参数：打开时只打印最终要执行的 vllm serve 完整命令（每个参数一行，便于审阅），"
            "不真正启动。请直接给出修改后的相关片段和说明，不要调用工具。\n\n```bash\n" + START + "\n```"},
   {"user": "再加一个启动前的参数检查：MAX_SEQS 和 DRAFT_TOKENS 必须是正整数，GPU_UTIL 必须在 (0,1] 之间，不合法就用中文报错并退出码 2。给出完整的检查函数。"},
   {"user": "最后用中文分点总结一下这个脚本从读取环境变量到拉起服务的完整流程，不超过 15 点。"},
 ]},
 {"id": "s6-tool-bash-test", "lang": "en", "steps": [
   {"user": "{NONCE} The gateway crate at /work/repo/gateway has a failing unit test after my change to the consistent hashing policy. "
            "Run the policy tests, find the cause, and fix it."},
   {"tool": tr(Bash=CARGO_FAIL, Grep=grep(FS + "/gateway/src/policies", "virtual|fn test_distribution"), Read=catn(CH), Glob="/work/repo/gateway/src/policies/consistent_hashing.rs")},
   {"tool": tr(Read=catn(CH), Grep=grep(FS + "/gateway/src/policies", "virtual|VIRTUAL"), Bash=CARGO_FAIL)},
   {"tool": tr(Read=catn(CH), Bash=CARGO_OK)},
   {"user": "Run the tests again to confirm, then summarize what you changed and why."},
   {"tool": tr(Bash=CARGO_OK)},
 ]},
 {"id": "s7-review-patch", "lang": "en", "steps": [
   {"user": "{NONCE} Please review this SGLang patch that adds a permanent `pin_prefix` to the radix cache. Look for correctness and "
            "concurrency bugs (lock_ref accounting, evictable/protected size bookkeeping, what happens when the pinned chain is later split). "
            "Do not use tools.\n\n```diff\n" + PIN + "\n```"},
   {"user": "Write a corrected version of the dec_lock_ref pin-floor logic that fixes the problems you found. Output complete Python code for the changed function(s)."},
   {"user": "Propose a minimal unit test (pytest, with a fake tree of 3 nodes) that demonstrates the original bug."},
   {"user": "One more: does /admin/pin_prefix need a way to unpin? Answer in 5 lines max."},
 ]},
 {"id": "s8-zh-tool-rust", "lang": "zh", "steps": [
   {"user": "{NONCE} 网关里的熔断器（circuit breaker）在半开状态下好像会一下放行很多请求，把刚恢复的后端又打挂了。代码在 /work/repo/gateway 下面，"
            "先用工具找到相关代码看一下，再用中文告诉我原因。"},
   {"tool": tr(Grep=grep(FS + "/gateway/src/core", "HalfOpen|half_open"), Glob="/work/repo/gateway/src/core/circuit_breaker.rs",
               Read=catn(CB), Bash=grep(FS + "/gateway/src/core", "HalfOpen"))},
   {"tool": tr(Read=catn(CB), Grep=grep(FS + "/gateway/src/core", "can_execute"))},
   {"tool": tr(Read=catn(CB))},
   {"user": "请给出修复方案：半开状态最多同时放行 half_open_max_requests 个探测请求（新增配置项，默认 1）。给出修改后的完整 Rust 代码片段，并补充对应的单元测试。"},
   {"user": "这个改动对现有的 metrics 有影响吗？用中文简要说明。"},
 ]},
]

json.dump({"system": system, "tools": tools, "ctx": ctx, "tool_names": TOOL_NAMES, "sessions": sessions},
          open(OUT, "w"), ensure_ascii=False, indent=1)
print("ok", OUT, os.path.getsize(OUT), "sys chars", len(system), "ctx chars", len(ctx),
      {s["id"]: sum(len(json.dumps(x, ensure_ascii=False)) for x in s["steps"]) for s in sessions})
