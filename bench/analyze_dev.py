# analyze_dev.py — 汇总 results/<V>-{cold,pin,c1,c4,c8}.json
import json, os, re, ast, textwrap, statistics, subprocess, tempfile, sys
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
W = json.load(open(os.path.join(os.path.dirname(R), "workload.json")))
NAMES = set(W["tool_names"])
VS = [v for v in ("A", "D2", "D3", "B", "Cp") if os.path.exists(f"{R}/{v}-c1.json")]
DRAFT = {"A": 4, "D2": 4, "D3": 4, "B": 8, "Cp": 8}


def med(x):
    x = [v for v in x if v is not None]; return statistics.median(x) if x else None


def mval(lines, name):
    t = [float(l.rsplit(" ", 1)[1]) for l in lines if l.startswith(name + "{") or l.startswith(name + " ")]
    return sum(t) if t else None


def accept(v, d, ntok):
    b, a = d["metrics_before"], d["metrics_after"]
    if v in ("A", "D2", "D3"):
        dd = mval(a, "vllm:spec_decode_num_drafts_total") - mval(b, "vllm:spec_decode_num_drafts_total")
        da = mval(a, "vllm:spec_decode_num_accepted_tokens_total") - mval(b, "vllm:spec_decode_num_accepted_tokens_total")
        return 1 + da / dd if dd else None
    dv = (mval(a, "sglang:spec_verify_calls_total") or 0) - (mval(b, "sglang:spec_verify_calls_total") or 0)
    return ntok / dv if dv else None


def check_rust(code):
    for wrap in ("{}", "fn __w() {{\n{}\n}}", "impl __W {{\n{}\n}}", "mod __w {{\n{}\n}}"):
        with tempfile.NamedTemporaryFile("w", suffix=".rs", delete=False) as f: f.write(wrap.format(code)); p = f.name
        ok = subprocess.run(["rustfmt", "--edition", "2021", "--check", p], capture_output=True).returncode in (0, 1)
        os.unlink(p)
        if ok: return True
    return False


def check_block(lang, code):
    lang = lang.lower()
    if lang in ("python", "py"):
        try: ast.parse(textwrap.dedent(code)); return True
        except SyntaxError:
            try: ast.parse("class _W:\n" + textwrap.indent(textwrap.dedent(code), "    ")); return True
            except SyntaxError: return False
    if lang in ("rust", "rs"): return check_rust(code)
    if lang in ("bash", "sh", "shell"):
        return subprocess.run(["bash", "-n"], input=code, text=True, capture_output=True).returncode == 0
    if lang == "json":
        try: json.loads(code); return True
        except Exception: return False
    if lang in ("diff", "patch"):
        return all(re.match(r"^([ +\-@\\]|diff |index |--- |\+\+\+ |$)", l) for l in code.splitlines())
    return None  # 不检查的语言


def code_stats(text, finish):
    blocks = re.findall(r"```([\w+-]*)\n(.*?)```", text, re.S)
    truncated = text.count("```") % 2 == 1
    res = []
    for lang, code in blocks:
        r = check_block(lang, code)
        if r is not None: res.append((lang.lower(), r))
    return res, truncated and finish == "length"


def quality(t):
    t = t.strip()
    if not t: return None
    sh = [t[i:i + 20] for i in range(0, max(1, len(t) - 20), 5)]
    tail = t[-400:]; tsh = [tail[i:i + 20] for i in range(0, max(1, len(tail) - 20), 5)]
    fl = []
    if sum(1 for c in t if c == "\ufffd"): fl.append("bad_chars")
    if (len(set(sh)) / len(sh) < 0.6) or (len(set(tsh)) / len(tsh) < 0.5): fl.append("repetitive")
    return ",".join(fl) or "ok"


rows = []; tool = {}; code = {}; ptoks = {}; qual = {}; samples = {}
for v in VS:
    cold = json.load(open(f"{R}/{v}-cold.json")) if os.path.exists(f"{R}/{v}-cold.json") else []
    tool[v] = {"calls": 0, "name_ok": 0, "json_ok": 0, "turns_with_calls": 0, "a_tool_turns": 0, "agree": 0,
               "leak": 0, "extra_calls_vs_A": 0}
    code[v] = {"blocks": 0, "ok": 0, "trunc": 0, "by": {}}
    for mode in ("c1", "c4", "c8"):
        p = f"{R}/{v}-{mode}.json"
        if not os.path.exists(p): continue
        d = json.load(open(p)); T = [t for s in d["sessions"] for t in s["turns"] if "error" not in t]
        ntok = sum(t["n"] for t in T)
        first = [t for t in T if t["k"] == 0]; later = [t for t in T if t["k"] > 0]
        txt = [t for t in T if not t["tool_calls"]]
        rows.append({"v": v, "mode": mode, "agg": d["agg_tps"], "wall": d["wall"],
                     "dec_med": med([t["decode_tps"] for t in txt]),
                     "ttft_first": med([t["ttft"] for t in first]), "ttft_later": med([t["ttft"] for t in later]),
                     "hit_first": med([(t["cached_tokens"] or 0) / t["prompt_tokens"] for t in first]),
                     "hit_later": med([(t["cached_tokens"] or 0) / t["prompt_tokens"] for t in later]),
                     "e2e_med": med([s["e2e"] for s in d["sessions"]]), "acc": accept(v, d, ntok),
                     "err": d["errors"], "ntok": ntok, "len_cut": sum(1 for t in T if t["finish"] == "length"),
                     "cold": med([c["ttft"] for c in cold]) if mode == "c1" else None,
                     "cold_pt": med([c["prompt_tokens"] for c in cold]) if mode == "c1" else None,
                     "cold_hit": med([c["cached_tokens"] or 0 for c in cold]) if mode == "c1" else None})
        for s in d["sessions"]:
            for t in s["turns"]:
                if "error" in t: continue
                ptoks.setdefault((mode, s["id"], t["k"]), {})[v] = t["prompt_tokens"]
                tv = tool[v]
                if t["a_tool"]: tv["a_tool_turns"] += 1
                if t["tool_calls"]:
                    tv["turns_with_calls"] += 1; tv["agree"] += bool(t["a_tool"]); tv["extra_calls_vs_A"] += (not t["a_tool"])
                for c in t["tool_calls"]:
                    tv["calls"] += 1; tv["name_ok"] += c["name"] in NAMES
                    try: tv["json_ok"] += isinstance(json.loads(c["arguments"] or "{}"), dict)
                    except Exception: pass
                if re.search(r"<tool_call>|<function=|</parameter>", t["text"]): tv["leak"] += 1
                cs, tr = code_stats(t["text"], t["finish"]); code[v]["trunc"] += tr
                for lang, ok in cs:
                    code[v]["blocks"] += 1; code[v]["ok"] += ok
                    b = code[v]["by"].setdefault(lang, [0, 0]); b[0] += 1; b[1] += ok
                q = quality(t["text"])
                if q: qual.setdefault(v, {}).setdefault(q, 0); qual[v][q] += 1
                if mode == "c1" and q: samples.setdefault(v, []).append((s["id"], t["k"], t["text"]))

f = lambda x, n=1: "-" if x is None else f"{x:.{n}f}"
print("| 配置 | 档位 | 聚合 tok/s | 单轮 decode 中位 | TTFT 会话首轮 | TTFT 后续轮 | 命中率 首/后 | 冷 TTFT (pt/hit) | 会话 e2e 中位 s | 接受长度 | 截断 | 错误 |")
print("|---|---|---:|---:|---:|---:|---|---|---:|---:|---:|---:|")
for r in rows:
    cold = f"{f(r['cold'], 2)} ({f(r['cold_pt'], 0)}/{f(r['cold_hit'], 0)})" if r["cold"] is not None else ""
    print(f"| {r['v']} | {r['mode']} | {f(r['agg'])} | {f(r['dec_med'])} | {f(r['ttft_first'], 2)} | {f(r['ttft_later'], 2)} | "
          f"{f(r['hit_first'], 2)}/{f(r['hit_later'], 2)} | {cold} | {f(r['e2e_med'])} | {f(r['acc'], 2)} | {r['len_cut']} | {r['err']} |")
print("\n工具调用：", json.dumps(tool, ensure_ascii=False))
print("代码块语法：", json.dumps(code, ensure_ascii=False))
print("质量：", json.dumps(qual, ensure_ascii=False))
# 输入一致性
diff = {}
for key, m in ptoks.items():
    for v in m:
        if v != "A" and "A" in m: diff.setdefault(v, []).append(m[v] - m["A"])
print("prompt_tokens 与 A 的差（min/med/max, 非零个数/总数）：",
      {v: (min(x), med(x), max(x), sum(1 for y in x if y), len(x)) for v, x in diff.items()})
for v in VS:
    pins = f"{R}/{v}-pin.json"
    if os.path.exists(pins): print("pin", v, json.dumps(json.load(open(pins)), ensure_ascii=False)[:300])
if "--samples" in sys.argv:
    import random; random.seed(1)
    for v, ss in samples.items():
        for sid, k, t in random.sample(ss, min(3, len(ss))):
            print(f"\n===== {v} {sid} k{k} len={len(t)}\n{t[:600]}\n...\n{t[-400:]}")
