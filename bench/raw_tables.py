# raw_tables.py — 把 abbench / devbench / nvidia-smi 的原始输出整理成逐轮 markdown 表（reports/data/raw-*.md 由它生成）
# 用法（作为模块）：
#   abbench_table(results_dir, tags, engine)   engine: {"A": ("vllm", 4), "B": ("sglang", 8)}，数字为每轮最多草稿数
#   devbench_table(results_dir, tags, engine)
#   smi_table(results_dir, tags)               读 <tag>-smi.csv（nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,
#                                              power.draw,temperature.gpu,utilization.gpu,clocks_event_reasons.active）
# 接受长度 = 每轮验证平均产出 token 数（含验证补上的 1 个）；接受率 = 被接受的草稿 token / 起草的草稿 token。
# vLLM 取 /metrics 的 spec_decode 计数差；SGLang 取 spec_verify_calls_total，接受率按 (L-1)/(block-1) 折算。
import csv, json, os, statistics

def med(x):
    x = [v for v in x if v is not None]
    return statistics.median(x) if x else None


def f(v, n=1):
    return "-" if v is None else f"{v:.{n}f}"


def mval(lines, name):
    v = [float(l.rsplit(" ", 1)[1]) for l in lines if l.startswith(name + "{") or l.startswith(name + " ")]
    return sum(v) if v else None


def accept(rnd, ntok, eng):
    kind, draft = eng
    b, a = rnd.get("metrics_before") or [], rnd.get("metrics_after") or []
    try:
        if kind == "vllm":
            dd = mval(a, "vllm:spec_decode_num_drafts_total") - mval(b, "vllm:spec_decode_num_drafts_total")
            da = mval(a, "vllm:spec_decode_num_accepted_tokens_total") - mval(b, "vllm:spec_decode_num_accepted_tokens_total")
            dt = mval(a, "vllm:spec_decode_num_draft_tokens_total") - mval(b, "vllm:spec_decode_num_draft_tokens_total")
            return (1 + da / dd if dd else None), (da / dt if dt else None)
        dv = mval(a, "sglang:spec_verify_calls_total") - mval(b, "sglang:spec_verify_calls_total")
        L = ntok / dv if dv else None
        return L, ((L - 1) / (draft - 1) if L else None)
    except TypeError:
        return None, None


def abbench_table(R, tags, engine, lang_names={"en": "英", "zh": "中"}):
    out = ["| 变体 | 档位 | 语言 | 轮 | 每路 decode 中位 tok/s | 聚合 tok/s | TTFT 中位 s | 输出 token | 接受长度 | 接受率 | 错误 |",
           "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for t in tags:
        for m in ("c1", "c4", "c8"):
            p = os.path.join(R, f"{t}-{m}.json")
            if not os.path.exists(p): continue
            d = json.load(open(p))
            for lang in ("en", "zh"):
                rs = [r for r in d["rounds"] if r["lang"] == lang]
                vals = []
                for r in rs:
                    ok = [q for q in r["reqs"] if "error" not in q]; nt = sum(q["n"] for q in ok)
                    L, a = accept(r, nt, engine[t])
                    vals.append((r["per_stream_median"], r["agg_tps"], r["ttft_median"], nt, L, a, r["errors"]))
                    out.append(f"| {t} | {m} | {lang_names[lang]} | r{r['round']} | {f(r['per_stream_median'])} | {f(r['agg_tps'])} | "
                               f"{f(r['ttft_median'], 2)} | {nt} | {f(L, 2)} | {f(a, 2)} | {r['errors']} |")
                col = lambda i: med([v[i] for v in vals if v[i] is not None])
                out.append(f"| {t} | {m} | {lang_names[lang]} | **中位** | **{f(col(0))}** | **{f(col(1))}** | **{f(col(2), 2)}** | "
                           f"{f(col(3), 0)} | **{f(col(4), 2)}** | **{f(col(5), 2)}** | {sum(v[6] for v in vals)} |")
    out += ["", "长输入（单流，max_tokens 512）：", "",
            "| 变体 | 档位 | 轮 | 输入 token | TTFT s | decode tok/s | 输出 token | 接受长度 | 接受率 | 错误 |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for t in tags:
        for m in ("long4k", "long32k"):
            p = os.path.join(R, f"{t}-{m}.json")
            if not os.path.exists(p): continue
            vals = []
            for r in json.load(open(p))["rounds"]:
                q = r["reqs"][0]
                if "error" in q:
                    out.append(f"| {t} | {m} | r{r['round']} | - | - | - | - | - | - | {r['errors']} |"); continue
                L, a = accept(r, q["n"], engine[t])
                vals.append((q["ttft"], q["decode_tps"], L, a))
                out.append(f"| {t} | {m} | r{r['round']} | {q['prompt_tokens']} | {f(q['ttft'], 2)} | {f(q['decode_tps'])} | {q['n']} | "
                           f"{f(L, 2)} | {f(a, 2)} | {r['errors']} |")
            col = lambda i: med([v[i] for v in vals if v[i] is not None])
            out.append(f"| {t} | {m} | **中位** | | **{f(col(0), 2)}** | **{f(col(1))}** | | **{f(col(2), 2)}** | **{f(col(3), 2)}** | |")
    return "\n".join(out)


def devbench_table(R, tags, engine):
    out = ["| 变体 | 档位 | 墙钟 s | 聚合 tok/s | 输出 token | 非工具轮 decode 中位 | TTFT 首轮中位 s | TTFT 后续轮中位 s | "
           "TTFT 后续轮均值 s | 命中率 首轮 / 后续（中位） | 会话耗时 中位 / 最大 s | 截断轮 | 接受长度 | 接受率 | 错误 |",
           "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|"]
    cold = ["| 变体 | 会话 | 输入 token | 命中 token | TTFT s | decode tok/s |", "|---|---|---:|---:|---:|---:|"]
    for t in tags:
        for m in ("c1", "c4", "c8"):
            p = os.path.join(R, f"{t}-{m}.json")
            if not os.path.exists(p): continue
            d = json.load(open(p)); T = [x for s in d["sessions"] for x in s["turns"] if "error" not in x]
            nt = sum(x["n"] for x in T); first = [x for x in T if x["k"] == 0]; later = [x for x in T if x["k"] > 0]
            L, a = accept(d, nt, engine[t])
            hit = lambda xs: med([(x["cached_tokens"] or 0) / x["prompt_tokens"] for x in xs])
            e2e = [s["e2e"] for s in d["sessions"]]
            out.append(f"| {t} | {m} | {f(d['wall'], 0)} | {f(d['agg_tps'])} | {nt} | {f(med([x['decode_tps'] for x in T if not x['tool_calls']]))} | "
                       f"{f(med([x['ttft'] for x in first]), 2)} | {f(med([x['ttft'] for x in later]), 2)} | "
                       f"{f(statistics.mean(x['ttft'] for x in later) if later else None, 2)} | {f(hit(first), 2)} / {f(hit(later), 2)} | {f(med(e2e))} / {f(max(e2e))} | "
                       f"{sum(1 for x in T if x['finish'] == 'length')} | {f(L, 2)} | {f(a, 2)} | {d['errors']} |")
        p = os.path.join(R, f"{t}-cold.json")
        if os.path.exists(p):
            for c in json.load(open(p)):
                cold.append(f"| {t} | {c.get('sid', '-')} | {c['prompt_tokens']} | {c['cached_tokens'] or 0} | {f(c['ttft'], 2)} | {f(c['decode_tps'])} |")
    return "\n".join(out) + "\n\n冷启动（system 开头加唯一 nonce，发会话首轮；每变体 2 次 × 2 条会话）：\n\n" + "\n".join(cold)


def devbench_sessions(R, tags):
    out = ["| 变体 | 档位 | 会话 | 轮数 | 会话耗时 s | 各轮 TTFT s | 各轮输出 token | 各轮输入 token / 命中 |", "|---|---|---|---:|---:|---|---|---|"]
    for t in tags:
        for m in ("c1", "c4", "c8"):
            p = os.path.join(R, f"{t}-{m}.json")
            if not os.path.exists(p): continue
            for s in json.load(open(p))["sessions"]:
                T = s["turns"]
                out.append(f"| {t} | {m} | {s['id']} | {len(T)} | {f(s['e2e'])} | " + " / ".join(f(x.get('ttft'), 2) for x in T) + " | "
                           + " / ".join(str(x.get("n", "-")) for x in T) + " | "
                           + " / ".join(f"{x.get('prompt_tokens')}:{x.get('cached_tokens') or 0}" for x in T) + " |")
    return "\n".join(out)


def smi_table(R, tags):
    out = ["| 变体 | 总样本 | 忙碌样本（利用率 >50%） | 功率中位 W | 功率 P90 W | 功率峰值 W | SM 时钟中位 MHz | SM 时钟最小 MHz | 显存时钟 MHz | "
           "温度中位 °C | 温度峰值 °C | SW Power Cap 占忙碌样本 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for t in tags:
        p = os.path.join(R, f"{t}-smi.csv")
        if not os.path.exists(p): continue
        rows = [r for r in csv.reader(open(p)) if len(r) >= 7]
        busy = [r for r in rows if r[5].strip().rstrip(" %").isdigit() and int(r[5].strip().rstrip(" %")) > 50]
        if not busy: continue
        num = lambda r, i: float(r[i].split()[0])
        pw = sorted(num(r, 3) for r in busy); tp = [num(r, 4) for r in busy]
        cap = sum(1 for r in busy if int(r[6].strip(), 16) & 0x4)
        out.append(f"| {t} | {len(rows)} | {len(busy)} | {med(pw):.0f} | {pw[int(len(pw) * 0.9) - 1]:.0f} | {max(pw):.0f} | "
                   f"{med([num(r, 1) for r in busy]):.0f} | {min(num(r, 1) for r in busy):.0f} | {med([num(r, 2) for r in busy]):.0f} | "
                   f"{med(tp):.0f} | {max(tp):.0f} | {cap / len(busy):.1%} |")
    return "\n".join(out)


def per_prompt_c1(R, tags, lang_names={"en": "英", "zh": "中"}):
    """c1 逐条：每条 prompt 两轮的 decode tok/s 与输出 token 数（下标对应 abbench.py 里 EN / ZH 的顺序）"""
    out = ["| 变体 | 语言 | 下标 | r1 decode | r2 decode | r1 输出 token | r2 输出 token | r1 TTFT s | r2 TTFT s | 结束原因 |",
           "|---|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for t in tags:
        p = os.path.join(R, f"{t}-c1.json")
        if not os.path.exists(p): continue
        d = json.load(open(p))
        for lang in ("en", "zh"):
            rs = sorted((r for r in d["rounds"] if r["lang"] == lang), key=lambda r: r["round"])
            for i in range(len(rs[0]["reqs"])):
                q = [r["reqs"][i] for r in rs]
                g = lambda k, n=1: [f(x.get(k), n) if isinstance(x.get(k), float) else str(x.get(k, "-")) for x in q]
                out.append(f"| {t} | {lang_names[lang]} | {i} | " + " | ".join(g("decode_tps")) + " | " + " | ".join(g("n")) + " | "
                           + " | ".join(g("ttft", 2)) + " | " + "/".join(str(x.get("finish")) for x in q) + " |")
    return "\n".join(out)
