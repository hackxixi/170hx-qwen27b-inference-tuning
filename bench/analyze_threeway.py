# analyze.py — 汇总 results/*.json：两轮取中位数（2 个值的中位数 = 均值）+ 接受长度 + 输出质量检查
import json, os, re, statistics, sys
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
SYS = [("A", "A vLLM+HyperQwen MTP k=4"), ("B", "B SGLang+EffThink AWQ+DFLASH"), ("Cp", "C′ SGLang+HyperQwen(-fast,head/embed→bf16)+DFLASH")]
DRAFT = {"A": 4, "B": 8, "Cp": 8}


def mval(lines, name):
    tot = 0.0; hit = False
    for l in lines:
        if l.startswith(name + "{") or l.startswith(name + " "):
            tot += float(l.rsplit(" ", 1)[1]); hit = True
    return tot if hit else None


def accept(tag, rnd):
    b, a = rnd["metrics_before"], rnd["metrics_after"]
    ntok = sum(r["n"] for r in rnd["reqs"] if "error" not in r)
    if tag == "A":
        dd = mval(a, "vllm:spec_decode_num_drafts_total") - mval(b, "vllm:spec_decode_num_drafts_total")
        da = mval(a, "vllm:spec_decode_num_accepted_tokens_total") - mval(b, "vllm:spec_decode_num_accepted_tokens_total")
        dt = mval(a, "vllm:spec_decode_num_draft_tokens_total") - mval(b, "vllm:spec_decode_num_draft_tokens_total")
        return (1 + da / dd if dd else None), (da / dt if dt else None)
    dv = mval(a, "sglang:spec_verify_calls_total") - mval(b, "sglang:spec_verify_calls_total")
    L = ntok / dv if dv else None
    return L, ((L - 1) / (DRAFT[tag] - 1) if L else None)


def quality(text):
    t = text.strip()
    if not t: return "EMPTY"
    bad = sum(1 for c in t if c == "�" or (ord(c) < 32 and c not in "\n\t\r"))
    sh = [t[i:i + 20] for i in range(0, max(1, len(t) - 20), 5)]
    uniq = len(set(sh)) / len(sh) if sh else 1
    tail = t[-400:]; tsh = [tail[i:i + 20] for i in range(0, max(1, len(tail) - 20), 5)]
    tuniq = len(set(tsh)) / len(tsh) if tsh else 1
    flag = []
    if bad: flag.append(f"bad_chars={bad}")
    if uniq < 0.6 or tuniq < 0.5: flag.append(f"repetitive(uniq={uniq:.2f},tail={tuniq:.2f})")
    if re.search(r"(.)\1{15,}", t): flag.append("char_run")
    return ",".join(flag) or "ok"


def med(x): return statistics.median(x) if x else float("nan")


rows = []; qual = {}; samples = {}
for tag, name in SYS:
    for mode in ("c1", "c4", "c8"):
        p = os.path.join(R, f"{tag}-{mode}.json")
        if not os.path.exists(p): continue
        d = json.load(open(p))
        for lang in ("en", "zh"):
            rs = [r for r in d["rounds"] if r["lang"] == lang]
            acc = [accept(tag, r) for r in rs]
            rows.append((tag, mode, lang, med([r["per_stream_median"] for r in rs]), med([r["agg_tps"] for r in rs]),
                         med([r["ttft_median"] for r in rs]), med([a[0] for a in acc if a[0]]), med([a[1] for a in acc if a[1]]),
                         sum(r["errors"] for r in rs), sum(q.get("reasoning_chars", 0) for r in rs for q in r["reqs"] if "error" not in q),
                         [round(r["agg_tps"], 1) for r in rs]))
            for r in rs:
                for q in r["reqs"]:
                    if "error" in q: continue
                    k = quality(q["text"]); qual.setdefault(tag, {}).setdefault(k, 0); qual[tag][k] += 1
                    if k != "ok": samples.setdefault(tag, []).append((mode, lang, k, q["text"][-200:]))
    for mode in ("long4k", "long32k"):
        p = os.path.join(R, f"{tag}-{mode}.json")
        if not os.path.exists(p): continue
        d = json.load(open(p)); rs = d["rounds"]; q = [r["reqs"][0] for r in rs]
        acc = [accept(tag, r) for r in rs]
        rows.append((tag, mode, "zh", med([x["decode_tps"] for x in q]), None, med([x["ttft"] for x in q]),
                     med([a[0] for a in acc if a[0]]), med([a[1] for a in acc if a[1]]), sum(r["errors"] for r in rs),
                     sum(x.get("reasoning_chars", 0) for x in q), [x["prompt_tokens"] for x in q]))
        for x in q:
            k = quality(x["text"]); qual.setdefault(tag, {}).setdefault(k, 0); qual[tag][k] += 1

f = lambda v, n=1: "-" if v is None or v != v else f"{v:.{n}f}"
print("| 方案 | 档位 | 语言 | 单流/每路 tok/s（中位） | 聚合 tok/s | TTFT 中位 s | 接受长度 | 接受率 | 错误 | 推理字符 | 两轮聚合/输入 tok |")
print("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|")
for r in rows:
    print(f"| {r[0]} | {r[1]} | {r[2]} | {f(r[3])} | {f(r[4])} | {f(r[5], 2)} | {f(r[6], 2)} | {f(r[7], 2)} | {r[8]} | {r[9]} | {r[10]} |")
print("\n质量检查（每条请求的输出）：", json.dumps(qual, ensure_ascii=False))
for tag, ss in samples.items():
    for s in ss[:3]: print("  flag", tag, s[:3], repr(s[3][-120:]))
