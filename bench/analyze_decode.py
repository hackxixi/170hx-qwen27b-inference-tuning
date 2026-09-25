# analyze.py [变体...] — decode 汇总：通用负载（两轮中位）、开发场景、gate/probe/tf、GPU 采样；收益一律相对同场 P
import json, os, sys, statistics, glob, csv
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
med = lambda x: statistics.median(x) if x else float("nan")
def mval(lines, name):
    v = [float(l.rsplit(" ", 1)[1]) for l in lines if l.startswith(name + "{") or l.startswith(name + " ")]
    return sum(v) if v else None
def acc(r):
    try:
        b, a = r["metrics_before"], r["metrics_after"]
        dd = mval(a, "vllm:spec_decode_num_drafts_total") - mval(b, "vllm:spec_decode_num_drafts_total")
        da = mval(a, "vllm:spec_decode_num_accepted_tokens_total") - mval(b, "vllm:spec_decode_num_accepted_tokens_total")
        return 1 + da / dd if dd else None
    except Exception: return None
BASE = os.environ.get("BASE", "P")
VS = sys.argv[1:] or sorted({os.path.basename(p).split("-")[0] for p in glob.glob(f"{R}/*-c1.json")})
if BASE in VS: VS.remove(BASE)
VS = [BASE] + VS
f = lambda v, n=1: "-" if v is None or v != v else f"{v:.{n}f}"
S = {}
for t in VS:
    for m in ("c1", "c4", "c8"):
        p = f"{R}/{t}-{m}.json"
        if not os.path.exists(p): continue
        d = json.load(open(p))
        for lang in ("en", "zh"):
            rs = [r for r in d["rounds"] if r["lang"] == lang]
            S[t, m, lang] = (med([r["per_stream_median"] for r in rs]), med([r["agg_tps"] for r in rs]), med([x for x in (acc(r) for r in rs) if x]), sum(r["errors"] for r in rs))
    for m in ("long4k", "long32k"):
        p = f"{R}/{t}-{m}.json"
        if not os.path.exists(p): continue
        rs = json.load(open(p))["rounds"]
        S[t, m, "zh"] = (med([r["reqs"][0].get("decode_tps", 0) for r in rs]), med([r["reqs"][0].get("ttft", 0) for r in rs]), med([x for x in (acc(r) for r in rs) if x]), sum(r["errors"] for r in rs))
keys = [(m, l) for m in ("c1", "c4", "c8") for l in ("en", "zh")] + [("long4k", "zh"), ("long32k", "zh")]
print("## 通用负载（两轮中位；c1/c4/long 看单流/每路 decode tok/s，c8 看聚合；long 第二列为 TTFT s）\n")
print("| 档位 | " + " | ".join(f"{t} 速度 | {t} 聚合/TTFT | {t} 接受" for t in VS) + " | " + " | ".join(f"{t} vs {BASE}" for t in VS[1:]) + " |")
print("|" + "---|" * (1 + 3 * len(VS) + len(VS) - 1))
for m, l in keys:
    cells = []
    for t in VS:
        v = S.get((t, m, l)); cells += [f(v[0]), f(v[1], 2 if m.startswith("long") else 1), f(v[2], 2)] if v else ["-"] * 3
    dl = []
    for t in VS[1:]:
        a, v = S.get((BASE, m, l)), S.get((t, m, l))
        k = 1 if m == "c8" else 0
        dl.append(f"{(v[k] / a[k] - 1) * 100:+.1f}%" if a and v and a[k] else "-")
    print(f"| {m} {l} | " + " | ".join(cells) + " | " + " | ".join(dl) + " |")
errs = {f"{k[0]}-{k[1]}-{k[2]}": v[3] for k, v in S.items() if v[3]}
if errs: print("\nerrors", errs)
# 开发场景
print("\n## 开发场景（devbench 回放 transcript；decode 中位=非工具轮单轮 decode tok/s；TTFT 后续轮=k≥1 中位）\n")
print("| 变体 | 档位 | 聚合 tok/s | decode 中位 | TTFT 首轮 | TTFT 后续轮 | 会话 e2e 中位 | 接受 | 错误 | 聚合 vs P | decode vs P |")
print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
DV = {}
for t in VS:
    for c in ("c1", "c4", "c8"):
        p = f"{R}/{t}-dev{c}.json" if os.path.exists(f"{R}/{t}-dev{c}.json") else None
        # devbench 写 <tag>-<mode>.json，与 abbench 同名冲突 → suite 里 devbench 用 dc*，文件名为 <tag>-c*.json? 见下
        p = p or f"{R}/dev/{t}-{c}.json"
        if not os.path.exists(p): continue
        d = json.load(open(p)); T = [x for s in d["sessions"] for x in s["turns"] if "error" not in x]
        dt = [x["decode_tps"] for x in T if x.get("decode_tps")]
        DV[t, c] = (d["agg_tps"], med(dt), med([x["ttft"] for x in T if x["k"] == 0]), med([x["ttft"] for x in T if x["k"] >= 1]),
                    med([s["e2e"] for s in d["sessions"]]), acc(d), d["errors"])
for t in VS:
    for c in ("c1", "c4", "c8"):
        v = DV.get((t, c))
        if not v: continue
        b = DV.get((BASE, c))
        rel = lambda i: f"{(v[i] / b[i] - 1) * 100:+.1f}%" if b and b[i] and t != BASE else "-"
        print(f"| {t} | {c} | {f(v[0])} | {f(v[1])} | {f(v[2], 2)} | {f(v[3], 2)} | {f(v[4])} | {f(v[5], 2)} | {v[6]} | {rel(0)} | {rel(1)} |")
# gate
def gate(tag):
    p = f"{R}/{tag}-gate.json"; return json.load(open(p)) if os.path.exists(p) else None
def cmp(x, y):
    X, Y = gate(x), gate(y)
    if not (X and Y): return
    div = []; same = 0
    for a, b in zip(X, Y):
        ta, tb = a["token_ids"], b["token_ids"]
        if ta == tb: same += 1; continue
        j = next((k for k in range(min(len(ta), len(tb))) if ta[k] != tb[k]), min(len(ta), len(tb)))
        top = a["top"][j] if j < len(a["top"]) else []; mg = top[0][1] - top[1][1] if len(top) >= 2 else None
        topy = b["top"][j] if j < len(b["top"]) else []; mgy = topy[0][1] - topy[1][1] if len(topy) >= 2 else None
        div.append((a["lang"], a["i"], j, len(ta), mg, mgy))
    print(f"\n[gate] {x} vs {y}: 全同 {same}/{len(X)}" + (f"，首分歧位中位 {med([d[2] for d in div]):.0f}，" +
          "分歧处 top1-top2 差（{x}/{y}）: " + ", ".join(f"{d[0]}#{d[1]}@{d[2]}({f(d[4], 3)}/{f(d[5], 3)})" for d in div) if div else ""))
print("\n## 贪心逐 token 比对（gate：16 条 900 tok + long4k 512 tok）")
for t in VS[1:] + [BASE + "b"]:
    cmp(BASE, t)
for p in sorted(glob.glob(f"{R}/*-probe.json")):
    d = json.load(open(p)); from collections import Counter
    print(f"[probe] {os.path.basename(p)}: {len(d)} 个分歧位，prefill 路径选择 {dict(Counter(x['pick'] for x in d))}，差距中位 {f(med([x['margin'] for x in d if x['margin'] is not None]), 3)}")
for p in sorted(glob.glob(f"{R}/*-tf.json")):
    d = json.load(open(p)); print(f"[tf] {os.path.basename(p)}: top1 一致 {d['agree']}/{d['n']} = {d['agree']/max(d['n'],1):.4%}")
# GPU 采样
print("\n## GPU 采样（测试全程 500ms；功率 W / SM MHz / 显存 MHz；非空闲事件原因计数）")
for t in VS:
    p = f"{R}/{t}-smi.csv"
    if not os.path.exists(p): continue
    rows = [r for r in csv.reader(open(p)) if len(r) >= 7]
    busy = [r for r in rows if r[5].strip().rstrip(" %").isdigit() and int(r[5].strip().rstrip(" %")) > 50]
    num = lambda r, i: float(r[i].split()[0])
    if not busy: continue
    from collections import Counter
    rs = Counter(r[6].strip() for r in busy)
    print(f"{t}: 样本 {len(busy)}，功率中位 {med([num(r,3) for r in busy]):.0f} 最大 {max(num(r,3) for r in busy):.0f}，SM 中位 {med([num(r,1) for r in busy]):.0f} 最小 {min(num(r,1) for r in busy):.0f}，显存 {med([num(r,2) for r in busy]):.0f}，原因 {dict(rs.most_common(4))}")
