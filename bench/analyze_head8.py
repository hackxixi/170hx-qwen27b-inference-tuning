# analyze8.py — head8 汇总：速度表（两轮中位）+ gate 分歧/正确性 + c1/c4/c8 文本分歧参考
import json, os, statistics, sys
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
TAGS = [t for t in ("A", "D1", "D2", "D3") if os.path.exists(f"{R}/{t}-c1.json")]
def mval(lines, name):
    v = [float(l.rsplit(" ", 1)[1]) for l in lines if l.startswith(name + "{") or l.startswith(name + " ")]
    return sum(v) if v else None
def acc(r):
    b, a = r["metrics_before"], r["metrics_after"]
    d = lambda n: mval(a, n) - mval(b, n)
    dd = d("vllm:spec_decode_num_drafts_total"); da = d("vllm:spec_decode_num_accepted_tokens_total")
    return 1 + da / dd if dd else None
med = lambda x: statistics.median(x) if x else float("nan")
S = {}
for t in TAGS:
    for m in ("c1", "c4", "c8"):
        d = json.load(open(f"{R}/{t}-{m}.json"))
        for lang in ("en", "zh"):
            rs = [r for r in d["rounds"] if r["lang"] == lang]
            S[t, m, lang] = (med([r["per_stream_median"] for r in rs]), med([r["agg_tps"] for r in rs]), med([acc(r) for r in rs]), sum(r["errors"] for r in rs))
    for m in ("long4k", "long32k"):
        p = f"{R}/{t}-{m}.json"
        if not os.path.exists(p): continue
        rs = json.load(open(p))["rounds"]
        S[t, m, "zh"] = (med([r["reqs"][0]["decode_tps"] for r in rs]), None, med([acc(r) for r in rs]), sum(r["errors"] for r in rs))
keys = [(m, l) for m in ("c1", "c4", "c8") for l in ("en", "zh")] + [("long4k", "zh"), ("long32k", "zh")]
f = lambda v, n=1: "-" if v is None or v != v else f"{v:.{n}f}"
print("| 档位 | " + " | ".join(f"{t} 单流/每路 | {t} 聚合 | {t} 接受" for t in TAGS) + " | " + " | ".join(f"{t} vs A" for t in TAGS if t != "A") + " |")
for m, l in keys:
    cells = []
    for t in TAGS:
        v = S.get((t, m, l)); cells += [f(v[0]), f(v[1]), f(v[2], 2)] if v else ["-"] * 3
    dl = []
    for t in TAGS:
        if t == "A": continue
        a, v = S.get(("A", m, l)), S.get((t, m, l))
        if not (a and v): dl.append("-"); continue
        k = 1 if m == "c8" else 0  # c8 看聚合，其余看单流/每路
        dl.append(f"{(v[k] / a[k] - 1) * 100:+.1f}%")
    print(f"| {m} {l} | " + " | ".join(cells) + " | " + " | ".join(dl) + " |")
print("errors", {k: v[3] for k, v in S.items() if v[3]})

def gate(tag):
    p = f"{R}/{tag}-gate.json"
    return json.load(open(p)) if os.path.exists(p) else None
def cmp(x, y, name):
    X, Y = gate(x), gate(y)
    if not (X and Y): return
    div = []; same = 0
    for a, b in zip(X, Y):
        ta, tb = a["token_ids"], b["token_ids"]
        if ta == tb: same += 1; continue
        j = next((k for k in range(min(len(ta), len(tb))) if ta[k] != tb[k]), min(len(ta), len(tb)))
        # 参考侧（x）在分歧位的 top1-top2 差
        top = a["top"][j] if j < len(a["top"]) else []
        mg = top[0][1] - top[1][1] if len(top) >= 2 else None
        topy = b["top"][j] if j < len(b["top"]) else []
        mgy = topy[0][1] - topy[1][1] if len(topy) >= 2 else None
        div.append((a["lang"], a["i"], j, len(ta), mg, mgy))
    n = len(X)
    print(f"\n[{name}] {x} vs {y}: 全同 {same}/{n}，分歧 {len(div)}/{n}" +
          (f"，首分歧 token 位置均值 {statistics.mean(d[2] for d in div):.0f}（中位 {med([d[2] for d in div]):.0f}）" if div else ""))
    for d in div: print(f"   {d[0]}#{d[1]} 首分歧@{d[2]}/{d[3]}  {x} top1-top2={f(d[4], 3)}  {y} top1-top2={f(d[5], 3)}")
cmp("D2", "D2b", "D2 自复现"); cmp("A", "D2", "精度侧 A→D2"); cmp("A", "D1", "A→D1"); cmp("D2", "D3", "正确性硬门 D2→D3"); cmp("A", "D3", "A→D3")
for t in ("A-self", "D2-on-A"):
    p = f"{R}/{t}-tf.json"
    if os.path.exists(p): d = json.load(open(p)); print(f"\nteacher forcing {t}: top1 一致 {d['agree']}/{d['n']} = {d['agree']/d['n']:.4%}")
# 并发档文本（参考，非硬门）
def texts(t, m):
    d = json.load(open(f"{R}/{t}-{m}.json")); return [q.get("text") for r in d["rounds"] for q in r["reqs"]]
for x, y in (("D2", "D3"), ("A", "D2")):
    if x in TAGS and y in TAGS:
        print(f"\n文本全同（参考）{x} vs {y}: " + ", ".join(f"{m} {sum(a == b for a, b in zip(texts(x, m), texts(y, m)))}/{len(texts(x, m))}" for m in ("c1", "c4", "c8", "long4k", "long32k")))
