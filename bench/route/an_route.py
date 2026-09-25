import json, statistics as st, sys
for r in ("r1", "r2"):
    for t in ("shuffle", "sticky"):
        d = json.load(open(f"{r}/route-{t}.json")); S = d["sessions"]
        T = [x for s in S.values() for x in s["turns"] if "error" not in x]
        err = sum(1 for s in S.values() for x in s["turns"] if "error" in x)
        later = [x for x in T if x["k"] >= 1]; sw = tot = 0
        for s in S.values():
            deps = [x.get("dep") for x in s["turns"]]
            for a, b in zip(deps, deps[1:]): tot += 1; sw += a != b
        hit = sum(x["cached"] for x in later) / sum(x["pt"] for x in later)
        print(f"| {r} | {t} | {sw}/{tot} | {st.median(x['ttft'] for x in later):.2f} | {st.mean(x['ttft'] for x in later):.2f} | {max(x['ttft'] for x in later):.1f} | {hit:.3f} | {st.median(s['e2e'] for s in S.values()):.1f} | {d['wall']:.0f} | {st.median(x['ttft'] for x in T if x['k']==0):.2f} | {err} |")
