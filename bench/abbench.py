# abbench.py — 通用 OpenAI 流式测速（A/B/C 三套共用）
# 用法: abbench.py <base_url> <model> <key|-> <mode> <tag> <outdir> [max_tokens]
#   mode: c1 | c4 | c8 | long4k | long32k | warm
# prompt 集：中英各 8 条，见下方 EN / ZH。
# 每条请求加确定性前缀 "[r<轮>-<序号>]"：三套相同、两轮不同，避免第二轮命中前缀缓存。
import json, os, sys, time, threading, urllib.request, statistics

EN = ["Write a Python function that parses an ISO-8601 duration string into seconds, with tests.",
      "Explain in detail how TCP congestion control works, covering slow start, AIMD and fast recovery.",
      "写一篇约800字的文章，介绍长江流域的地理、气候和主要城市。",
      "Implement a thread-safe LRU cache in Go with generics and explain the design.",
      "Summarize the causes and consequences of the 2008 financial crisis in a structured way.",
      "用 Rust 写一个命令行工具，统计目录下各类文件的行数，并解释关键代码。",
      "Describe how a transformer decoder works, step by step, including KV caching.",
      "Write a SQL schema for a library system and five non-trivial example queries."]
ZH = ["写一篇约800字的文章，介绍长江流域的地理、气候和主要城市。",
      "请详细解释 TCP 拥塞控制的原理，包括慢启动、AIMD 和快速恢复。",
      "用中文写一份新员工入职培训计划，分周列出目标、内容和考核方式。",
      "请比较 Python、Go 和 Rust 在后端开发中的优缺点，用中文详细回答。",
      "讲一个关于人工智能和一位老木匠的中文短篇故事，约600字。",
      "请用中文总结 2008 年金融危机的起因、过程和影响，分点论述。",
      "以产品经理的视角，用中文写一份智能家居 App 的需求文档大纲并解释每一部分。",
      "请用中文解释 Transformer 解码器的工作原理，包括 KV 缓存。"]

url, model, key, mode, tag, outdir = sys.argv[1:7]
mt = int(sys.argv[7]) if len(sys.argv) > 7 else 900
HERE = os.path.dirname(os.path.abspath(__file__))


def req(content, max_tokens):
    body = {"model": model, "messages": [{"role": "user", "content": content}], "max_tokens": max_tokens,
            "temperature": 0, "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    h = {"Content-Type": "application/json"}
    if key != "-":
        h["Authorization"] = "Bearer " + key
    r = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(), headers=h)
    t0 = time.time(); first = None; usage = {}; text = []; rtext = []; finish = None
    with urllib.request.urlopen(r, timeout=3600) as resp:
        for line in resp:
            line = line.decode().strip()
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            d = json.loads(line[5:])
            if d.get("usage"):
                usage = d["usage"]
            for ch in d.get("choices") or []:
                de = ch.get("delta") or {}
                c = de.get("content"); rc = de.get("reasoning_content") or de.get("reasoning")
                if (c or rc) and first is None:
                    first = time.time()
                if c: text.append(c)
                if rc: rtext.append(rc)
                if ch.get("finish_reason"): finish = ch["finish_reason"]
    t1 = time.time()
    n = usage.get("completion_tokens", 0)
    return {"t0": t0, "t1": t1, "ttft": (first or t1) - t0, "n": n, "prompt_tokens": usage.get("prompt_tokens"),
            "decode_tps": (n - 1) / (t1 - first) if first and n > 1 and t1 > first else 0.0,
            "finish": finish, "reasoning_chars": len("".join(rtext)), "usage": usage, "text": "".join(text)}


def metrics():
    try:
        h = {} if key == "-" else {"Authorization": "Bearer " + key}
        with urllib.request.urlopen(urllib.request.Request(url + "/metrics", headers=h), timeout=10) as r:
            return [l for l in r.read().decode().splitlines() if "spec" in l.lower() and not l.startswith("#")]
    except Exception as e:
        return ["ERR " + repr(e)]


def batch(prompts, rnd, max_tokens):
    res = [None] * len(prompts)
    def one(i):
        try:
            res[i] = req(f"[r{rnd}-{i}] " + prompts[i], max_tokens)
        except Exception as e:
            res[i] = {"error": repr(e)}
    T0 = time.time()
    ts = [threading.Thread(target=one, args=(i,)) for i in range(len(prompts))]
    [t.start() for t in ts]; [t.join() for t in ts]
    ok = [r for r in res if "error" not in r]
    wall = max(r["t1"] for r in ok) - T0 if ok else 0
    return {"wall": wall, "agg_tps": sum(r["n"] for r in ok) / wall if wall else 0,
            "per_stream_median": statistics.median([r["decode_tps"] for r in ok]) if ok else 0,
            "ttft_median": statistics.median([r["ttft"] for r in ok]) if ok else 0,
            "errors": len(res) - len(ok), "reqs": res}


def run():
    out = {"tag": tag, "mode": mode, "url": url, "max_tokens": mt, "rounds": []}
    sets = [("en", EN), ("zh", ZH)]
    if mode == "warm":
        for i in range(2):
            r = req(f"[warm-{i}] " + EN[i], 128); print("warm", i, r["n"], round(r["decode_tps"], 1), repr(r["text"][:80]))
        return
    if mode in ("c1", "c4", "c8"):
        for lang, P in sets:
            for rnd in (1, 2):
                m0 = metrics()
                if mode == "c1":  # 8 条顺序单流
                    rr = [batch([p], rnd, mt) for p in P]
                    reqs = [x["reqs"][0] for x in rr]
                    ok = [r for r in reqs if "error" not in r]
                    b = {"wall": sum(x["wall"] for x in rr), "agg_tps": sum(r["n"] for r in ok) / sum(x["wall"] for x in rr),
                         "per_stream_median": statistics.median([r["decode_tps"] for r in ok]),
                         "ttft_median": statistics.median([r["ttft"] for r in ok]), "errors": len(reqs) - len(ok), "reqs": reqs}
                else:
                    b = batch(P[: int(mode[1:])], rnd, mt)
                b.update({"lang": lang, "round": rnd, "metrics_before": m0, "metrics_after": metrics()})
                out["rounds"].append(b)
                print(f"{tag} {mode} {lang} r{rnd}: agg={b['agg_tps']:.1f} per_stream_med={b['per_stream_median']:.1f} "
                      f"ttft_med={b['ttft_median']:.2f}s err={b['errors']} "
                      f"reason_chars={sum(r.get('reasoning_chars', 0) for r in b['reqs'] if 'error' not in r)}", flush=True)
    else:  # long4k / long32k：单流，两轮不同前缀
        doc = open(os.path.join(HERE, f"{mode}.txt")).read()
        for rnd in (1, 2):
            m0 = metrics()
            b = batch([doc + "\n\n请用中文分点总结上面材料的主要内容，约 400 字。"], rnd, 512)
            b.update({"round": rnd, "metrics_before": m0, "metrics_after": metrics()})
            out["rounds"].append(b)
            r = b["reqs"][0]
            print(f"{tag} {mode} r{rnd}: prompt_tokens={r.get('prompt_tokens')} ttft={r.get('ttft', 0):.2f}s "
                  f"decode={r.get('decode_tps', 0):.1f} n={r.get('n')} err={b['errors']}", flush=True)
    os.makedirs(outdir, exist_ok=True)
    json.dump(out, open(os.path.join(outdir, f"{tag}-{mode}.json"), "w"), ensure_ascii=False, indent=1)


run()
