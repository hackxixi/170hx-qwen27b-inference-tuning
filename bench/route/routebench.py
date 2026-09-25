# routebench.py <url> <key> <tag> <conc> <outdir> [header] — 经临时 litellm 回放 dev transcript 的 8 条会话（每轮输入=之前全部消息）
# header=1 时每条会话带 x-litellm-session-id；记录每轮 TTFT、cached_tokens、服务副本（x-litellm-model-id）
import json, os, sys, time, threading, queue, copy, urllib.request
url, key, tag, conc, outdir = sys.argv[1:6]; conc = int(conc); hdr = len(sys.argv) > 6 and sys.argv[6] == "1"
HERE = os.path.dirname(os.path.abspath(__file__))
W = json.load(open(f"{HERE}/workload.json")); TR = json.load(open(f"{HERE}/transcript.json")); nonce = f"[route-{tag}-{int(time.time())}]"
def req(msgs, mt, sid):
    body = {"model": "qwen3.8-27b", "messages": [{"role": "system", "content": W["system"]}] + msgs, "tools": W["tools"],
            "tool_choice": "auto", "max_tokens": mt, "temperature": 0, "stream": True, "stream_options": {"include_usage": True}}
    h = {"Content-Type": "application/json", "Authorization": "Bearer " + key}
    if hdr: h["x-litellm-session-id"] = sid
    t0 = time.time(); fa = None; usage = {}
    with urllib.request.urlopen(urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(), headers=h), timeout=900) as r:
        dep = r.headers.get("x-litellm-model-id")
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line.endswith("[DONE]"): continue
            d = json.loads(line[5:])
            if d.get("usage"): usage = d["usage"]
            for ch in d.get("choices") or []:
                de = ch.get("delta") or {}
                if fa is None and (de.get("content") or de.get("tool_calls") or de.get("reasoning_content")): fa = time.time()
    t1 = time.time()
    return {"ttft": (fa or t1) - t0, "e2e": t1 - t0, "dep": dep, "pt": usage.get("prompt_tokens"), "n": usage.get("completion_tokens"),
            "cached": (usage.get("prompt_tokens_details") or {}).get("cached_tokens")}
q = queue.Queue(); [q.put(i) for i in range(len(TR))]; res = {}
def worker():
    while True:
        try: i = q.get_nowait()
        except queue.Empty: return
        s = TR[i]; sid = f"{nonce}-{s['id']}"; turns = []; ts = time.time()
        for k in range(len(s["gen_at"])):
            m = copy.deepcopy(s["messages"][: s["gen_at"][k]]); m[0]["content"] = m[0]["content"].replace("{NONCE}", nonce + s["id"])
            try: r = req(m, s["max_tokens"][k], sid)
            except Exception as e: r = {"error": repr(e)[:300]}
            r["k"] = k; turns.append(r); print(tag, s["id"], k, r, flush=True)
        res[s["id"]] = {"e2e": time.time() - ts, "turns": turns}
T0 = time.time(); th = [threading.Thread(target=worker) for _ in range(conc)]; [t.start() for t in th]; [t.join() for t in th]
json.dump({"tag": tag, "wall": time.time() - T0, "sessions": res}, open(f"{outdir}/route-{tag}.json", "w"), indent=1)
print("ROUTE_DONE", tag, round(time.time() - T0, 1))
