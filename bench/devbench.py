# devbench.py — 开发/agent 场景测速（多套配置共用，建议在推理节点本机跑）
# 用法: devbench.py <base_url> <model> <key|-> <mode> <tag> <outdir>
#   mode: smoke | record | pin | cold | c1 | c4 | c8
#   record：只在 A 上跑一次，产出 transcript.json（A 的回复作为全体配置的历史回填）
#   c1/c4/c8：按 transcript 回放；N 路并发会话池跑完全部 8 条会话；每轮输入 = transcript 中该轮之前的全部消息
# {NONCE} 在首条 user 消息的共享前缀（system+tools+ctx）之后：各档位会话内容互不命中，前缀照常命中
import json, os, sys, time, threading, urllib.request, queue, copy

url, model, key, mode, tag, outdir = sys.argv[1:7]
HERE = os.path.dirname(os.path.abspath(__file__))
W = json.load(open(os.path.join(HERE, "workload.json")))
TR_PATH = os.path.join(HERE, "transcript.json")
os.makedirs(outdir, exist_ok=True)
H = {"Content-Type": "application/json"}
if key != "-": H["Authorization"] = "Bearer " + key


def req(messages, max_tokens, system=None):
    body = {"model": model, "messages": [{"role": "system", "content": system or W["system"]}] + messages,
            "tools": W["tools"], "tool_choice": "auto", "max_tokens": max_tokens, "temperature": 0,
            "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    r = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(), headers=H)
    t0 = time.time(); fa = fc = ft = None; usage = {}; text = []; rtext = []; finish = None; tcs = {}
    try: resp = urllib.request.urlopen(r, timeout=3600)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: {e.read().decode()[:500]}")
    with resp:
        for line in resp:
            line = line.decode().strip()
            if not line.startswith("data:") or line.endswith("[DONE]"): continue
            d = json.loads(line[5:])
            if d.get("usage"): usage = d["usage"]
            for ch in d.get("choices") or []:
                de = ch.get("delta") or {}
                c = de.get("content"); rc = de.get("reasoning_content") or de.get("reasoning")
                now = time.time()
                if c:
                    fc = fc or now; text.append(c)
                if rc: rtext.append(rc)
                for tc in de.get("tool_calls") or []:
                    ft = ft or now
                    e = tcs.setdefault(tc.get("index", 0), {"id": None, "name": "", "arguments": ""})
                    if tc.get("id"): e["id"] = tc["id"]
                    fn = tc.get("function") or {}
                    if fn.get("name"): e["name"] += fn["name"]
                    if fn.get("arguments"): e["arguments"] += fn["arguments"]
                if (c or rc or de.get("tool_calls")) and fa is None: fa = now
                if ch.get("finish_reason"): finish = ch["finish_reason"]
    t1 = time.time(); n = usage.get("completion_tokens", 0)
    ptd = usage.get("prompt_tokens_details") or {}
    return {"t0": t0, "t1": t1, "ttft": (fa or t1) - t0, "ttft_content": (fc - t0) if fc else None,
            "ttft_tool": (ft - t0) if ft else None, "e2e": t1 - t0, "n": n,
            "prompt_tokens": usage.get("prompt_tokens"), "cached_tokens": ptd.get("cached_tokens"),
            "decode_tps": (n - 1) / (t1 - fc) if fc and n > 1 and t1 > fc and not tcs else None,
            "finish": finish, "reasoning_chars": len("".join(rtext)), "text": "".join(text),
            "tool_calls": [tcs[k] for k in sorted(tcs)]}


def metrics():
    try:
        with urllib.request.urlopen(urllib.request.Request(url + "/metrics", headers=H), timeout=10) as r:
            return [l for l in r.read().decode().splitlines() if "spec" in l.lower() and not l.startswith("#")]
    except Exception as e:
        return ["ERR " + repr(e)]


def first_user(step_text, nonce):
    return W["ctx"] + "\n" + step_text.replace("{NONCE}", nonce)


def tool_msgs(calls, tmap, sid, turn):
    out = []
    for j, c in enumerate(calls):
        out.append({"role": "tool", "tool_call_id": c["id"], "content": tmap.get(c["name"], tmap.get("_default", "OK"))})
    return out


def asst_msg(r, sid, turn):
    m = {"role": "assistant", "content": r["text"] or None}
    if r["tool_calls"]:
        m["tool_calls"] = [{"id": c["id"] or f"call_{sid}_{turn}_{j}", "type": "function",
                            "function": {"name": c["name"], "arguments": c["arguments"]}} for j, c in enumerate(r["tool_calls"])]
        for c, mc in zip(r["tool_calls"], m["tool_calls"]): c["id"] = mc["id"]
    return m


def record():
    """在 A 上按脚本逐步生成；A 的回复（含 tool_calls）进历史；tool 步骤仅在上一轮 A 调了工具时生效。"""
    out = []
    for s in W["sessions"]:
        msgs = []; gen_at = []; maxt = []; atool = []; last = None; tmap = {"_default": "OK"}; turn = 0
        for st in s["steps"]:
            if "user" in st:
                if last and last["tool_calls"]:  # 挂起的工具调用：静默回填上一份工具表，不额外生成
                    msgs += tool_msgs(last["tool_calls"], tmap, s["id"], turn)
                txt = first_user(st["user"], "{NONCE}") if not msgs else st["user"]
                msgs.append({"role": "user", "content": txt})
            else:
                if not (last and last["tool_calls"]): continue
                tmap = st["tool"]; msgs += tool_msgs(last["tool_calls"], tmap, s["id"], turn)
            gen_at.append(len(msgs))
            r = req([dict(m, content=m["content"].replace("{NONCE}", "[rec]")) if m["role"] == "user" and m is msgs[0] else m for m in msgs], 4096)
            for c in r["tool_calls"]:
                try: json.loads(c["arguments"] or "{}")
                except Exception: print("  bad args -> {}", c["name"], c["arguments"][:80]); c["arguments"] = "{}"
            last = r; turn += 1
            maxt.append(1500 if r["tool_calls"] else 1500)  # 回放统一 1500；a_tool 另记
            atool.append(bool(r["tool_calls"]))
            msgs.append(asst_msg(r, s["id"], turn))
            print(f"rec {s['id']} t{turn} pt={r['prompt_tokens']} cached={r['cached_tokens']} n={r['n']} fin={r['finish']} "
                  f"tools={[c['name'] for c in r['tool_calls']]} {r['text'][:70]!r}", flush=True)
        out.append({"id": s["id"], "lang": s["lang"], "messages": msgs, "gen_at": gen_at, "max_tokens": maxt, "a_tool": atool})
    json.dump(out, open(TR_PATH + ".tmp", "w"), ensure_ascii=False, indent=1); os.replace(TR_PATH + ".tmp", TR_PATH)
    print("transcript saved", TR_PATH, sum(len(x["gen_at"]) for x in out), "turns")


def session_msgs(sess, k, nonce):
    m = copy.deepcopy(sess["messages"][: sess["gen_at"][k]])
    m[0]["content"] = m[0]["content"].replace("{NONCE}", nonce)
    return m


def replay(conc):
    TR = json.load(open(TR_PATH)); nonce = f"[{mode}]"
    q = queue.Queue(); [q.put(i) for i in range(len(TR))]
    res = {}
    def worker():
        while True:
            try: i = q.get_nowait()
            except queue.Empty: return
            s = TR[i]; turns = []; ts = time.time()
            for k in range(len(s["gen_at"])):
                try: r = req(session_msgs(s, k, nonce), s["max_tokens"][k])
                except Exception as e: r = {"error": repr(e)}
                r["k"] = k; r["a_tool"] = s["a_tool"][k]; turns.append(r)
            res[s["id"]] = {"id": s["id"], "lang": s["lang"], "e2e": time.time() - ts, "turns": turns}
    m0 = metrics(); T0 = time.time()
    th = [threading.Thread(target=worker) for _ in range(conc)]; [t.start() for t in th]; [t.join() for t in th]
    wall = time.time() - T0; m1 = metrics()
    allt = [t for v in res.values() for t in v["turns"] if "error" not in t]
    out = {"tag": tag, "mode": mode, "conc": conc, "wall": wall, "agg_tps": sum(t["n"] for t in allt) / wall,
           "errors": sum(1 for v in res.values() for t in v["turns"] if "error" in t),
           "metrics_before": m0, "metrics_after": m1, "sessions": [res[s["id"]] for s in TR if s["id"] in res]}
    json.dump(out, open(os.path.join(outdir, f"{tag}-{mode}.json"), "w"), ensure_ascii=False, indent=1)
    dt = sorted(t["decode_tps"] for t in allt if t["decode_tps"])
    print(f"{tag} {mode}: wall={wall:.1f}s agg={out['agg_tps']:.1f} tok/s turns={len(allt)} err={out['errors']} "
          f"decode_med={dt[len(dt)//2] if dt else 0:.1f}", flush=True)


def cold():
    """首轮冷启动：system 开头加唯一 nonce → system+tools+ctx+会话内容全部不命中；与 c1 首轮（前缀命中）对照"""
    TR = json.load(open(TR_PATH)); out = []
    for i in range(2):
        for sid in (0, 4):
            s = TR[sid]
            r = req(session_msgs(s, 0, "[cold]"), s["max_tokens"][0], system=f"[cold-{tag}-{i}-{sid}] " + W["system"])
            r["sid"] = s["id"]; out.append(r)
            print(f"{tag} cold {s['id']} pt={r['prompt_tokens']} cached={r['cached_tokens']} ttft={r['ttft']:.2f}s n={r['n']}", flush=True)
    json.dump(out, open(os.path.join(outdir, f"{tag}-cold.json"), "w"), ensure_ascii=False, indent=1)


def pin():
    """只含共享前缀（system+tools+ctx）的请求 → /admin/pin_prefix（仅 SGLang 补丁 0005 有此端点）"""
    r = req([{"role": "user", "content": W["ctx"] + "\nhi"}], 1)
    print(f"{tag} prefix-only pt={r['prompt_tokens']} cached={r['cached_tokens']}")
    try:
        rq = urllib.request.Request(url + "/admin/pin_prefix", data=b"{}", headers=H, method="POST")
        with urllib.request.urlopen(rq, timeout=30) as x: body = x.read().decode(); code = x.status
    except urllib.error.HTTPError as e: body = e.read().decode(); code = e.code
    except Exception as e: body = repr(e); code = None
    print(f"{tag} pin_prefix http={code} {body[:300]}")
    json.dump({"prefix_only": {k: r[k] for k in ("prompt_tokens", "cached_tokens", "ttft")}, "pin_http": code, "pin_body": body},
              open(os.path.join(outdir, f"{tag}-pin.json"), "w"), ensure_ascii=False, indent=1)


def smoke():
    """流式 tool_call 与历史回填格式冒烟：一轮带工具调用的历史 + tool 结果"""
    r = req([{"role": "user", "content": W["ctx"] + "\n[smoke] Find where MTP_DRAFT_VOCAB is used in /work/repo. Use Grep."}], 200)
    print("turn1", r["prompt_tokens"], r["cached_tokens"], r["finish"], r["tool_calls"], repr(r["text"][:100]), f"ttft={r['ttft']:.2f}")
    if not r["tool_calls"]: return
    a = asst_msg(r, "smoke", 1)
    hist = [{"role": "user", "content": W["ctx"] + "\n[smoke] Find where MTP_DRAFT_VOCAB is used in /work/repo. Use Grep."}, a,
            {"role": "tool", "tool_call_id": a["tool_calls"][0]["id"], "content": "/work/repo/x.sh:3:MTP_DRAFT_VOCAB=0"}]
    r2 = req(hist, 100)
    print("turn2", r2["prompt_tokens"], r2["cached_tokens"], r2["finish"], r2["tool_calls"], repr(r2["text"][:150]))


{"record": record, "cold": cold, "pin": pin, "smoke": smoke,
 "c1": lambda: replay(1), "c4": lambda: replay(4), "c8": lambda: replay(8)}[mode]()
