# gate.py <base_url> <model> <key> <tag> <outdir> — 顺序单流、贪心、关思考、max_tokens=900，
# 16 条（abbench 的 EN/ZH，round-1 前缀 "[r1-i]"）+ long4k（512 tok），返回 token ids 与 top-2 logprobs。
import json, os, sys, time, urllib.request
sys.argv, _a = sys.argv[:1] + ["x", "x", "x", "x", "x", "x"], sys.argv
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "abbench.py")).read().split("url, model, key")[0]
exec(src)  # 取 EN / ZH，与测速完全同一份
url, model, key, tag, outdir = _a[1:6]

def chat(content, mt):
    body = {"model": model, "messages": [{"role": "user", "content": content}], "max_tokens": mt,
            "temperature": 0, "logprobs": True, "top_logprobs": 2, "return_token_ids": True,
            "chat_template_kwargs": {"enable_thinking": False}}
    h = {"Content-Type": "application/json", "Authorization": "Bearer " + key}
    r = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(), headers=h)
    t0 = time.time(); d = json.load(urllib.request.urlopen(r, timeout=3600)); dt = time.time() - t0
    c = d["choices"][0]; lp = (c.get("logprobs") or {}).get("content") or []
    return {"prompt_token_ids": d.get("prompt_token_ids") or c.get("prompt_token_ids"), "token_ids": c.get("token_ids"),
            "text": c["message"].get("content"), "finish": c.get("finish_reason"), "sec": dt,
            "top": [[(t["token"], t["logprob"]) for t in x.get("top_logprobs", [])[:2]] for x in lp]}

items = [("en", i, f"[r1-{i}] " + p, 900) for i, p in enumerate(EN)] + [("zh", i, f"[r1-{i}] " + p, 900) for i, p in enumerate(ZH)]
doc = open(os.path.join(HERE, "long4k.txt")).read()
items.append(("long4k", 0, "[r1-0] " + doc + "\n\n请用中文分点总结上面材料的主要内容，约 400 字。", 512))
out = []
for lang, i, content, mt in items:
    r = chat(content, mt); r.update({"lang": lang, "i": i}); out.append(r)
    print(tag, lang, i, "n=", len(r["token_ids"] or []), f"{r['sec']:.1f}s", repr((r["text"] or "")[:40]), flush=True)
os.makedirs(outdir, exist_ok=True)
json.dump(out, open(os.path.join(outdir, f"{tag}-gate.json"), "w"), ensure_ascii=False)
