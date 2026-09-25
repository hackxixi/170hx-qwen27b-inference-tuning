# probe.py <base_url> <model> <key> <refA_gate.json> <B_gate.json> <tag> <outdir>
# 对 A/B 每个首分歧位 j：把 prompt_ids + 公共前缀[:j] 以贪心 max_tokens=1 发给本服务端（prefill 路径），看选 A 的还是 B 的 token，并记 top-2 差
import json, os, sys, urllib.request
url, model, key, fa, fb, tag, outdir = sys.argv[1:8]
A, B = json.load(open(fa)), json.load(open(fb)); out = []
h = {"Content-Type": "application/json", "Authorization": "Bearer " + key}
for a, b in zip(A, B):
    ta, tb = a["token_ids"], b["token_ids"]
    if ta == tb: continue
    j = next((k for k in range(min(len(ta), len(tb))) if ta[k] != tb[k]), None)
    if j is None: continue
    body = {"model": model, "prompt": a["prompt_token_ids"] + ta[:j], "max_tokens": 1, "temperature": 0, "logprobs": 5, "return_token_ids": True}
    try:
        d = json.load(urllib.request.urlopen(urllib.request.Request(url + "/v1/completions", data=json.dumps(body).encode(), headers=h), timeout=600))
        c = d["choices"][0]; tid = (c.get("token_ids") or [None])[0]
        tl = (c.get("logprobs") or {}).get("top_logprobs") or [{}]
        v = sorted(tl[0].values(), reverse=True)
        mg = v[0] - v[1] if len(v) > 1 else None
    except Exception as e:
        tid, mg = "ERR " + repr(e)[:100], None
    r = {"lang": a["lang"], "i": a["i"], "j": j, "A": ta[j], "B": tb[j], "probe": tid, "margin": mg,
         "pick": "A" if tid == ta[j] else "B" if tid == tb[j] else "other"}
    out.append(r); print(tag, r, flush=True)
json.dump(out, open(os.path.join(outdir, f"{tag}-probe.json"), "w"))
