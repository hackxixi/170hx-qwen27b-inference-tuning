# tf.py <base_url> <model> <key> <ref_gate.json> <tag> <outdir> [langs,逗号分隔] — teacher forcing：参考输出（prompt ids + 输出 ids）
# 喂给 /v1/completions，prompt_logprobs=1，统计本模型 argmax 与参考 token 一致率（逐输出位置）。单条出错跳过、不退出。
import json, os, sys, urllib.request
url, model, key, ref, tag, outdir = sys.argv[1:7]
langs = set(sys.argv[7].split(",")) if len(sys.argv) > 7 else {"en", "zh"}
R = [r for r in json.load(open(ref)) if r["lang"] in langs]; res = []
for r in R:
    p, o = r["prompt_token_ids"], r["token_ids"]
    body = {"model": model, "prompt": p + o, "max_tokens": 1, "temperature": 0, "prompt_logprobs": 1}
    h = {"Content-Type": "application/json", "Authorization": "Bearer " + key}
    try:
        d = json.load(urllib.request.urlopen(urllib.request.Request(url + "/v1/completions", data=json.dumps(body).encode(), headers=h), timeout=3600))
    except Exception as e:
        print("ERR", r["lang"], r["i"], repr(e)[:300], flush=True); continue
    pl = d["choices"][0].get("prompt_logprobs") or d.get("prompt_logprobs")
    agree = 0; n = 0
    for j, tok in enumerate(o):
        ent = pl[len(p) + j]
        top = min(ent.items(), key=lambda kv: kv[1]["rank"])
        n += 1; agree += int(int(top[0]) == tok)
    res.append({"lang": r["lang"], "i": r["i"], "n": n, "agree": agree})
    print(tag, r["lang"], r["i"], agree, "/", n, flush=True)
tot = sum(x["n"] for x in res); ag = sum(x["agree"] for x in res)
print(tag, "TOP1_AGREE", ag, "/", tot, f"{ag/max(tot,1):.5f}")
json.dump({"per": res, "agree": ag, "n": tot}, open(os.path.join(outdir, f"{tag}-tf.json"), "w"))
