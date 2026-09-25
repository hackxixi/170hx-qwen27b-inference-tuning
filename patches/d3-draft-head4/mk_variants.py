# 生成 D2 / D3 目录（只写有变化的文件，其余 symlink 到 -fast）
# D2：-fast 全部 symlink；model-00007（lm_head）改指基座的 int8 版；config group_1 num_bits 4→8
# D3：D2 同款 + model_draft_head4.safetensors（-fast 的 int4 lm_head 改名 mtp.draft_head4.*）+ index 加 3 个键
import json, os
from safetensors import safe_open
from safetensors.torch import save_file
import sys
M = sys.argv[1] if len(sys.argv) > 1 else "/opt/hq/hyperqwen/repo/models"; FAST = M + "/Qwen3.8-27B-W4A16-AutoRound-fast"; BASE = M + "/Qwen3.8-27B-W4A16-AutoRound"
H8 = sys.argv[2] if len(sys.argv) > 2 else "/opt/hq/ab-models/head8"; S7 = "model-00007-of-00007.safetensors"; DH = "model_draft_head4.safetensors"
cfg = json.load(open(FAST + "/config.json")); g = cfg["quantization_config"]["config_groups"]
lm = [k for k, v in g.items() if v["targets"] == ["re:.*lm_head$"]]; assert len(lm) == 1 and g[lm[0]]["weights"]["num_bits"] == 4
g[lm[0]]["weights"]["num_bits"] = 8
print("groups", {k: (v["targets"], v["weights"]["num_bits"]) for k, v in g.items()})
idx = json.load(open(FAST + "/model.safetensors.index.json"))
for V in ("D2", "D3"):
    D = f"{H8}/{V}"; os.makedirs(D, exist_ok=True)
    for f in os.listdir(FAST):
        if f in ("config.json", "model.safetensors.index.json"): continue
        p = os.path.join(D, f)
        if not os.path.lexists(p): os.symlink(os.path.join(BASE if f == S7 else FAST, f), p)
    json.dump(cfg, open(D + "/config.json", "w"), indent=2)
    ix = json.loads(json.dumps(idx))
    if V == "D3":
        with safe_open(FAST + "/" + S7, "pt") as f:
            t = {k.replace("lm_head.", "mtp.draft_head4."): f.get_tensor(k) for k in f.keys() if k.startswith("lm_head.")}
        print("draft_head4", {k: (tuple(v.shape), str(v.dtype)) for k, v in t.items()})
        if not os.path.exists(D + "/" + DH): save_file(t, D + "/" + DH, metadata={"format": "pt"})
        for k in t: ix["weight_map"][k] = DH
    json.dump(ix, open(D + "/model.safetensors.index.json", "w"), indent=2)
    print(V, "ok", len(ix["weight_map"]))
