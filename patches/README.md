# patches

所有 `.diff` 都是 unified diff。补丁的基底**不是**上游原版 vLLM 文件，而是打过 [HyperQwen](https://github.com/syv-ai/HyperQwen) 补丁系列之后、镜像里实际安装的那份文件（文件里带有 `syv patch` 标记）。许可证见仓库根目录的 `NOTICE`。

| 目录 | 内容 | 状态 |
|---|---|---|
| `d3-draft-head4/` | int4 起草头：MTP 起草读单独存放的 int4 头，目标模型验证仍用 int8 头 | **已上线**（组合版） |
| `head8-prod/` | 生产用 `-fast-head8` 模型目录的构建脚本 | **已上线** |
| `suffix-hybrid-NEGATIVE/` | MTP + suffix decoding 混合起草 | **实测负收益，仅供参考** |

## d3-draft-head4

| 文件 | 基底 |
|---|---|
| `qwen3_5_mtp.v028.diff` | HyperQwen 684e927（vLLM 0.28）的 `vllm/model_executor/models/qwen3_5_mtp.py` |
| `qwen3_5_mtp.v030.diff` | vLLM 0.30.0 + [HyperQwen PR#189](https://github.com/syv-ai/HyperQwen/pull/189) 补丁系列之后的同一文件 |
| `mk_variants.py` | 生成模型目录。D2 = `-fast` + 基座的 int8 lm_head shard；D3 = D2 + `model_draft_head4.safetensors`（把 `-fast` 的 int4 lm_head 改名为 `mtp.draft_head4.*`）+ index 里补 3 个键 |

用法：
1. 在 HyperQwen 的模型目录上运行 `python3 mk_variants.py <models_dir> <out_dir>`。输入需要 `Qwen3.8-27B-W4A16-AutoRound`（基座）和 `-fast` 两个目录，输出的 D3 目录约多出 626 MB。
   - 注意：`mk_variants.py` 是测试用的，建的是**绝对路径** symlink，产物在 `<out_dir>/D2`、`<out_dir>/D3`。整个 models 目录挂进容器以后，这些链接会断。
   - 生产目录 `Qwen3.8-27B-W4A16-AutoRound-fast-head8-d3` 要放在 models 目录下，结构如下：
     - 除 `model.safetensors.index.json` 外，其余文件全部用**相对** symlink 指向同级的 `-fast-head8` 目录（见 `head8-prod/`）；
     - 放入实体文件 `model_draft_head4.safetensors`（可以直接取 `mk_variants.py` 生成的那份）；
     - `model.safetensors.index.json` 用 D3 里的版本，比 `-fast-head8` 多 3 个 `mtp.draft_head4.*` 键。
   - 这个生产目录是手工组装的，本仓库没有提供对应脚本。
2. 对容器内的 `qwen3_5_mtp.py` 打补丁，可以挂载覆盖，也可以 `patch -p1`。
3. 启动时设 `MODEL=<D3 目录>` 和 `MTP_DRAFT_HEAD4=1`。不设这个变量时，补丁不改变任何行为。
4. 确认生效：启动日志里应出现 `head8: MTP drafter uses separate draft_head4`，并且能看到 draft_head4 与 lm_head 的形状。

命名上的坑：新参数名不能以 `lm_head` 结尾，否则 compressed-tensors 会按首个匹配把它归进 lm_head 量化组；也不能包含 `draft_lm_head`，否则加载时会被 HyperQwen 的截断词表逻辑跳过。

## head8-prod
`mk-head8-prod.sh <base|shard 路径>` 在生产模型目录旁边建 `-fast-head8` 目录：其余文件全部用相对 symlink 指回 `-fast`，只有 `config.json` 和 lm_head 所在的 shard 用 int8 版。脚本同时会备份 `hq.env`，并往里追加 `MODEL=` 和 `VERIFY=0`。里面的路径是占位符（`/opt/hq/...`），用之前按自己的环境改。

## suffix-hybrid-NEGATIVE
- `hybrid_suffix.py.diff`：新文件 `vllm/v1/spec_decode/hybrid_suffix.py`。
- `gpu_model_runner.v028.diff`：vLLM 0.28 的 V1 runner 加 3 处钩子，共 19 行，`HYBRID_SUFFIX=1` 时才启用。
- 运行时依赖 [ArcticInference](https://github.com/snowflakedb/ArcticInference) 的 `arctic_inference.suffix_decoding`（C++/nanobind，只需编这个子包）。
- 实测结论见 [reports/05](../reports/05-mtp-suffix-hybrid.md)：开发场景单会话慢 12%。
