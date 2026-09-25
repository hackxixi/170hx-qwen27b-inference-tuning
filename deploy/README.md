# deploy

基于 [HyperQwen](https://github.com/syv-ai/HyperQwen) 的单卡单副本部署：每张卡一个容器，参数统一放在 `hq.env`，用 `run-replica.sh <名> <GPU-UUID> <端口>` 重建。本目录只放在它之上新增的部分，路径统一写成 `/opt/hq/...` 占位，用之前按自己的环境改。

| 文件 | 用途 |
|---|---|
| `run-replica-c1.sh <name> <gpu-uuid> <port>` | 启动组合版副本：`hyperqwen:v030` 镜像（vLLM 0.30 + PR#189，需自建）+ D3 模型目录 + int4 起草头补丁 |
| `roll-c1.sh <端口:容器>...` | 逐个把副本切换为组合版，每个都检查 /health、试出字、核对日志标记，任一失败即停 |
| `roll-head8.sh <端口:容器>...` | 同上，切换到 int8 输出头 |
| `litellm/session-affinity.snippet.yaml` | 网关会话粘滞配置片段（生产用） |
| `litellm/*-2replica-test.yaml` | 两副本对照测试用的完整 litellm 配置（shuffle / sticky） |

注意事项：
- 加卡以后设备序号会错位，所以一律按 GPU UUID 绑卡。
- sm80 上 `SPEC=dflash2` 会稳定触发 Xid 31，这是 HyperQwen 的默认配置，要改成 `SPEC=mtp`。
- 170HX 上 27B 不要跨卡做 TP：PCIe Gen2、无 P2P，比单卡慢 6–9 倍。
- 多个副本串行启动。热缓存下每个副本约 2.5–7 分钟可以通过 /health。
- 64GB 显存依赖 [cmpunlocker](https://github.com/buliaoyin/cmpunlocker) 解锁驱动。NVIDIA 驱动升级以后必须重新打补丁，否则重启后显存会回到 8GB，PCIe 也会降到 Gen1。
