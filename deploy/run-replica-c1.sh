#!/bin/bash
# run-replica-c1.sh <name> <gpu-uuid> <host-port> — 组合版 C1 副本：vLLM 0.30（hyperqwen:v030，PR#189 自建）
# + D3（MTP 起草用 int4 输出头、验证用 int8；补丁 patches/qwen3_5_mtp.v030.head8.py，MTP_DRAFT_HEAD4=1）
# 其余参数与 run-replica.sh / hq.env 相同。回滚：bash run-replica.sh <name> <uuid> <port>
set -eu
name=$1 uuid=$2 port=$3
H=/opt/hq/hyperqwen
VP=/app/venv/lib/python3.12/site-packages/vllm/model_executor/models/qwen3_5_mtp.py
docker rm -f "$name" 2>/dev/null || true
docker run -d --name "$name" --gpus "\"device=$uuid\"" --shm-size=16g \
  -p "$port:18020" --env-file $H/hq.env \
  -e MODEL=/app/models/Qwen3.8-27B-W4A16-AutoRound-fast-head8-d3 -e MTP_DRAFT_HEAD4=1 \
  -v $H/patches/qwen3_5_mtp.v030.head8.py:$VP:ro \
  -v $H/repo/models:/app/models:ro -v hq-cache-v030:/cache \
  --restart unless-stopped hyperqwen:v030 single
