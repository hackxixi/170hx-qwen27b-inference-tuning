#!/bin/bash
# roll-head8.sh <端口:容器>... — 逐个用 run-replica.sh 重建，等 /health 200 且试出字成功再动下一个；任何一个失败即停止
set -u
K=$(grep VLLM_API_KEY /opt/hq/hyperqwen/hq.env | cut -d= -f2)
for item in "$@"; do
  port=${item%%:*}; name=${item#*:}
  uuid=$(docker inspect $name --format '{{index (index .HostConfig.DeviceRequests 0).DeviceIDs 0}}')
  echo "$(date +%T) rebuild $name $uuid $port"
  bash /opt/hq/hyperqwen/run-replica.sh $name $uuid $port >/dev/null
  ok=0
  for i in $(seq 1 60); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' localhost:$port/health)" = 200 ] && { ok=1; break; }; sleep 15
  done
  [ $ok = 1 ] || { echo "$(date +%T) FAIL health $name"; exit 1; }
  out=$(curl -s localhost:$port/v1/chat/completions -H "Authorization: Bearer $K" -H 'Content-Type: application/json' \
    -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"1+1=?"}],"max_tokens":20,"chat_template_kwargs":{"enable_thinking":false}}')
  echo "$out" | grep -q '"content"' || { echo "$(date +%T) FAIL chat $name: ${out:0:200}"; exit 1; }
  docker logs $name 2>&1 | grep -q 'fast-head8' && echo "$(date +%T) OK $name (head8)" || { echo "$(date +%T) FAIL model $name"; exit 1; }
done
echo "$(date +%T) ROLL_DONE"
