#!/bin/bash
# 在生产模型目录旁建 -fast-head8：全部相对 symlink 到 -fast，仅 config.json 与 shard7 为 int8 版
# 用法：bash -s <base|shard7路径>
set -eu
M=/opt/hq/hyperqwen/repo/models; F=Qwen3.8-27B-W4A16-AutoRound-fast; N=$M/$F-head8; S7=$1
mkdir -p $N; cd $N
for f in $(ls $M/$F); do
  case $f in config.json|model-00007-of-00007.safetensors) ;; *) ln -sfn ../$F/$f $f ;; esac
done
cp /opt/hq/ab-models/head8/D2/config.json config.json
if [ "$S7" = base ]; then ln -sfn ../Qwen3.8-27B-W4A16-AutoRound/model-00007-of-00007.safetensors model-00007-of-00007.safetensors
else cp --remove-destination "$S7" model-00007-of-00007.safetensors; fi
echo "shard7 md5: $(md5sum -L model-00007-of-00007.safetensors 2>/dev/null | cut -c1-12 || md5sum $(readlink -f model-00007-of-00007.safetensors) | cut -c1-12)"
python3 -c "import json;c=json.load(open('config.json'));q=c.get('quantization_config') or c['text_config']['quantization_config'];print({k:g['weights']['num_bits'] for k,g in q['config_groups'].items()})"
cd /opt/hq/hyperqwen; cp -n hq.env hq.env.bak-20260925-pre-head8
grep -q '^MODEL=' hq.env || printf 'MODEL=/app/models/%s-head8\nVERIFY=0\n' $F >> hq.env
grep -E '^(MODEL|VERIFY)=' hq.env; ls $N | wc -l; ls -la hq.env*
