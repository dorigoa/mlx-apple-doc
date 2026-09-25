# conda activate mlx
# #cd ~/mlx-dist
# Mname=Qwen3.8-27B
# M=mlx-community/${Mname}-8bit
# export HF_HUB_OFFLINE=1
# mlx.launch --verbose --backend jaccl --hostfile ./hosts.json --env MLX_METAL_FAST_SYNCH=1 -- \
#   $HOME/miniforge3/envs/mlx/bin/python -m mlx_lm.server \
#   --model $M \
#   --host 0.0.0.0 --port 8080 \
#   --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 \
#   --max-tokens 16384 \
#   --chat-template ${Mname}.json \
#   --chat-template-args '{"enable_thinking": false}'
  


#!/bin/bash

usage() {
    echo "Usage: $0 <4bit|8bit> [-think <0|1>] [-port <number>]"
    echo "  <4bit|8bit>    Obbligatorio: precisione del modello"
    echo "  -think <0|1>    Opzionale: abilita il modo 'thinking' (default: 0)"
    echo "  -port <number>  Opzionale: porta server (default: 8080)"
    exit 1
}

if [ $# -lt 1 ]; then
    usage
fi

BIT_DEPTH=$1
shift

THINK_MODE=0
PORT=8080

while getopts "t:p:" opt; do
  case $opt in
    t) THINK_MODE=$OPTARG ;;
    p) PORT=$OPTARG ;;
    *) usage ;;
  esac
done

if [[ "$BIT_DEPTH" != "4bit" && "$BIT_DEPTH" != "8bit" ]]; then
    echo "Error: First argument must be '4bit' or '8bit'"
    usage
fi

Mname="Qwen3.8-27B"
M="mlx-community/${Mname}-${BIT_DEPTH}"

eval "$(conda shell.bash hook)"
conda activate mlx

export HF_HUB_OFFLINE=1

echo "Server launch: Model=$M, Port=$PORT, Thinking=$THINK_MODE"
[[ $THINK_MODE -eq 1 ]] && THINK_BOOL="true" || THINK_BOOL="false"

echo "thinkbool=$THINK_BOOL"

mlx.launch --verbose --backend jaccl --hostfile ./hosts.json --env MLX_METAL_FAST_SYNCH=1 -- \
  $HOME/miniforge3/envs/mlx/bin/python -m mlx_lm.server \
  --model $M \
  --host 0.0.0.0 --port $PORT \
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 \
  --max-tokens 16384 \
  --chat-template ${Mname}.json \
  --chat-template-args "{\"enable_thinking\": $THINK_BOOL}"
