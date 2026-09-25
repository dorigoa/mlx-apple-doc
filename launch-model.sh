# #!/bin/bash

# usage() {
#     echo "Usage: $0 <Qwen3.6-35B-A3B-UD-MLX|Qwen3.8-27B> -quant <4bit|8bit> [-think <0|1>] [-port <number>]"
#     echo "  <Qwen3.6-35B-A3B-UD-MLX|Qwen3.8-27B>    Obbligatorio: precisione del modello"
#     echo "  -think <0|1>    Opzionale: abilita il modo 'thinking' (default: 0)"
#     echo "  -port <number>  Opzionale: porta server (default: 8080)"
#     exit 1
# }

# if [ $# -lt 1 ]; then
#     usage
# fi

# BIT_DEPTH=$1
# shift

# THINK_MODE=0
# PORT=8080

# while [[ $# -gt 0 ]]; do
#   case $1 in
#     Qwen3.6-35B-A3B-UD-MLX|Qwen3.8-27B)
#       Mname="$1"
#       shift
#       ;;
#     -quant)
#       if [[ -n "$2" && "$2" =~ ^[4-8]bit$ ]]; then
#         BIT_DEPTH="$2"
#         shift 2
#       else
#         echo "Errore: -think richiede 0 o 1"
#         exit 1
#       fi
#       ;;
#     -think)
#       if [[ -n "$2" && "$2" =~ ^[0-1]$ ]]; then
#         THINK_MODE="$2"
#         shift 2
#       else
#         echo "Errore: -think richiede 0 o 1"
#         exit 1
#       fi
#       ;;
#     -port)
#       if [[ -n "$2" && "$2" =~ ^[0-9]+$ ]]; then
#         PORT="$2"
#         shift 2
#       else
#         echo "Errore: -port richiede un numero"
#         exit 1
#       fi
#       ;;
#     *)
#       echo "Errore: Argomento sconosciuto: $1"
#       echo "Uso: $0 [4bit|8bit] -think [0|1] -port [numero]"
#       exit 1
#       ;;
#   esac
# done


# if [[ "$BIT_DEPTH" != "4bit" && "$BIT_DEPTH" != "8bit" ]]; then
#     echo "Error: First argument must be '4bit' or '8bit'"
#     usage
# fi

# M="mlx-community/${Mname}-${BIT_DEPTH}"

# eval "$(conda shell.bash hook)"
# conda activate mlx

# export HF_HUB_OFFLINE=1

# [[ "$THINK_MODE" -eq 1 ]] && THINK_BOOL="true" || THINK_BOOL="false"

# echo "Server launch: Model=$M, Port=$PORT, Thinking=$THINK_BOOL"

# conda activate mlx
# mlx.launch --verbose --backend jaccl --hostfile ./hosts.json --env MLX_METAL_FAST_SYNCH=1 -- \
#   $HOME/miniforge3/envs/mlx/bin/python -m mlx_lm.server \
#   --model $M \
#   --host 0.0.0.0 --port $PORT \
#   --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 \
#   --max-tokens 16384 \
#   #--chat-template ${Mname}.json \
#   --chat-template-args "{\"enable_thinking\": $THINK_BOOL}"














#!/usr/bin/env bash

readonly MODELS=("Qwen3.8-27B" "GLM-4.7-Flash" "Qwen3.6-35B-A3B", "Llama-3.3-70B-Instruct")
readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly HOSTFILE="${SCRIPT_DIR}/hosts.json"
readonly PYTHON_BIN="${HOME}/miniforge3/envs/mlx/bin/python"

usage() {
    cat >&2 <<EOF
Uso: $(basename "$0") <modello> -quant <4bit|8bit> [-think <0|1>] [-port <1-65535>]
  <modello>            Obbligatorio: $(IFS='|'; echo "${MODELS[*]}")
  -quant <4bit|8bit>   Obbligatorio: quantizzazione
  -think <0|1>         Opzionale: modalità thinking (default: 0)
  -port <n>            Opzionale: porta del server (default: 8080)
  -h                   Mostra questo aiuto
EOF
    exit "${1:-1}"
}

die() { echo "Errore: $*" >&2; exit 1; }

is_model() {
    local m
    for m in "${MODELS[@]}"; do [[ "$1" == "$m" ]] && return 0; done
    return 1
}

MNAME=""
BIT_DEPTH=""
THINK_MODE=0
PORT=8080

while [[ $# -gt 0 ]]; do
    case "$1" in
        -quant)
            [[ "${2:-}" =~ ^(4|8)bit$ ]] || die "-quant requires '4bit' or '8bit'"
            BIT_DEPTH="$2"; shift 2 ;;
        -think)
            [[ "${2:-}" =~ ^[01]$ ]] || die "-think requires 0 or 1"
            THINK_MODE="$2"; shift 2 ;;
        -port)
            [[ "${2:-}" =~ ^[0-9]{1,5}$ ]] && (( 10#$2 >= 1 && 10#$2 <= 65535 )) \
                || die "-port requires an integer in the range 1024-65535"
            PORT="$((10#$2))"; shift 2 ;;
        -h|--help)
            usage 0 ;;
        -*)
            echo "Error: unknown option: $1" >&2; usage ;;
        *)
            is_model "$1" || { echo "Error: unknown model: $1" >&2; usage; }
            [[ -z "$MNAME" ]] || die "Model specified more than once"
            MNAME="$1"; shift ;;
    esac
done

[[ -n "$MNAME" ]]      || { echo "Error: missing model" >&2; usage; }
[[ -n "$BIT_DEPTH" ]]  || { echo "Error: missing -quant " >&2; usage; }
[[ -f "$HOSTFILE" ]]   || die "hostfile not found: $HOSTFILE"
[[ -x "$PYTHON_BIN" ]] || die "Python interpreter not found: $PYTHON_BIN"

MODEL="mlx-community/${MNAME}-${BIT_DEPTH}"
(( THINK_MODE )) && THINK_BOOL=true || THINK_BOOL=false

command -v conda >/dev/null 2>&1 || die "conda not found in the PATH"
eval "$(conda shell.bash hook)"  || die "conda init failed"
conda activate mlx               || die "could not activated conda env 'mlx'"
command -v mlx.launch >/dev/null 2>&1 || die "mlx.launch not found in 'mlx' env"

export HF_HUB_OFFLINE=1

echo "Server start: model=$MODEL, port=$PORT, thinking=$THINK_BOOL" >&2

server_args=(
    -m mlx_lm.server
    --model "$MODEL"
    --host 0.0.0.0 --port "$PORT"
    --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0
    --max-tokens 16384
    # --chat-template "${SCRIPT_DIR}/${MNAME}.json"
    --chat-template-args "{\"enable_thinking\": ${THINK_BOOL}}"
)

exec mlx.launch --verbose --backend jaccl --hostfile "$HOSTFILE" \
    --env MLX_METAL_FAST_SYNCH=1 --env HF_HUB_OFFLINE=1 \
    -- "$PYTHON_BIN" "${server_args[@]}"