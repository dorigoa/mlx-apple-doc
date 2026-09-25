#!/usr/bin/env bash

readonly MODELS=("Qwen3.8-27B" "GLM-4.7-Flash" "Qwen3.6-35B-A3B", "Llama-3.3-70B-Instruct")
readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly HOSTFILE="${SCRIPT_DIR}/hosts.json"
readonly PYTHON_BIN="${HOME}/miniforge3/envs/mlx/bin/python"

usage() {
    cat >&2 <<EOF
Uso: $(basename "$0") <model> [-8bit] [-nothink] [-port <1-65535>]
  <modello>  Mandatory: $(IFS='|'; echo "${MODELS[*]}")
  -8bit      Optional: use 8bit quantization (default: 4bit)
  -nothink   Optional: disable thinking (default enable)
  -port <n>  Optional: listening port for API server(default: 8080)
  -h         Show this help
EOF
    exit "${1:-1}"
}

die() { echo "Error: $*" >&2; exit 1; }

is_model() {
    local m
    for m in "${MODELS[@]}"; do [[ "$1" == "$m" ]] && return 0; done
    return 1
}

MNAME=""
BIT_DEPTH="4bit"
THINK_MODE=1
PORT=8080

while [[ $# -gt 0 ]]; do
    case "$1" in
        -8bit)
            BIT_DEPTH="8bit"; shift ;;
        -nothink)
            #[[ "${2:-}" =~ ^[01]$ ]] || die "-think requires 0 or 1"
            THINK_MODE=0; shift ;;
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