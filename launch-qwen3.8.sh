conda activate mlx
#cd ~/mlx-dist
Mname=Qwen3.8-27B
M=mlx-community/${Mname}-8bit

mlx.launch --verbose --backend jaccl --hostfile ~/mlx-dist/hosts.json --env MLX_METAL_FAST_SYNCH=1 -- \
  $HOME/miniforge3/envs/mlx/bin/python -m mlx_lm.server \
  --model $M \
  --host 0.0.0.0 --port 8080 \
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 \
  --max-tokens 16384 \
  --chat-template ${Mname}.json
  
