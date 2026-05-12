CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 vllm serve \
 /path/to/Qwen3-VL-30B-A3B-Instruct \
  --host 0.0.0.0 \
  --port 8001 \
  --tensor-parallel-size 8 \
  --gpu-memory-utilization 0.95 \
  --served-model-name "Qwen3-VL-30B-A3B-Instruct" \
  --max_model_len 160000 \
  --mm-processor-cache-gb 0 \
  --no-enable-prefix-caching