# export HF_ENDPOINT=https://hf-mirror.com 

torchrun --nproc_per_node=8 scripts/eval_t2icompbench.py \
    --lora "path/to/your/lora" \
    --benchmark t2i_compbench \
    --output_dir ./eval_results/compbench_images
