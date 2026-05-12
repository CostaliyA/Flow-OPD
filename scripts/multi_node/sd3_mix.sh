#!/bin/bash
# Common part for all nodes
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5
export NCCL_DEBUG=WARN
export NCCL_IB_GID_INDEX=3


export WANDB_API_KEY=""
pip install paddlepaddle-gpu==2.6.2
pip install paddleocr==2.9.1
pip install python-Levenshtein
python scripts/prepare.py

MASTER_PORT=19001
RANK=$1
MASTER_ADDR=TBD
# Launch command (parameters automatically read from accelerate_multi_node.yaml)
accelerate launch --config_file scripts/accelerate_configs/multi_node.yaml \
    --num_machines 4 --num_processes 32 \
    --machine_rank ${RANK} --main_process_ip ${MASTER_ADDR} --main_process_port ${MASTER_PORT} \
    scripts/train_sd3_mixed.py \
    --config config/grpo.py:geneval_ocr_deqa_pickscore_32gpu
