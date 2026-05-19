python scripts/merge.py \
    --lora_paths \
        /path/to/SD3.5M-FlowGRPO-GenEval \
        /path/to/SD3.5M-FlowGRPO-Text\
        /path/to/SD3.5M-FlowGRPO-pick \
    --save_path /your/path/to/new/lora \
    --merge_mode average
    # --merge_mode weighted \
    # --weights 0.5 0.3 0.2
