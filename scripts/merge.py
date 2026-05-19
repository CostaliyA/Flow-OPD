#!/usr/bin/env python3
"""
Merge multiple LoRA adapters by averaging or weighting their weights.
Saves the merged result to the first LoRA's directory.

Usage:
    # Average mode (default):
    python scripts/merge.py \
        --lora_paths /path/to/lora1 /path/to/lora2 /path/to/lora3 \
        --save_path /path/to/output \
        --merge_mode average

    # Weighted mode:
    python scripts/merge.py \
        --lora_paths /path/to/lora1 /path/to/lora2 /path/to/lora3 \
        --save_path /path/to/output \
        --merge_mode weighted \
        --weights 0.5 0.3 0.2
"""

import argparse
import os
import shutil
import torch
from safetensors.torch import load_file, save_file
from collections import OrderedDict


def load_lora_weights(lora_path):
    """Load LoRA weights from a safetensors file."""
    safetensors_file = os.path.join(lora_path, "adapter_model.safetensors")
    if not os.path.exists(safetensors_file):
        raise FileNotFoundError(f"adapter_model.safetensors not found in {lora_path}")

    weights = load_file(safetensors_file)
    return weights


def merge_lora_weights(lora_paths, merge_mode="average", weights=None):
    """
    Merge multiple LoRA weights.

    Args:
        lora_paths: List of paths to LoRA checkpoints.
        merge_mode: "average" (mean), "sum" (add up), or "weighted" (custom weights).
        weights: List of floats for weighted merge (must match len(lora_paths)).
    """
    if len(lora_paths) < 2:
        raise ValueError("Need at least 2 LoRA paths to merge")

    # Load all weights
    all_weights = [load_lora_weights(path) for path in lora_paths]

    # Verify that all checkpoints have the same keys
    reference_keys = set(all_weights[0].keys())
    for i, lora_w in enumerate(all_weights[1:], start=2):
        keys = set(lora_w.keys())
        if keys != reference_keys:
            missing = reference_keys - keys
            extra = keys - reference_keys
            raise ValueError(
                f"LoRA {i} has mismatched keys. "
                f"Missing: {missing}, Extra: {extra}"
            )

    # Merge by averaging (or summing or weighted)
    merged = OrderedDict()
    for key in all_weights[0].keys():
        tensors = [w[key] for w in all_weights]
        if merge_mode == "average":
            merged[key] = torch.stack(tensors).mean(dim=0)
        elif merge_mode == "sum":
            merged[key] = torch.stack(tensors).sum(dim=0)
        elif merge_mode == "weighted":
            if weights is None:
                raise ValueError("weights must be provided for weighted merge mode")
            weights_tensor = torch.tensor(weights, dtype=torch.float32)
            weights_tensor = weights_tensor / weights_tensor.sum()
            merged[key] = sum(t * w for t, w in zip(tensors, weights_tensor))
        else:
            raise ValueError(f"Unknown merge_mode: {merge_mode}")

    return merged


def save_merged_lora(merged_weights, save_path, copy_from_path):
    """
    Save merged LoRA weights to save_path.
    Copies adapter_config.json from copy_from_path.
    """
    os.makedirs(save_path, exist_ok=True)

    # Save merged safetensors
    safetensors_path = os.path.join(save_path, "adapter_model.safetensors")
    save_file(merged_weights, safetensors_path)
    print(f"Saved merged weights to {safetensors_path}")

    # Copy adapter_config.json from the first LoRA
    config_src = os.path.join(copy_from_path, "adapter_config.json")
    config_dst = os.path.join(save_path, "adapter_config.json")
    if os.path.exists(config_src):
        shutil.copy2(config_src, config_dst)
        print(f"Copied adapter_config.json from {config_src} to {config_dst}")
    else:
        print(f"Warning: adapter_config.json not found in {copy_from_path}, skipping.")


def parse_args():
    parser = argparse.ArgumentParser(description="Merge multiple LoRA adapters")
    parser.add_argument(
        "--lora_paths",
        type=str,
        nargs="+",
        required=True,
        help="Paths to LoRA checkpoints (at least 2)"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="Directory to save merged LoRA (defaults to first lora_path)"
    )
    parser.add_argument(
        "--merge_mode",
        type=str,
        default="average",
        choices=["average", "sum", "weighted"],
        help="How to merge weights: 'average' (mean), 'sum', or 'weighted' (custom weights)"
    )
    parser.add_argument(
        "--weights",
        type=float,
        nargs="+",
        default=None,
        help="Weights for weighted merge mode (must match number of LoRA paths). "
             "Example: --weights 0.5 0.3 0.2"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    lora_paths = args.lora_paths
    if len(lora_paths) < 2:
        raise ValueError("Need at least 2 LoRA paths to merge")

    # Default save_path to the first LoRA path
    save_path = args.save_path
    if not save_path:
        raise ValueError("--save_path must be specified")
    print(f"=== LoRA Merge ===")
    print(f"Input LoRAs ({len(lora_paths)}):")
    for p in lora_paths:
        print(f"  - {p}")
    print(f"Merge mode: {args.merge_mode}")
    if args.merge_mode == "weighted":
        if args.weights is None:
            raise ValueError("--weights must be specified when using weighted mode")
        if len(args.weights) != len(lora_paths):
            raise ValueError(f"Number of weights ({len(args.weights)}) must match number of LoRAs ({len(lora_paths)})")
        print(f"Weights: {args.weights}")
    print(f"Save to: {save_path}")
    print()

    # Verify all paths exist
    for p in lora_paths:
        if not os.path.isdir(p):
            raise NotADirectoryError(f"Not a directory: {p}")
        safetensors = os.path.join(p, "adapter_model.safetensors")
        if not os.path.exists(safetensors):
            raise FileNotFoundError(f"adapter_model.safetensors not found in {p}")

    # Merge
    merged_weights = merge_lora_weights(lora_paths, merge_mode=args.merge_mode, weights=args.weights)

    # Save
    save_merged_lora(merged_weights, save_path, copy_from_path=lora_paths[0])

    print(f"\n=== Done! ===")


if __name__ == "__main__":
    main()
