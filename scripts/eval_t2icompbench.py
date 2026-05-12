#!/usr/bin/env python3
"""
简化的LoRA评估脚本 (支持多GPU, 包含T2I-CompBench)
用法:
    # 单GPU
    python scripts/eval_lora_simple.py \
        --lora_path /path/to/lora \
        --benchmark pickscore

    # 多GPU (torchrun)
    torchrun --nproc_per_node=8 scripts/eval_lora_simple.py \
        --lora_path /path/to/lora \
        --benchmark ocr \
        --num_gpus 8

    # T2I-CompBench (所有类别, 只生成图片, 后续用 cal_t2i_compbench_value.sh 打分)
    python scripts/eval_lora_simple.py \
        --lora_path /path/to/lora \
        --benchmark t2i_compbench

    # T2I-CompBench (单个类别)
    python scripts/eval_lora_simple.py \
        --lora_path /path/to/lora \
        --benchmark t2i_compbench_3d_spatial

    # 不带LoRA, 纯base model推理
    python scripts/eval_lora_simple.py \
        --lora_path "" \
        --benchmark t2i_compbench
"""

import os
import argparse
import json
import time
from PIL import Image
import numpy as np
import torch
from diffusers import StableDiffusion3Pipeline
from peft import PeftModel
import flow_grpo.rewards
from flow_grpo.diffusers_patch.train_dreambooth_lora_sd3 import encode_prompt

BASE_MODEL_ID = "/your/path/to/stable-diffusion-3.5-medium"

T2I_COMPBENCH_CATEGORIES = [
    ("3d_spatial_val", 0),
    ("color_val", 1),
    ("complex_val", 2),
    ("new_objects", 3),
    ("non_spatial_val", 4),
    ("numeracy_val", 5),
    ("shape_val", 6),
    ("spatial_val", 7),
    ("texture_val", 8),
]
T2I_COMPBENCH_CAT_NAMES = [c[0] for c in T2I_COMPBENCH_CATEGORIES]
T2I_COMPBENCH_CAT2ID = {c[0]: c[1] for c in T2I_COMPBENCH_CATEGORIES}


def parse_args():
    parser = argparse.ArgumentParser(description="评估SD3.5-LoRA在指定benchmark上的表现")
    parser.add_argument("--lora_path", type=str, required=True,
                        help="LoRA权重路径 (为空则使用base model)")
    parser.add_argument("--benchmark", type=str, default="pickscore",
                        help="单个或多个benchmark(逗号分隔) "
                             "t2i_compbench (所有类别), t2i_compbench_<cat> (单个类别)")
    parser.add_argument("--base_model", type=str, default=BASE_MODEL_ID,
                        help="SD3.5 base model路径")
    parser.add_argument("--dataset_root", type=str, default="./dataset",
                        help="数据集根目录")
    parser.add_argument("--output_dir", type=str, default="./eval_results",
                        help="结果输出目录")
    parser.add_argument("--resolution", type=int, default=512,
                        help="图片分辨率")
    parser.add_argument("--num_steps", type=int, default=40,
                        help="扩散步数")
    parser.add_argument("--guidance_scale", type=float, default=4.5,
                        help="CFG scale")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="批大小")
    parser.add_argument("--num_samples", type=int, default=0,
                        help="评估样本数(0=全部)")
    parser.add_argument("--save_images", action="store_true",
                        help="保存生成的图片")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    parser.add_argument("--num_gpus", type=int, default=8,
                        help="GPU数量")
    parser.add_argument("--only_strict", action="store_true",
                        help="geneval only strict模式")
    parser.add_argument("--no_optimize", action="store_true",
                        help="禁用xformers/torch.compile优化")
    parser.add_argument("--merge_lora", action="store_true", default=True,
                        help="是否将LoRA权重merge到base model (False则保持LoRA结构)")
    parser.add_argument("--compbench_root", type=str, default=None,
                        help="T2I-CompBench数据集根目录 (默认为 dataset_root/T2I-CompBench/dataset_val)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

class TextPromptDataset:
    def __init__(self, dataset_path, split="test"):
        self.file_path = os.path.join(dataset_path, f"{split}.txt")
        with open(self.file_path, "r", encoding="utf-8") as f:
            self.prompts = [line.strip() for line in f if line.strip()]
        self.metadatas = [{} for _ in self.prompts]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": self.metadatas[idx]}


class GenEvalDataset:
    def __init__(self, dataset_path, split="test"):
        self.file_path = os.path.join(dataset_path, f"{split}_metadata.jsonl")
        with open(self.file_path, "r", encoding="utf-8") as f:
            self.metadatas = [json.loads(line) for line in f]
            self.prompts = [m["prompt"] for m in self.metadatas]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": self.metadatas[idx]}


class CompBenchDataset:
    """T2I-CompBench dataset loader."""
    def __init__(self, dataset_root, categories=None):
        self.prompts = []
        self.metadatas = []
        self._categories = categories if categories is not None else T2I_COMPBENCH_CAT_NAMES

        for cat in self._categories:
            txt_file = os.path.join(dataset_root, f"{cat}.txt")
            if not os.path.isfile(txt_file):
                print(f"[CompBenchDataset] Warning: {txt_file} not found, skipping")
                continue
            with open(txt_file, "r", encoding="utf-8") as f:
                lines = [l.strip() for l in f if l.strip()]
            cat_id = T2I_COMPBENCH_CAT2ID.get(cat, -1)
            self.prompts.extend(lines)
            self.metadatas.extend([{"category": cat, "category_id": cat_id}] * len(lines))

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": self.metadatas[idx]}


def get_compbench_root(args, dataset_root):
    if args.compbench_root:
        return args.compbench_root
    for candidate in [
        os.path.join(dataset_root, "T2I-CompBench", "dataset_val"),
        os.path.join(dataset_root, "T2I-CompBench"),
    ]:
        if os.path.isdir(candidate):
            return candidate
    return os.path.join(dataset_root, "T2I-CompBench", "dataset_val")


def load_dataset(benchmark: str, dataset_root: str, num_samples: int = 0):
    path = os.path.join(dataset_root, benchmark)
    if not os.path.isdir(path):
        path = dataset_root

    if benchmark == "geneval":
        ds = GenEvalDataset(path, "test")
    elif benchmark.startswith("t2i_compbench"):
        root = get_compbench_root(args_, dataset_root)
        if benchmark == "t2i_compbench":
            ds = CompBenchDataset(root, categories=None)
        else:
            cat = benchmark.replace("t2i_compbench_", "")
            if cat not in T2I_COMPBENCH_CAT2ID:
                raise ValueError(f"Unknown compbench category: {cat}")
            ds = CompBenchDataset(root, categories=[cat])
    else:
        ds = TextPromptDataset(path, "test")

    items = [{"prompt": p, "metadata": m} for p, m in zip(ds.prompts, ds.metadatas)]
    return items[:num_samples] if num_samples > 0 else items


REWARD_MAP = {
    "pickscore": "pickscore_score",
    "pickscore_sfw": "pickscore_score",
    "pickscore_sfw2": "pickscore_score",
    "clipscore": "clip_score",
    "aesthetic": "aesthetic_score",
    "imagereward": "imagereward_score",
    "ocr": "ocr_score",
    "geneval": "geneval_score",
    "deqa": "deqa_score_remote",
    "qwenvl": "qwenvl_score",
    "jpeg_comp": "jpeg_compressibility",
    "unifiedreward": "unifiedreward_score_sglang",
}


# ---------------------------------------------------------------------------
# Pipeline building
# ---------------------------------------------------------------------------

def build_lora_pipeline(base_model, lora_path, device, enable_optimize=True, merge_lora=True):
    """Load base model, optionally merge LoRA, and move to device."""
    pipeline = StableDiffusion3Pipeline.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
    )

    if lora_path and lora_path.strip():
        print(f"[Pipeline] Loading LoRA from: {lora_path}")
        pipeline.transformer = PeftModel.from_pretrained(
            pipeline.transformer, lora_path, local_files_only=True
        )
        if merge_lora:
            pipeline.transformer = pipeline.transformer.merge_and_unload()
            print(f"[Pipeline] LoRA merged and unloaded")
        else:
            print(f"[Pipeline] LoRA loaded (NOT merged, kept as adapters)")
    else:
        print(f"[Pipeline] No LoRA loaded, using base model only")

    pipeline.to(device)

    if enable_optimize:
        try:
            pipeline.transformer.enable_xformers_memory_efficient_attention()
            print(f"[Optimize] xformers enabled")
        except Exception as e:
            print(f"[Optimize] xformers not available: {e}")

        pipeline.vae.enable_slicing()
        pipeline.vae.enable_tiling()

        try:
            pipeline.transformer = torch.compile(
                pipeline.transformer,
                mode="reduce-overhead",
                fullgraph=True,
            )
            print(f"[Optimize] torch.compile enabled")
        except Exception as e:
            print(f"[Optimize] torch.compile not available: {e}")

    return pipeline


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------

def encode_prompts_batch(pipeline, prompts, device):
    text_encoders = [
        pipeline.text_encoder,
        pipeline.text_encoder_2,
        pipeline.text_encoder_3,
    ]
    tokenizers = [
        pipeline.tokenizer,
        pipeline.tokenizer_2,
        pipeline.tokenizer_3,
    ]
    pos_embeds, pos_pooled = encode_prompt(
        text_encoders, tokenizers, prompts,
        max_sequence_length=128, device=device,
    )
    neg_embeds, neg_pooled = encode_prompt(
        text_encoders, tokenizers, [""],
        max_sequence_length=128, device=device,
    )
    bsz = len(prompts)
    neg_embeds = neg_embeds.repeat(bsz, 1, 1).to(device)
    neg_pooled = neg_pooled.repeat(bsz, 1).to(device)
    return pos_embeds, pos_pooled, neg_embeds, neg_pooled


def generate_images(pipeline, pos_embeds, pos_pooled, neg_embeds, neg_pooled,
                    num_steps, guidance_scale, resolution, device, seed=42):
    generator = torch.Generator(device=device).manual_seed(seed)
    with torch.no_grad():
        output = pipeline(
            prompt_embeds=pos_embeds,
            pooled_prompt_embeds=pos_pooled,
            negative_prompt_embeds=neg_embeds,
            negative_pooled_prompt_embeds=neg_pooled,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            output_type="pt",
            height=resolution,
            width=resolution,
            generator=generator,
        )
        images = output.images
    return images.cpu().numpy()


def chw_to_hwc_uint8(chw_arr):
    arr = np.clip(chw_arr.transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)
    return arr


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

def setup_distributed():
    if "LOCAL_RANK" in os.environ and "RANK" in os.environ:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        torch.distributed.init_process_group(backend="nccl")
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        torch.cuda.set_device(0)
        rank = 0
        world_size = 1
        local_rank = 0
    return rank, world_size, local_rank


def cleanup_distributed():
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def maybe_barrier():
    if torch.distributed.is_initialized():
        torch.distributed.barrier()


# ---------------------------------------------------------------------------
# T2I-CompBench inference (reference-style: save images per category)
# ---------------------------------------------------------------------------

def inference_compbench(
    pipeline,
    dataset,
    args,
    device,
    rank=0,
    world_size=1,
):
    """
    T2I-CompBench 推理: 完全对齐 reason_inference_t2i_compbench.py 的保存逻辑.
    - 目录结构: {output_dir}/{category}/samples/
    - 每条 prompt 生成 repeat_size=10 张图 (各用不同 seed)
    - 文件名: {原始prompt}_{idx:05d}.png
    - 只要第一张图存在就跳过整条 prompt
    """
    is_main = (rank == 0)
    repeat_size = 10

    n_prompts = len(dataset)
    samples_per_rank = (n_prompts + world_size - 1) // world_size
    start_idx = rank * samples_per_rank
    end_idx = min(start_idx + samples_per_rank, n_prompts)
    my_dataset = dataset[start_idx:end_idx]

    if not my_dataset:
        return 0

    saved_count = 0

    for global_item_idx in range(start_idx, end_idx):
        item = dataset[global_item_idx]
        prompt = item["prompt"]
        meta = item["metadata"]
        category = meta.get("category", "unknown")

        output_path = os.path.join(args.output_dir, category, "samples")
        os.makedirs(output_path, exist_ok=True)

        first_img_file = os.path.join(output_path, f"{prompt}_00000.png")
        if os.path.exists(first_img_file):
            if is_main:
                print(f"[SKIP] {first_img_file} already exists")
            continue

        pos_embeds, pos_pooled, neg_embeds, neg_pooled = encode_prompts_batch(
            pipeline, [prompt], device
        )

        for idx in range(repeat_size):
            images_np = generate_images(
                pipeline, pos_embeds, pos_pooled, neg_embeds, neg_pooled,
                num_steps=args.num_steps,
                guidance_scale=args.guidance_scale,
                resolution=args.resolution,
                device=device,
                seed=args.seed + global_item_idx * repeat_size + idx,
            )
            img_hwc = chw_to_hwc_uint8(images_np[0])
            pil_img = Image.fromarray(img_hwc)
            img_file = os.path.join(output_path, f"{prompt}_{idx:05d}.png")
            pil_img.save(img_file)

            if is_main:
                print(f"[{category}] Saved: {img_file}")

        saved_count += repeat_size

    return saved_count


# ---------------------------------------------------------------------------
# Generic evaluation (non-compbench benchmarks)
# ---------------------------------------------------------------------------

def evaluate_benchmark(
    pipeline,
    benchmark,
    dataset,
    reward_fn,
    reward_kwargs,
    accepted_kwargs,
    args,
    device,
    rank=0,
    world_size=1,
):
    is_main = (rank == 0)
    filtered_kwargs = {k: v for k, v in reward_kwargs.items() if k in accepted_kwargs}

    n_prompts = len(dataset)
    prompts_per_rank = (n_prompts + world_size - 1) // world_size
    start_idx = rank * prompts_per_rank
    end_idx = min(start_idx + prompts_per_rank, n_prompts)
    my_dataset = dataset[start_idx:end_idx]

    if not my_dataset:
        return []

    results = []

    for batch_start in range(0, len(my_dataset), args.batch_size):
        batch_end = min(batch_start + args.batch_size, len(my_dataset))
        batch = my_dataset[batch_start:batch_end]
        prompts_batch = [item["prompt"] for item in batch]
        metadatas_batch = [item["metadata"] for item in batch]

        pos_embeds, pos_pooled, neg_embeds, neg_pooled = encode_prompts_batch(
            pipeline, prompts_batch, device
        )

        bsz = len(prompts_batch)
        images_np = generate_images(
            pipeline, pos_embeds, pos_pooled, neg_embeds, neg_pooled,
            num_steps=args.num_steps,
            guidance_scale=args.guidance_scale,
            resolution=args.resolution,
            device=device,
            seed=args.seed + batch_start,
        )
        images_hwc = [chw_to_hwc_uint8(images_np[i]) for i in range(bsz)]

        raw_result = reward_fn(images_hwc, prompts_batch, metadatas_batch, **filtered_kwargs)

        def to_numpy(x):
            if hasattr(x, 'device') and x.device.type == 'cuda':
                return x.cpu().numpy()
            return np.asarray(x)

        if isinstance(raw_result, dict):
            rewards = {k: to_numpy(v) for k, v in raw_result.items()}
        elif isinstance(raw_result, tuple) and len(raw_result) == 5:
            scores, rewards_list, strict_rewards, group_rewards, group_strict_rewards = raw_result
            rewards = {
                "score": to_numpy(scores),
                "reward": to_numpy(rewards_list),
                "strict_reward": to_numpy(strict_rewards),
            }
        elif isinstance(raw_result, tuple) and len(raw_result) == 2:
            scores, _ = raw_result
            rewards = {"score": to_numpy(scores)}
        else:
            rewards = {"score": to_numpy(raw_result) if np.isscalar(raw_result) else raw_result}

        for i in range(bsz):
            record = {
                "prompt": prompts_batch[i],
                "reward": {k: float(v[i]) for k, v in rewards.items()},
                "benchmark": benchmark,
            }
            if "metadata" in batch[i]:
                record["metadata"] = batch[i]["metadata"]

            if args.save_images and is_main:
                img_dir = os.path.join(args.output_dir, benchmark, "images")
                os.makedirs(img_dir, exist_ok=True)
                pil_img = Image.fromarray(images_hwc[i])
                img_path = os.path.join(img_dir, f"idx{batch_start + i:06d}.jpg")
                pil_img.save(img_path, quality=95)
                record["image_path"] = img_path

                prompt_path = os.path.join(img_dir, f"idx{batch_start + i:06d}.txt")
                with open(prompt_path, "w", encoding="utf-8") as f:
                    f.write(prompts_batch[i])

            results.append(record)

        if is_main:
            print(f"  [{benchmark}] {batch_end}/{len(my_dataset)} done", end="\r")

    if is_main:
        print()

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global args_
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")
    args_ = parse_args()
    args = args_

    is_main = (rank == 0)

    if is_main:
        print(f"[Config] LoRA: {args.lora_path if args.lora_path else '(base model only)'}")
        print(f"[Config] Merge LoRA: {args.merge_lora}")
        print(f"[Config] Benchmark: {args.benchmark}")
        print(f"[Config] Device: {device}")
        print(f"[Config] Multi-GPU: {world_size > 1} (world_size={world_size})")

    if is_main:
        print(f"\n[1/2] Building pipeline...")
    pipeline = build_lora_pipeline(
        args.base_model, args.lora_path, device,
        enable_optimize=not args.no_optimize,
        merge_lora=args.merge_lora,
    )

    if is_main:
        print(f"[2/2] Starting evaluation...")

    os.makedirs(args.output_dir, exist_ok=True)

    benchmarks = [b.strip() for b in args.benchmark.split(",")]

    for benchmark in benchmarks:
        if is_main:
            print(f"\n{'='*50}")
            print(f"Benchmark: {benchmark}")

        if benchmark == "geneval":
            reward_kwargs = {"only_strict": True} if args.only_strict else {}
            accepted_kwargs = ["only_strict"]
        else:
            reward_kwargs = {}
            accepted_kwargs = []

        is_compbench = benchmark.startswith("t2i_compbench")

        try:
            dataset = load_dataset(benchmark, args.dataset_root, args.num_samples)
        except FileNotFoundError as e:
            if is_main:
                print(f"[WARN] Dataset not found: {e}")
            continue
        except ValueError as e:
            if is_main:
                print(f"[WARN] {e}")
            continue

        if not dataset:
            if is_main:
                print(f"[WARN] Empty dataset")
            continue

        if is_main:
            print(f"  Prompts: {len(dataset)}")

        # ---------- T2I-CompBench: 只生成图片 ----------
        if is_compbench:
            if is_main:
                print(f"\n  [T2I-CompBench] Generating images (output_dir={args.output_dir})...")

            t0 = time.time()
            saved_count = inference_compbench(
                pipeline=pipeline,
                dataset=dataset,
                args=args,
                device=device,
                rank=rank,
                world_size=world_size,
            )
            elapsed = time.time() - t0

            if world_size > 1:
                all_counts = [0] * world_size
                torch.distributed.all_gather_object(all_counts, saved_count)
                total_saved = sum(all_counts)
            else:
                total_saved = saved_count

            if world_size > 1:
                maybe_barrier()

            if is_main:
                print(f"\n  [T2I-CompBench] Done. {elapsed:.1f}s total, {total_saved} saved.")
                print(f"  Images saved under: {args.output_dir}")
                print(f"  Run cal_t2i_compbench_value.sh for scoring.")
            continue

        # ---------- Non-compbench: reward evaluation ----------
        reward_fn_name = REWARD_MAP.get(benchmark)
        if reward_fn_name is None:
            if is_main:
                print(f"[WARN] Unknown benchmark: {benchmark}")
            continue

        if is_main:
            print(f"  Reward: {reward_fn_name}")

        reward_fn = getattr(flow_grpo.rewards, reward_fn_name)(device)

        if world_size > 1:
            maybe_barrier()

        t0 = time.time()
        results = evaluate_benchmark(
            pipeline=pipeline,
            benchmark=benchmark,
            dataset=dataset,
            reward_fn=reward_fn,
            reward_kwargs=reward_kwargs,
            accepted_kwargs=accepted_kwargs,
            args=args,
            device=device,
            rank=rank,
            world_size=world_size,
        )

        if world_size > 1:
            maybe_barrier()

        elapsed = time.time() - t0

        if world_size > 1:
            all_flat = [None] * world_size
            torch.distributed.all_gather_object(all_flat, results)
            gathered = []
            for shard in all_flat:
                gathered.extend(shard)
            results = gathered

        if is_main and results:
            reward_keys = list(results[0]["reward"].keys())
            print(f"\n  [{benchmark}] {len(results)} samples | {elapsed:.1f}s total")

            for rk in reward_keys:
                vals = [r["reward"].get(rk, float("nan")) for r in results]
                mean_val = float(np.mean(vals))
                std_val = float(np.std(vals))
                min_val = float(np.min(vals))
                max_val = float(np.max(vals))
                print(f"    {rk:30s}  mean={mean_val:.4f}  "
                      f"std={std_val:.4f}  "
                      f"min={min_val:.4f}  max={max_val:.4f}")

            out_path = os.path.join(args.output_dir, f"{benchmark}_results.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"    Saved: {out_path}")

            summary_path = os.path.join(args.output_dir, f"{benchmark}_summary.json")
            summary = {
                "benchmark": benchmark,
                "num_samples": len(results),
                "total_time_seconds": round(elapsed, 1),
                "averages": {
                    rk: {
                        "mean": float(np.mean([r["reward"].get(rk, float("nan")) for r in results])),
                        "std": float(np.std([r["reward"].get(rk, float("nan")) for r in results])),
                    }
                    for rk in reward_keys
                },
            }
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            print(f"    Saved: {summary_path}")

    if is_main:
        print(f"\nDone! Results saved to: {os.path.abspath(args.output_dir)}")

    if world_size > 1:
        maybe_barrier()
    cleanup_distributed()


if __name__ == "__main__":
    main()
