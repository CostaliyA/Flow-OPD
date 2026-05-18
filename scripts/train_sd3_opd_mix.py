#!/usr/bin/env python3


from collections import defaultdict
import contextlib
import os
import datetime
from concurrent import futures
import time
import json
import hashlib
from absl import app, flags
from accelerate import Accelerator
from ml_collections import config_flags
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger
from diffusers import StableDiffusion3Pipeline
from diffusers.utils.torch_utils import is_compiled_module
import numpy as np
import flow_grpo.prompts
import flow_grpo.rewards
from flow_grpo.stat_tracking import PerPromptStatTracker
from flow_grpo.diffusers_patch.sd3_pipeline_with_logprob import pipeline_with_logprob
from flow_grpo.diffusers_patch.sd3_sde_with_logprob import sde_step_with_logprob
from flow_grpo.diffusers_patch.train_dreambooth_lora_sd3 import encode_prompt
import torch
import wandb
from functools import partial
import tqdm
import tempfile
from PIL import Image
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict, PeftModel
import random
from torch.utils.data import Dataset, DataLoader, Sampler
from flow_grpo.ema import EMAModuleWrapper

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/base.py", "Training configuration.")

logger = get_logger(__name__)


class TextPromptDataset(Dataset):
    """OCR 等文本提示数据集"""
    def __init__(self, dataset, split='train'):
        self.file_path = os.path.join(dataset, f'{split}.txt')
        with open(self.file_path, 'r') as f:
            self.prompts = [line.strip() for line in f.readlines()]
        
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": {}}

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas


class GenevalPromptDataset(Dataset):
    """GenEval 对象检测数据集"""
    def __init__(self, dataset, split='train'):
        self.file_path = os.path.join(dataset, f'{split}_metadata.jsonl')
        with open(self.file_path, 'r', encoding='utf-8') as f:
            self.metadatas = [json.loads(line) for line in f]
            self.prompts = [item['prompt'] for item in self.metadatas]
        
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": self.metadatas[idx]}

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas


class DistributedKRepeatSampler(Sampler):
    """分布式 K 重复采样器（与原始实现相同）"""
    def __init__(self, dataset, batch_size, k, num_replicas, rank, seed=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.k = k
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        
        self.total_samples = self.num_replicas * self.batch_size
        assert self.total_samples % self.k == 0, f"k can not divide n*b, k{k}-num_replicas{num_replicas}-batch_size{batch_size}"
        self.m = self.total_samples // self.k
        self.epoch = 0

    def __iter__(self):
        while True:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            
            indices = torch.randperm(len(self.dataset), generator=g)[:self.m].tolist()
            repeated_indices = [idx for idx in indices for _ in range(self.k)]
            
            shuffled_indices = torch.randperm(len(repeated_indices), generator=g).tolist()
            shuffled_samples = [repeated_indices[i] for i in shuffled_indices]
            
            per_card_samples = []
            for i in range(self.num_replicas):
                start = i * self.batch_size
                end = start + self.batch_size
                per_card_samples.append(shuffled_samples[start:end])
            
            yield per_card_samples[self.rank]
    
    def set_epoch(self, epoch):
        self.epoch = epoch


# ============================================================
# 修复后的混合数据集实现
# ============================================================

class MixedPromptDatasetV2:
    """
    混合数据集 V2：修复了与 DataLoader 的兼容性问题
    
    核心改动：
    - 不再使用 batch_sampler 模式
    - 使用自定义迭代器直接返回 (prompt, metadata, dataset_id) 元组
    - 与 DistributedKRepeatSampler 保持相同的数据分发逻辑
    """
    def __init__(self, mixed_datasets_config):
        self.datasets = []
        self.dataset_weights = []
        self.dataset_info = []
        
        # 构建 idx -> (dataset_idx, sample_idx) 的全局映射
        self.global_idx_to_local = []  # [(ds_idx, local_idx), ...]
        self.global_idx_to_ds_name = []  # [ds_name, ...]
        
        for ds_config in mixed_datasets_config:
            ds_type = ds_config.get("prompt_fn", ds_config.get("name", "unknown"))
            if ds_type == "geneval":
                dataset = GenevalPromptDataset(ds_config["path"], 'train')
            else:  # general_ocr 等
                dataset = TextPromptDataset(ds_config["path"], 'train')

            self.datasets.append(dataset)
            self.dataset_weights.append(ds_config.get("weight", 1.0))
            self.dataset_info.append({
                "name": ds_config["name"],
                "reward_fn": ds_config["reward_fn"],
                "test_batch_size": ds_config.get("test_batch_size", 16),
                "path": ds_config["path"],
            })
            
            # 建立全局索引映射
            for local_idx in range(len(dataset)):
                self.global_idx_to_local.append((len(self.datasets) - 1, local_idx))
                self.global_idx_to_ds_name.append(ds_config["name"])

        total_weight = sum(self.dataset_weights)
        self.dataset_weights = [w / total_weight for w in self.dataset_weights]
        self.total_size = len(self.global_idx_to_local)
        
        # 预先计算每个数据集的样本数
        self.ds_sizes = [len(ds) for ds in self.datasets]
        self.ds_cumulative = [0] + list(np.cumsum(self.ds_sizes))

    def __len__(self):
        return self.total_size

    def get_dataset_info(self):
        return self.dataset_info

    def __getitem__(self, global_idx):
        ds_idx, local_idx = self.global_idx_to_local[global_idx]
        item = self.datasets[ds_idx][local_idx]
        return {
            "prompt": item["prompt"],
            "metadata": item["metadata"],
            "dataset_id": ds_idx,
            "dataset_name": self.dataset_info[ds_idx]["name"],
        }


class MixedDataIterator:
    """
    混合数据集迭代器：直接使用 DistributedKRepeatSampler 的逻辑
    返回 (prompts, metadatas, dataset_ids) 元组
    """
    def __init__(self, mixed_dataset, batch_size, k, num_replicas, rank, seed=0):
        self.mixed_dataset = mixed_dataset
        self.batch_size = batch_size
        self.k = k
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.epoch = 0
        
        self.total_batch_size = self.num_replicas * self.batch_size
        assert self.total_batch_size % self.k == 0, \
            f"k cannot divide n*b, k={self.k}, num_replicas={num_replicas}, batch_size={batch_size}"
        self.m = self.total_batch_size // self.k

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        while True:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)

            # 按权重计算每个数据集应采样的数量
            samples_per_dataset = []
            remaining = self.m
            for i, (size, weight) in enumerate(zip(self.mixed_dataset.ds_sizes, self.mixed_dataset.dataset_weights)):
                if i == len(self.mixed_dataset.ds_sizes) - 1:
                    n = remaining
                else:
                    n = int(self.m * weight)
                    n = min(n, size)
                remaining -= n
                samples_per_dataset.append(n)

            # 从每个数据集采样
            all_global_indices = []
            dataset_ids = []
            for ds_idx, n in enumerate(samples_per_dataset):
                if n > 0:
                    # 在该数据集的范围内随机采样
                    start_idx = self.mixed_dataset.ds_cumulative[ds_idx]
                    end_idx = self.mixed_dataset.ds_cumulative[ds_idx + 1]
                    local_indices = torch.randperm(self.mixed_dataset.ds_sizes[ds_idx], generator=g)[:n]
                    global_indices = (start_idx + local_indices).tolist()
                    all_global_indices.extend(global_indices)
                    dataset_ids.extend([ds_idx] * n)

            # 重复 k 次
            repeated_indices = []
            repeated_dataset_ids = []
            for idx, ds_id in zip(all_global_indices, dataset_ids):
                for _ in range(self.k):
                    repeated_indices.append(idx)
                    repeated_dataset_ids.append(ds_id)

            # 打乱
            perm = torch.randperm(len(repeated_indices), generator=g).tolist()
            shuffled_indices = [repeated_indices[i] for i in perm]
            shuffled_dataset_ids = [repeated_dataset_ids[i] for i in perm]

            # 分配给各个进程
            per_card_samples = []
            per_card_dataset_ids = []
            for i in range(self.num_replicas):
                start = i * self.batch_size
                end = start + self.batch_size
                per_card_samples.append(shuffled_indices[start:end])
                per_card_dataset_ids.append(shuffled_dataset_ids[start:end])

            # 返回当前进程的样本
            my_indices = per_card_samples[self.rank]
            my_dataset_ids = per_card_dataset_ids[self.rank]
            
            # 获取实际数据
            prompts = []
            metadatas = []
            for global_idx in my_indices:
                item = self.mixed_dataset[global_idx]
                prompts.append(item["prompt"])
                metadatas.append(item["metadata"])
            
            yield prompts, metadatas, my_dataset_ids

    def __len__(self):
        return self.m


def compute_text_embeddings(prompt, text_encoders, tokenizers, max_sequence_length, device):
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds = encode_prompt(
            text_encoders, tokenizers, prompt, max_sequence_length
        )
        prompt_embeds = prompt_embeds.to(device)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device)
    return prompt_embeds, pooled_prompt_embeds


def calculate_zero_std_ratio(prompts, gathered_rewards):
    prompt_array = np.array(prompts)
    
    unique_prompts, inverse_indices, counts = np.unique(
        prompt_array, 
        return_inverse=True,
        return_counts=True
    )
    
    grouped_rewards = gathered_rewards['ori_avg'][np.argsort(inverse_indices)]
    split_indices = np.cumsum(counts)[:-1]
    reward_groups = np.split(grouped_rewards, split_indices)
    
    prompt_std_devs = np.array([np.std(group) for group in reward_groups])
    zero_std_count = np.count_nonzero(prompt_std_devs == 0)
    zero_std_ratio = zero_std_count / len(prompt_std_devs)
    
    return zero_std_ratio, prompt_std_devs.mean()


def create_generator(prompts, base_seed):
    generators = []
    for prompt in prompts:
        hash_digest = hashlib.sha256(prompt.encode()).digest()
        prompt_hash_int = int.from_bytes(hash_digest[:4], 'big')
        seed = (base_seed + prompt_hash_int) % (2**31)
        gen = torch.Generator().manual_seed(seed)
        generators.append(gen)
    return generators

        
def compute_log_prob(transformer, pipeline, sample, j, embeds, pooled_embeds, config):
    if config.train.cfg:
        noise_pred = transformer(
            hidden_states=torch.cat([sample["latents"][:, j]] * 2),
            timestep=torch.cat([sample["timesteps"][:, j]] * 2),
            encoder_hidden_states=embeds,
            pooled_projections=pooled_embeds,
            return_dict=False,
        )[0]
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = (
            noise_pred_uncond
            + config.sample.guidance_scale
            * (noise_pred_text - noise_pred_uncond)
        )
    else:
        noise_pred = transformer(
            hidden_states=sample["latents"][:, j],
            timestep=sample["timesteps"][:, j],
            encoder_hidden_states=embeds,
            pooled_projections=pooled_embeds,
            return_dict=False,
        )[0]

    prev_sample, log_prob, prev_sample_mean, std_dev_t = sde_step_with_logprob(
        pipeline.scheduler,
        noise_pred.float(),
        sample["timesteps"][:, j],
        sample["latents"][:, j].float(),
        prev_sample=sample["next_latents"][:, j].float(),
        noise_level=config.sample.noise_level,
    )

    return prev_sample, log_prob, prev_sample_mean, std_dev_t


def compute_log_prob_with_ref(
    transformer, ref_transformer, pipeline, sample, j, embeds, pooled_embeds, config
):
    """
    Forward pass through both policy and reference transformer, returning
    both policy and reference mean trajectories for KL computation.

    Used in multi-dataset OPD training where each dataset has its own
    kl_ref_lora_path reference.
    """
    if config.train.cfg:
        noise_pred = transformer(
            hidden_states=torch.cat([sample["latents"][:, j]] * 2),
            timestep=torch.cat([sample["timesteps"][:, j]] * 2),
            encoder_hidden_states=embeds,
            pooled_projections=pooled_embeds,
            return_dict=False,
        )[0]
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = (
            noise_pred_uncond
            + config.sample.guidance_scale
            * (noise_pred_text - noise_pred_uncond)
        )
        # Reference forward
        ref_noise_pred = ref_transformer(
            hidden_states=torch.cat([sample["latents"][:, j]] * 2),
            timestep=torch.cat([sample["timesteps"][:, j]] * 2),
            # embeds are fp16 inside autocast; cast to float32 to match ref_transformer (float32)
            encoder_hidden_states=embeds.to(torch.float32),
            pooled_projections=pooled_embeds.to(torch.float32),
            return_dict=False,
        )[0]
        ref_noise_pred_uncond, ref_noise_pred_text = ref_noise_pred.chunk(2)
        ref_noise_pred = (
            ref_noise_pred_uncond
            + config.sample.guidance_scale
            * (ref_noise_pred_text - ref_noise_pred_uncond)
        )
    else:
        noise_pred = transformer(
            hidden_states=sample["latents"][:, j],
            timestep=sample["timesteps"][:, j],
            encoder_hidden_states=embeds,
            pooled_projections=pooled_embeds,
            return_dict=False,
        )[0]
        ref_noise_pred = ref_transformer(
            hidden_states=sample["latents"][:, j],
            timestep=sample["timesteps"][:, j],
            # embeds are fp16 inside autocast; cast to float32 to match ref_transformer (float32)
            encoder_hidden_states=embeds.to(torch.float32),
            pooled_projections=pooled_embeds.to(torch.float32),
            return_dict=False,
        )[0]

    prev_sample, log_prob, prev_sample_mean, std_dev_t = sde_step_with_logprob(
        pipeline.scheduler,
        noise_pred.float(),
        sample["timesteps"][:, j],
        sample["latents"][:, j].float(),
        prev_sample=sample["next_latents"][:, j].float(),
        noise_level=config.sample.noise_level,
    )
    _, _, prev_sample_mean_ref, _ = sde_step_with_logprob(
        pipeline.scheduler,
        ref_noise_pred.float(),
        sample["timesteps"][:, j],
        sample["latents"][:, j].float(),
        prev_sample=sample["next_latents"][:, j].float(),
        noise_level=config.sample.noise_level,
    )

    return prev_sample, log_prob, prev_sample_mean, std_dev_t, prev_sample_mean_ref


def eval(pipeline, test_dataloader, text_encoders, tokenizers, config, accelerator, global_step, reward_fn, executor, autocast, num_train_timesteps, ema, transformer_trainable_parameters):
    """单一数据集评估函数"""
    if config.train.ema:
        ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)
    neg_prompt_embed, neg_pooled_prompt_embed = compute_text_embeddings([""], text_encoders, tokenizers, max_sequence_length=128, device=accelerator.device)

    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.test_batch_size, 1, 1)
    sample_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.sample.test_batch_size, 1)

    all_rewards = defaultdict(list)
    for test_batch in tqdm(
            test_dataloader,
            desc="Eval: ",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
        prompts, prompt_metadata = test_batch
        prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
            prompts,
            text_encoders,
            tokenizers,
            max_sequence_length=128,
            device=accelerator.device
        )

        # 调整 negative embeddings 以匹配当前 batch size
        current_batch_size = len(prompt_embeds)
        if current_batch_size < config.sample.test_batch_size:
            sample_neg_prompt_embeds = sample_neg_prompt_embeds[:current_batch_size]
            sample_neg_pooled_prompt_embeds = sample_neg_pooled_prompt_embeds[:current_batch_size]
        elif current_batch_size > config.sample.test_batch_size:
            sample_neg_prompt_embeds = neg_prompt_embed.repeat(current_batch_size, 1, 1)
            sample_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(current_batch_size, 1)

        with autocast():
            with torch.no_grad():
                images, _, _ = pipeline_with_logprob(
                    pipeline,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    negative_prompt_embeds=sample_neg_prompt_embeds,
                    negative_pooled_prompt_embeds=sample_neg_pooled_prompt_embeds,
                    num_inference_steps=config.sample.eval_num_steps,
                    guidance_scale=config.sample.eval_guidance_scale if hasattr(config.sample, 'eval_guidance_scale') else config.sample.guidance_scale,
                    output_type="pt",
                    height=config.resolution,
                    width=config.resolution,
                    noise_level=0,
                )
        rewards = executor.submit(reward_fn, images, prompts, prompt_metadata, only_strict=False)
        time.sleep(0)
        rewards, reward_metadata = rewards.result()

        for key, value in rewards.items():
            rewards_gather = accelerator.gather(torch.as_tensor(value, device=accelerator.device)).cpu().numpy()
            all_rewards[key].append(rewards_gather)
    
    last_batch_images_gather = accelerator.gather(torch.as_tensor(images, device=accelerator.device)).cpu().numpy()
    last_batch_prompt_ids = tokenizers[0](
        prompts,
        padding="max_length",
        max_length=256,
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(accelerator.device)
    last_batch_prompt_ids_gather = accelerator.gather(last_batch_prompt_ids).cpu().numpy()
    last_batch_prompts_gather = pipeline.tokenizer.batch_decode(
        last_batch_prompt_ids_gather, skip_special_tokens=True
    )
    last_batch_rewards_gather = {}
    for key, value in rewards.items():
        last_batch_rewards_gather[key] = accelerator.gather(torch.as_tensor(value, device=accelerator.device)).cpu().numpy()

    all_rewards = {key: np.concatenate(value) for key, value in all_rewards.items()}
    if accelerator.is_main_process:
        with tempfile.TemporaryDirectory() as tmpdir:
            num_samples = min(15, len(last_batch_images_gather))
            sample_indices = range(num_samples)
            for idx, index in enumerate(sample_indices):
                image = last_batch_images_gather[index]
                pil = Image.fromarray(
                    (image.transpose(1, 2, 0) * 255).astype(np.uint8)
                )
                pil = pil.resize((config.resolution, config.resolution))
                pil.save(os.path.join(tmpdir, f"{idx}.jpg"))
            sampled_prompts = [last_batch_prompts_gather[index] for index in sample_indices]
            sampled_rewards = [{k: last_batch_rewards_gather[k][index] for k in last_batch_rewards_gather} for index in sample_indices]
            for key, value in all_rewards.items():
                print(key, value.shape)
            wandb.log(
                {
                    "eval_images": [
                        wandb.Image(
                            os.path.join(tmpdir, f"{idx}.jpg"),
                            caption=f"{prompt:.1000} | " + " | ".join(f"{k}: {v:.2f}" for k, v in reward.items() if v != -10),
                        )
                        for idx, (prompt, reward) in enumerate(zip(sampled_prompts, sampled_rewards))
                    ],
                    **{f"eval_reward_{key}": np.mean(value[value != -10]) for key, value in all_rewards.items()},
                },
                step=global_step,
            )
    if config.train.ema:
        ema.copy_temp_to(transformer_trainable_parameters)


def unwrap_model(model, accelerator):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model


def save_ckpt(save_dir, transformer, global_step, accelerator, ema, transformer_trainable_parameters, config):
    save_root = os.path.join(save_dir, "checkpoints", f"checkpoint-{global_step}")
    save_root_lora = os.path.join(save_root, "lora")
    os.makedirs(save_root_lora, exist_ok=True)
    if accelerator.is_main_process:
        if config.train.ema:
            ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)
        unwrap_model(transformer, accelerator).save_pretrained(save_root_lora)
        if config.train.ema:
            ema.copy_temp_to(transformer_trainable_parameters)


def eval_mixed(pipeline, test_dataloaders, test_reward_fns, dataset_info, text_encoders, tokenizers, config, accelerator, global_step, executor, autocast, num_train_timesteps, ema, transformer_trainable_parameters):
    """
    混合数据集评估函数 - 修复版
    分别对每个数据集进行评估，并正确处理多卡情况
    """
    if config.train.ema:
        ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)

    all_rewards = defaultdict(list)
    all_dataset_rewards = {ds_info["name"]: defaultdict(list) for ds_info in dataset_info}
    
    eval_guidance_scale = config.sample.eval_guidance_scale if hasattr(config.sample, 'eval_guidance_scale') else config.sample.guidance_scale

    for ds_name, test_dataloader in test_dataloaders.items():
        test_reward_fn = test_reward_fns[ds_name]
        
        neg_prompt_embed, neg_pooled_prompt_embed = compute_text_embeddings(
            [""], text_encoders, tokenizers, max_sequence_length=128, device=accelerator.device
        )

        for test_batch in tqdm(
                test_dataloader,
                desc=f"Eval [{ds_name}]: ",
                disable=not accelerator.is_local_main_process,
                position=0,
            ):
            prompts, prompt_metadata = test_batch
            
            # 处理最后一个 batch 可能小于 batch_size 的情况
            current_batch_size = len(prompts)
            sample_neg_prompt_embeds = neg_prompt_embed.repeat(current_batch_size, 1, 1)
            sample_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(current_batch_size, 1)
            
            prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
                prompts,
                text_encoders,
                tokenizers,
                max_sequence_length=128,
                device=accelerator.device
            )

            with autocast():
                with torch.no_grad():
                    images, _, _ = pipeline_with_logprob(
                        pipeline,
                        prompt_embeds=prompt_embeds,
                        pooled_prompt_embeds=pooled_prompt_embeds,
                        negative_prompt_embeds=sample_neg_prompt_embeds,
                        negative_pooled_prompt_embeds=sample_neg_pooled_prompt_embeds,  # 修复
                        num_inference_steps=config.sample.eval_num_steps,
                        guidance_scale=eval_guidance_scale,
                        output_type="pt",
                        height=config.resolution,
                        width=config.resolution,
                        noise_level=0,
                    )

            rewards = executor.submit(test_reward_fn, images, prompts, prompt_metadata, only_strict=False)
            time.sleep(0)
            rewards, reward_metadata = rewards.result()

            for key, value in rewards.items():
                rewards_gather = accelerator.gather(torch.as_tensor(value, device=accelerator.device)).cpu().numpy()
                all_rewards[key].append(rewards_gather)
                all_dataset_rewards[ds_name][key].append(rewards_gather)

    # 合并所有 reward
    all_rewards = {key: np.concatenate(value) for key, value in all_rewards.items()}

    # 保存最后一个 batch 的图像用于 wandb
    last_batch_images_gather = accelerator.gather(torch.as_tensor(images, device=accelerator.device)).cpu().numpy()
    last_batch_prompt_ids = tokenizers[0](
        prompts,
        padding="max_length",
        max_length=256,
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(accelerator.device)
    last_batch_prompt_ids_gather = accelerator.gather(last_batch_prompt_ids).cpu().numpy()
    last_batch_prompts_gather = pipeline.tokenizer.batch_decode(
        last_batch_prompt_ids_gather, skip_special_tokens=True
    )
    last_batch_rewards_gather = {}
    for key, value in rewards.items():
        last_batch_rewards_gather[key] = accelerator.gather(torch.as_tensor(value, device=accelerator.device)).cpu().numpy()

    if accelerator.is_main_process:
        with tempfile.TemporaryDirectory() as tmpdir:
            num_samples = min(15, len(last_batch_images_gather))
            sample_indices = range(num_samples)
            for idx, index in enumerate(sample_indices):
                image = last_batch_images_gather[index]
                pil = Image.fromarray(
                    (image.transpose(1, 2, 0) * 255).astype(np.uint8)
                )
                pil = pil.resize((config.resolution, config.resolution))
                pil.save(os.path.join(tmpdir, f"{idx}.jpg"))

            sampled_prompts = [last_batch_prompts_gather[index] for index in sample_indices]
            sampled_rewards = [{k: last_batch_rewards_gather[k][index] for k in last_batch_rewards_gather} for index in sample_indices]

            wandb_log_dict = {
                "eval_images": [
                    wandb.Image(
                        os.path.join(tmpdir, f"{idx}.jpg"),
                        caption=f"{prompt:.1000} | " + " | ".join(f"{k}: {v:.2f}" for k, v in reward.items() if v != -10),
                    )
                    for idx, (prompt, reward) in enumerate(zip(sampled_prompts, sampled_rewards))
                ],
            }

            # 记录每个数据集的评估 reward
            for ds_name, ds_rewards in all_dataset_rewards.items():
                for key, value in ds_rewards.items():
                    valid_mask = value != -10
                    if valid_mask.any():
                        wandb_log_dict[f"eval_reward_{ds_name}_{key}"] = np.mean(value[valid_mask])

            # 记录总体评估 reward
            for key, value in all_rewards.items():
                valid_mask = value != -10
                if valid_mask.any():
                    wandb_log_dict[f"eval_reward_{key}"] = np.mean(value[valid_mask])

            wandb.log(wandb_log_dict, step=global_step)

    if config.train.ema:
        ema.copy_temp_to(transformer_trainable_parameters)


def main(_):
    config = FLAGS.config

    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    if not config.run_name:
        config.run_name = unique_id
    else:
        config.run_name += "_" + unique_id

    num_train_timesteps = int(config.sample.num_steps * config.train.timestep_fraction)

    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.logdir, config.run_name),
        automatic_checkpoint_naming=True,
        total_limit=config.num_checkpoint_limit,
    )

    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        gradient_accumulation_steps=config.train.gradient_accumulation_steps * num_train_timesteps,
    )
    if accelerator.is_main_process:
        wandb.init(project="flow_grpo",name=config.run_name)
    logger.info(f"\n{config}")

    set_seed(config.seed, device_specific=True)

    # 加载模型
    pipeline = StableDiffusion3Pipeline.from_pretrained(config.pretrained.model)
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.text_encoder_2.requires_grad_(False)
    pipeline.text_encoder_3.requires_grad_(False)
    pipeline.transformer.requires_grad_(not config.use_lora)

    text_encoders = [pipeline.text_encoder, pipeline.text_encoder_2, pipeline.text_encoder_3]
    tokenizers = [pipeline.tokenizer, pipeline.tokenizer_2, pipeline.tokenizer_3]

    pipeline.safety_checker = None
    pipeline.set_progress_bar_config(
        position=1,
        disable=not accelerator.is_local_main_process,
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )

    inference_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16

    pipeline.vae.to(accelerator.device, dtype=torch.float32)
    pipeline.text_encoder.to(accelerator.device, dtype=inference_dtype)
    pipeline.text_encoder_2.to(accelerator.device, dtype=inference_dtype)
    pipeline.text_encoder_3.to(accelerator.device, dtype=inference_dtype)
    pipeline.transformer.to(accelerator.device)

    # LoRA 配置
    if config.use_lora:
        target_modules = [
            "attn.add_k_proj", "attn.add_q_proj", "attn.add_v_proj",
            "attn.to_add_out", "attn.to_k", "attn.to_out.0", "attn.to_q", "attn.to_v",
        ]
        transformer_lora_config = LoraConfig(
            r=32, lora_alpha=64, init_lora_weights="gaussian", target_modules=target_modules,
        )
        if config.train.lora_path:
            pipeline.transformer = PeftModel.from_pretrained(pipeline.transformer, config.train.lora_path)
            pipeline.transformer.set_adapter("default")
        else:
            pipeline.transformer = get_peft_model(pipeline.transformer, transformer_lora_config)

    transformer = pipeline.transformer
    transformer_trainable_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))

    ema = EMAModuleWrapper(transformer_trainable_parameters, decay=0.9, update_step_interval=8, device=accelerator.device)
    
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Optimizer
    if config.train.use_8bit_adam:
        import bitsandbytes as bnb
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        transformer_trainable_parameters,
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )

    # ============================================================
    # 数据集准备
    # ============================================================

    is_alternate_mode = getattr(config, 'training_mode', None) == "alternate"
    is_mixed_mode = (config.prompt_fn == "mixed")

    if is_alternate_mode:
        # ============ 交替训练模式 ============
        # 按 epoch 交替训练数据集
        # 每个数据集有自己的 epochs_per_cycle 配置
        alternate_datasets = config.alternate_datasets

        # 为每个数据集创建 Dataset 和 DataLoader
        train_datasets = {}
        train_samplers = {}
        train_dataloaders = {}
        test_datasets = {}
        test_dataloaders = {}
        reward_fns = {}
        test_reward_fns = {}

        for ds_config in alternate_datasets:
            ds_name = ds_config["name"]
            ds_path = ds_config["path"]
            ds_prompt_fn = ds_config["prompt_fn"]

            # 创建数据集
            if ds_prompt_fn == "geneval":
                train_ds = GenevalPromptDataset(ds_path, 'train')
                test_ds = GenevalPromptDataset(ds_path, 'test')
                collate_fn = GenevalPromptDataset.collate_fn
            else:  # general_ocr 等
                train_ds = TextPromptDataset(ds_path, 'train')
                test_ds = TextPromptDataset(ds_path, 'test')
                collate_fn = TextPromptDataset.collate_fn

            train_datasets[ds_name] = train_ds
            test_datasets[ds_name] = test_ds

            # 创建 train sampler 和 dataloader
            train_sampler = DistributedKRepeatSampler(
                dataset=train_ds,
                batch_size=config.sample.train_batch_size,
                k=config.sample.num_image_per_prompt,
                num_replicas=accelerator.num_processes,
                rank=accelerator.process_index,
                seed=42
            )
            train_dataloader = DataLoader(
                train_ds,
                batch_sampler=train_sampler,
                num_workers=1,
                collate_fn=collate_fn,
            )
            train_dataloaders[ds_name] = train_dataloader
            train_samplers[ds_name] = train_sampler

            # 创建 test dataloader
            test_dataloader = DataLoader(
                test_ds,
                batch_size=ds_config["test_batch_size"],
                collate_fn=collate_fn,
                shuffle=False,
                num_workers=8,
            )
            test_dataloaders[ds_name] = test_dataloader

            # 创建 reward_fn
            reward_fns[ds_name] = getattr(flow_grpo.rewards, 'multi_score')(
                accelerator.device, ds_config["reward_fn"]
            )
            test_reward_fns[ds_name] = reward_fns[ds_name]

        # 准备 dataloaders
        for ds_name in train_dataloaders:
            train_dataloaders[ds_name] = accelerator.prepare(train_dataloaders[ds_name])
        for ds_name in test_dataloaders:
            test_dataloaders[ds_name] = accelerator.prepare(test_dataloaders[ds_name])

        # 获取当前训练数据集的索引
        def get_current_dataset_idx(epoch):
            """根据 epoch 返回当前应该训练的数据集索引"""
            total_epochs_in_cycle = sum(ds["epochs_per_cycle"] for ds in alternate_datasets)
            epoch_in_cycle = epoch % total_epochs_in_cycle

            cumulative_epochs = 0
            for idx, ds in enumerate(alternate_datasets):
                if epoch_in_cycle < cumulative_epochs + ds["epochs_per_cycle"]:
                    return idx
                cumulative_epochs += ds["epochs_per_cycle"]
            return 0

        dataset_info = alternate_datasets
        dataset_names = [ds["name"] for ds in alternate_datasets]
        is_mixed_mode = False  # 交替模式不是混合模式

        # ================================================================
        # Per-dataset reference transformers for OPD KL reward
        # Each dataset uses its own kl_ref_lora_path as LoRA+base reference.
        # Stored in a dict: { dataset_name: ref_transformer }
        # ================================================================
        ref_transformers = {}
        is_opd_mode = False
        if config.train.beta > 0:
            for ds_config in alternate_datasets:
                ds_name = ds_config["name"]
                kl_ref_path = ds_config.get("kl_ref_lora_path")
                if kl_ref_path:
                    is_opd_mode = True
                    ref_pipe = StableDiffusion3Pipeline.from_pretrained(config.pretrained.model)
                    ref_pipe.transformer.to(accelerator.device)
                    ref_trans = PeftModel.from_pretrained(ref_pipe.transformer, kl_ref_path)
                    ref_trans.set_adapter("default")
                    ref_trans.eval()
                    ref_transformers[ds_name] = ref_trans
                    if accelerator.is_main_process:
                        logger.info(f"[OPD] Loaded kl_ref_lora for dataset '{ds_name}': {kl_ref_path}")
                else:
                    if accelerator.is_main_process:
                        logger.info(f"[OPD] No kl_ref_lora_path for dataset '{ds_name}', using base model as reference.")
                    ref_transformers[ds_name] = None

    elif is_mixed_mode:
        # 混合数据集模式
        mixed_dataset = MixedPromptDatasetV2(config.mixed_datasets)
        dataset_info = mixed_dataset.get_dataset_info()

        # 为每个数据集创建单独的 reward_fn
        reward_fns = {}
        for ds_info in dataset_info:
            reward_fns[ds_info["name"]] = getattr(flow_grpo.rewards, 'multi_score')(
                accelerator.device, ds_info["reward_fn"]
            )

        # 创建混合数据迭代器
        train_data_iterator = MixedDataIterator(
            mixed_dataset=mixed_dataset,
            batch_size=config.sample.train_batch_size,
            k=config.sample.num_image_per_prompt,
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index,
            seed=42
        )

        # 为每个数据集创建单独的 test dataloader
        test_dataloaders = {}
        test_reward_fns = {}
        for ds_info in dataset_info:
            ds_name = ds_info["name"]
            if ds_name == "geneval":
                test_ds = GenevalPromptDataset(ds_info["path"], 'test')
                test_dl = DataLoader(
                    test_ds,
                    batch_size=ds_info["test_batch_size"],
                    collate_fn=GenevalPromptDataset.collate_fn,
                    shuffle=False,
                    num_workers=8,
                )
            else:
                test_ds = TextPromptDataset(ds_info["path"], 'test')
                test_dl = DataLoader(
                    test_ds,
                    batch_size=ds_info["test_batch_size"],
                    collate_fn=TextPromptDataset.collate_fn,
                    shuffle=False,
                    num_workers=8,
                )
            # 修复：对每个 test dataloader 进行 prepare
            test_dl = accelerator.prepare(test_dl)
            test_dataloaders[ds_name] = test_dl
            test_reward_fns[ds_name] = reward_fns[ds_name]

    elif config.prompt_fn == "general_ocr":
        train_dataset = TextPromptDataset(config.dataset, 'train')
        test_dataset = TextPromptDataset(config.dataset, 'test')

        train_sampler = DistributedKRepeatSampler( 
            dataset=train_dataset,
            batch_size=config.sample.train_batch_size,
            k=config.sample.num_image_per_prompt,
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index,
            seed=42
        )

        train_dataloader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=1,
            collate_fn=TextPromptDataset.collate_fn,
        )

        test_dataloader = DataLoader(
            test_dataset,
            batch_size=config.sample.test_batch_size,
            collate_fn=TextPromptDataset.collate_fn,
            shuffle=False,
            num_workers=8,
        )
    
    elif config.prompt_fn == "geneval":
        train_dataset = GenevalPromptDataset(config.dataset, 'train')
        test_dataset = GenevalPromptDataset(config.dataset, 'test')

        train_sampler = DistributedKRepeatSampler( 
            dataset=train_dataset,
            batch_size=config.sample.train_batch_size,
            k=config.sample.num_image_per_prompt,
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index,
            seed=42
        )

        train_dataloader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=1,
            collate_fn=GenevalPromptDataset.collate_fn,
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=config.sample.test_batch_size,
            collate_fn=GenevalPromptDataset.collate_fn,
            shuffle=False,
            num_workers=8,
        )
    else:
        raise NotImplementedError(f"Unsupported prompt_fn: {config.prompt_fn}")

    # Initialize opd mode vars for non-alternate modes (single / mixed)
    if not is_alternate_mode:
        ref_transformers = {}
        is_opd_mode = False

    # 修复：negative pooled prompt embed
    neg_prompt_embed, neg_pooled_prompt_embed = compute_text_embeddings(
        [""], text_encoders, tokenizers, max_sequence_length=128, device=accelerator.device
    )
    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.train_batch_size, 1, 1)
    train_neg_prompt_embeds = neg_prompt_embed.repeat(config.train.batch_size, 1, 1)
    sample_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.sample.train_batch_size, 1)  # 修复
    train_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.train.batch_size, 1)  # 修复

    if config.sample.num_image_per_prompt == 1:
        config.per_prompt_stat_tracking = False
    if config.per_prompt_stat_tracking:
        stat_tracker = PerPromptStatTracker(config.sample.global_std)

    autocast = contextlib.nullcontext if config.use_lora else accelerator.autocast

    # 准备模型
    transformer, optimizer = accelerator.prepare(transformer, optimizer)
    
    # 准备 train_dataloader（混合模式不需要 prepare，因为它使用自定义迭代器）
    if not is_mixed_mode:
        train_dataloader = accelerator.prepare(train_dataloader)
        test_dataloader = accelerator.prepare(test_dataloader)

    executor = futures.ThreadPoolExecutor(max_workers=8)

    # 训练统计
    samples_per_epoch = (
        config.sample.train_batch_size
        * accelerator.num_processes
        * config.sample.num_batches_per_epoch
    )
    total_train_batch_size = (
        config.train.batch_size
        * accelerator.num_processes
        * config.train.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Sample batch size per device = {config.sample.train_batch_size}")
    logger.info(f"  Train batch size per device = {config.train.batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.train.gradient_accumulation_steps}")
    logger.info(f"  Total samples per epoch = {samples_per_epoch}")
    logger.info(f"  Total train batch size = {total_train_batch_size}")
    logger.info(f"  Alternate mode = {is_alternate_mode}")
    logger.info(f"  Mixed mode = {is_mixed_mode}")
    if is_opd_mode:
        logger.info(f"  OPD mode = True (per-dataset kl_ref_lora_path)")
        for ds_name in ref_transformers:
            kl_ref = alternate_datasets[[d["name"] for d in alternate_datasets].index(ds_name)].get("kl_ref_lora_path") if is_alternate_mode else "N/A"
            logger.info(f"    Dataset '{ds_name}' ref: {kl_ref}")

    epoch = 0
    global_step = 0
    train_iter = None
    current_train_dataloader = None
    current_train_sampler = None

    while True:
        # ============================================================
        # 确定当前 epoch 使用的数据集
        # ============================================================
        if is_alternate_mode:
            current_ds_idx = get_current_dataset_idx(epoch)
            current_ds_name = dataset_names[current_ds_idx]
            current_ds_info = alternate_datasets[current_ds_idx]

            # 切换训练 dataloader
            if current_train_dataloader is None or current_train_dataloader != train_dataloaders[current_ds_name]:
                current_train_dataloader = train_dataloaders[current_ds_name]
                current_train_sampler = train_samplers[current_ds_name]
                train_iter = iter(current_train_dataloader)
                if accelerator.is_main_process:
                    logger.info(f"[Epoch {epoch}] Switching to dataset: {current_ds_name}")

        # ============================================================
        # 评估阶段
        # ============================================================
        pipeline.transformer.eval()
        if epoch % config.eval_freq == 0 and epoch>0:
            if is_alternate_mode:
                # 交替模式下评估所有数据集
                for ds_name in dataset_names:
                    current_test_dataloader = test_dataloaders[ds_name]
                    current_test_reward_fn = test_reward_fns[ds_name]
                    if accelerator.is_main_process:
                        logger.info(f"[Eval] Evaluating dataset: {ds_name}")
                    eval(pipeline, current_test_dataloader, text_encoders, tokenizers, config,
                         accelerator, global_step, current_test_reward_fn, executor, autocast,
                         num_train_timesteps, ema, transformer_trainable_parameters)
            elif is_mixed_mode:
                eval_mixed(pipeline, test_dataloaders, test_reward_fns, dataset_info,
                          text_encoders, tokenizers, config, accelerator, global_step,
                          executor, autocast, num_train_timesteps, ema, transformer_trainable_parameters)
            else:
                eval(pipeline, test_dataloader, text_encoders, tokenizers, config,
                     accelerator, global_step, eval_reward_fn, executor, autocast,
                     num_train_timesteps, ema, transformer_trainable_parameters)

        if epoch % config.save_freq == 0 and epoch > 0 and accelerator.is_main_process:
            save_ckpt(config.save_dir, transformer, global_step, accelerator,
                     ema, transformer_trainable_parameters, config)

        # ============================================================
        # 采样阶段
        # ============================================================
        pipeline.transformer.eval()
        samples = []
        prompts = []
        current_ds_name_for_sample = current_ds_name if is_alternate_mode else None

        # 确定要使用的数据集信息
        if is_alternate_mode:
            sample_reward_fn = reward_fns[current_ds_name]
            sample_dataset_id = dataset_names.index(current_ds_name)
        else:
            sample_reward_fn = reward_fn
            sample_dataset_id = None

        for i in tqdm(
            range(config.sample.num_batches_per_epoch),
            desc=f"Epoch {epoch}: sampling" + (f" ({current_ds_name_for_sample})" if is_alternate_mode else ""),
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            if is_alternate_mode:
                current_train_sampler.set_epoch(epoch * config.sample.num_batches_per_epoch + i)
            elif is_mixed_mode:
                train_data_iterator.set_epoch(epoch * config.sample.num_batches_per_epoch + i)
            else:
                train_sampler.set_epoch(epoch * config.sample.num_batches_per_epoch + i)

            batch = next(train_iter)

            if is_mixed_mode:
                batch_prompts, prompt_metadata, dataset_ids = batch
                prompts.extend(batch_prompts)
            else:
                batch_prompts, prompt_metadata = batch
                prompts.extend(batch_prompts)
                dataset_ids = None

            prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
                batch_prompts, 
                text_encoders, 
                tokenizers, 
                max_sequence_length=128, 
                device=accelerator.device
            )
            prompt_ids = tokenizers[0](
                batch_prompts,
                padding="max_length",
                max_length=256,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(accelerator.device)

            # 采样
            if config.sample.same_latent:
                generator = create_generator(batch_prompts, base_seed=epoch*10000+i)
            else:
                generator = None
            with autocast():
                with torch.no_grad():
                    images, latents, log_probs = pipeline_with_logprob(
                        pipeline,
                        prompt_embeds=prompt_embeds,
                        pooled_prompt_embeds=pooled_prompt_embeds,
                        negative_prompt_embeds=sample_neg_prompt_embeds,
                        negative_pooled_prompt_embeds=sample_neg_pooled_prompt_embeds,
                        num_inference_steps=config.sample.num_steps,
                        guidance_scale=config.sample.guidance_scale,
                        output_type="pt",
                        height=config.resolution,
                        width=config.resolution, 
                        noise_level=config.sample.noise_level,
                        generator=generator
                )

            latents = torch.stack(latents, dim=1)
            log_probs = torch.stack(log_probs, dim=1)

            timesteps = pipeline.scheduler.timesteps.repeat(
                config.sample.train_batch_size, 1
            )

            # 计算 reward
            if is_alternate_mode:
                # 交替模式：使用当前数据集的 reward_fn
                rewards = executor.submit(sample_reward_fn, images, batch_prompts, prompt_metadata, only_strict=True)
            elif is_mixed_mode:
                ds_id_to_indices = {}
                for idx, ds_id in enumerate(dataset_ids):
                    if ds_id not in ds_id_to_indices:
                        ds_id_to_indices[ds_id] = []
                    ds_id_to_indices[ds_id].append(idx)

                sample_rewards = {}
                for ds_id, indices in ds_id_to_indices.items():
                    ds_name = dataset_info[ds_id]["name"]
                    ds_reward_fn = reward_fns[ds_name]
                    batch_images = images[indices]
                    batch_prompts_sub = [batch_prompts[i] for i in indices]
                    batch_metadata = [prompt_metadata[i] for i in indices]
                    sample_rewards[ds_id] = executor.submit(
                        ds_reward_fn, batch_images, batch_prompts_sub, batch_metadata, only_strict=True
                    )

                merged_rewards = {}
                for ds_id, reward_future in sample_rewards.items():
                    reward_result, _ = reward_future.result()
                    for key, value in reward_result.items():
                        if key not in merged_rewards:
                            merged_rewards[key] = [None] * len(images)
                        for idx, global_idx in enumerate(ds_id_to_indices[ds_id]):
                            merged_rewards[key][global_idx] = value[idx]
                merged_rewards = {k: np.array(v) for k, v in merged_rewards.items()}

                class MockFuture:
                    def __init__(self, result):
                        self._result = result
                    def result(self):
                        return self._result, {}

                sample_rewards = MockFuture((merged_rewards, {}))
                rewards = sample_rewards
            else:
                rewards = executor.submit(reward_fn, images, batch_prompts, prompt_metadata, only_strict=True)

            time.sleep(0)

            sample_dict = {
                "prompt_ids": prompt_ids,
                "prompt_embeds": prompt_embeds,
                "pooled_prompt_embeds": pooled_prompt_embeds,
                "timesteps": timesteps,
                "latents": latents[:, :-1],
                "next_latents": latents[:, 1:],
                "log_probs": log_probs,
                "rewards": rewards,
            }

            if is_mixed_mode:
                sample_dict["dataset_ids"] = dataset_ids

            samples.append(sample_dict)

        # 等待所有 reward 计算完成
        for sample in tqdm(
            samples,
            desc="Waiting for rewards",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            rewards, reward_metadata = sample["rewards"].result()
            sample["rewards"] = {
                key: torch.as_tensor(value, device=accelerator.device).float()
                for key, value in rewards.items()
            }

        # 整理 samples
        samples = {
            k: torch.cat([s[k] for s in samples], dim=0)
            if not isinstance(samples[0][k], dict)
            else {
                sub_key: torch.cat([s[k][sub_key] for s in samples], dim=0)
                for sub_key in samples[0][k]
            }
            for k in samples[0].keys()
        }

        # WandB 日志
        if epoch % 10 == 0 and accelerator.is_main_process:
            with tempfile.TemporaryDirectory() as tmpdir:
                num_samples = min(15, len(images))
                sample_indices = random.sample(range(len(images)), num_samples)

                for idx, i in enumerate(sample_indices):
                    image = images[i]
                    pil = Image.fromarray(
                        (image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                    )
                    pil = pil.resize((config.resolution, config.resolution))
                    pil.save(os.path.join(tmpdir, f"{idx}.jpg"))

                sampled_prompts = [prompts[i] for i in sample_indices]
                # 获取当前 samples 中的 avg reward
                current_samples_avg = samples["rewards"]["avg"].cpu().numpy()
                sampled_rewards = [current_samples_avg[i] for i in sample_indices]

                log_dict = {
                    "images": [
                        wandb.Image(
                            os.path.join(tmpdir, f"{idx}.jpg"),
                            caption=f"{prompt:.100} | avg: {avg_reward:.2f}",
                        )
                        for idx, (prompt, avg_reward) in enumerate(zip(sampled_prompts, sampled_rewards))
                    ],
                    "current_dataset": current_ds_name_for_sample if is_alternate_mode else "mixed" if is_mixed_mode else "single",
                }

                if is_mixed_mode:
                    sampled_dataset_ids = [samples["dataset_ids"][i] for i in sample_indices]
                    for idx, (ds_id, prompt, avg_reward) in enumerate(zip(sampled_dataset_ids, sampled_prompts, sampled_rewards)):
                        ds_name = dataset_info[ds_id]["name"]
                        log_dict["images"][idx].caption = f"[{ds_name}] {prompt:.100} | avg: {avg_reward:.2f}"
                elif is_alternate_mode:
                    for idx, (prompt, avg_reward) in enumerate(zip(sampled_prompts, sampled_rewards)):
                        log_dict["images"][idx].caption = f"[{current_ds_name_for_sample}] {prompt:.100} | avg: {avg_reward:.2f}"

                wandb.log(log_dict, step=global_step)

        samples["rewards"]["ori_avg"] = samples["rewards"]["avg"]
        samples["rewards"]["avg"] = samples["rewards"]["avg"].unsqueeze(1).repeat(1, num_train_timesteps)
        gathered_rewards = {key: accelerator.gather(value) for key, value in samples["rewards"].items()}
        gathered_rewards = {key: value.cpu().numpy() for key, value in gathered_rewards.items()}

        # Reward 日志
        if accelerator.is_main_process:
            reward_log_dict = {
                "epoch": epoch,
                "current_dataset": current_ds_name_for_sample if is_alternate_mode else "mixed" if is_mixed_mode else "single"
            }

            if is_mixed_mode:
                dataset_ids_np = samples.get("dataset_ids", None)
                if dataset_ids_np is not None:
                    dataset_ids_np = dataset_ids_np.cpu().numpy() if hasattr(dataset_ids_np, 'cpu') else np.array(dataset_ids_np)
                    gathered_dataset_ids = accelerator.gather(
                        torch.as_tensor(dataset_ids_np, device=accelerator.device)
                    ).cpu().numpy()

                    for ds_idx, ds_info in enumerate(dataset_info):
                        ds_name = ds_info["name"]
                        mask = gathered_dataset_ids == ds_idx
                        for key, value in gathered_rewards.items():
                            if '_strict_accuracy' not in key and '_accuracy' not in key:
                                ds_mean = value[mask].mean()
                                reward_log_dict[f"reward_{ds_name}_{key}"] = ds_mean

                for key, value in gathered_rewards.items():
                    if '_strict_accuracy' not in key and '_accuracy' not in key:
                        reward_log_dict[f"reward_{key}"] = value.mean()
            else:
                for key, value in gathered_rewards.items():
                    if '_strict_accuracy' not in key and '_accuracy' not in key:
                        reward_log_dict[f"reward_{key}"] = value.mean()

            wandb.log(reward_log_dict, step=global_step)

        # Per-prompt stat tracking
        if config.per_prompt_stat_tracking:
            prompt_ids = accelerator.gather(samples["prompt_ids"]).cpu().numpy()
            prompts = pipeline.tokenizer.batch_decode(prompt_ids, skip_special_tokens=True)
            advantages = stat_tracker.update(prompts, gathered_rewards['avg'])
            if accelerator.is_local_main_process:
                print("len(prompts)", len(prompts))
                print("len unique prompts", len(set(prompts)))

            group_size, trained_prompt_num = stat_tracker.get_stats()
            zero_std_ratio, reward_std_mean = calculate_zero_std_ratio(prompts, gathered_rewards)

            if accelerator.is_main_process:
                wandb.log(
                    {
                        "group_size": group_size,
                        "trained_prompt_num": trained_prompt_num,
                        "zero_std_ratio": zero_std_ratio,
                        "reward_std_mean": reward_std_mean,
                    },
                    step=global_step,
                )
            stat_tracker.clear()
        else:
            advantages = (gathered_rewards['avg'] - gathered_rewards['avg'].mean()) / (gathered_rewards['avg'].std() + 1e-4)

        # Ungather advantages
        advantages = torch.as_tensor(advantages)
        samples["advantages"] = (
            advantages.reshape(accelerator.num_processes, -1, advantages.shape[-1])[accelerator.process_index]
            .to(accelerator.device)
        )
        if accelerator.is_local_main_process:
            print("advantages: ", samples["advantages"].abs().mean())

        del samples["rewards"]
        del samples["prompt_ids"]

        # Mask
        mask = (samples["advantages"].abs().sum(dim=1) != 0)
        num_batches = config.sample.num_batches_per_epoch
        true_count = mask.sum()
        if true_count % num_batches != 0:
            false_indices = torch.where(~mask)[0]
            num_to_change = num_batches - (true_count % num_batches)
            if len(false_indices) >= num_to_change:
                random_indices = torch.randperm(len(false_indices))[:num_to_change]
                mask[false_indices[random_indices]] = True
        if accelerator.is_main_process:
            wandb.log(
                {"actual_batch_size": mask.sum().item() // config.sample.num_batches_per_epoch},
                step=global_step,
            )
        samples = {k: v[mask] for k, v in samples.items()}

        total_batch_size, num_timesteps = samples["timesteps"].shape
        assert num_timesteps == config.sample.num_steps

        # ============================================================
        # 训练阶段
        # ============================================================
        for inner_epoch in range(config.train.num_inner_epochs):
            perm = torch.randperm(total_batch_size, device=accelerator.device)
            samples = {k: v[perm] for k, v in samples.items()}

            samples_batched = {
                k: v.reshape(-1, total_batch_size // config.sample.num_batches_per_epoch, *v.shape[1:])
                for k, v in samples.items()
            }

            samples_batched = [
                dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())
            ]

            pipeline.transformer.train()
            info = defaultdict(list)

            # In OPD mode, get current dataset's ref transformer for KL reward
            if is_opd_mode and is_alternate_mode:
                current_ref_transformer = ref_transformers.get(current_ds_name)
            else:
                current_ref_transformer = None

            for i, sample in tqdm(
                list(enumerate(samples_batched)),
                desc=f"Epoch {epoch}.{inner_epoch}: training" + (f" ({current_ds_name})" if is_alternate_mode else ""),
                position=0,
                disable=not accelerator.is_local_main_process,
            ):
                if config.train.cfg:
                    embeds = torch.cat(
                        [train_neg_prompt_embeds[:len(sample["prompt_embeds"])], sample["prompt_embeds"]]
                    )
                    pooled_embeds = torch.cat(
                        [train_neg_pooled_prompt_embeds[:len(sample["pooled_prompt_embeds"])], sample["pooled_prompt_embeds"]]
                    )
                else:
                    embeds = sample["prompt_embeds"]
                    pooled_embeds = sample["pooled_prompt_embeds"]

                train_timesteps = [step_index for step_index in range(num_train_timesteps)]
                for j in tqdm(
                    train_timesteps,
                    desc="Timestep",
                    position=1,
                    leave=False,
                    disable=not accelerator.is_local_main_process,
                ):
                    with accelerator.accumulate(transformer):
                        with autocast():
                            # OPD mode: use per-dataset ref transformer for KL reward
                            if is_opd_mode and current_ref_transformer is not None:
                                prev_sample, log_prob, prev_sample_mean, std_dev_t, prev_sample_mean_ref = compute_log_prob_with_ref(
                                    transformer, current_ref_transformer, pipeline, sample, j, embeds, pooled_embeds, config
                                )
                            else:
                                prev_sample, log_prob, prev_sample_mean, std_dev_t = compute_log_prob(
                                    transformer, pipeline, sample, j, embeds, pooled_embeds, config
                                )
                                if config.train.beta > 0:
                                    with torch.no_grad():
                                        with transformer.module.disable_adapter():
                                            _, _, prev_sample_mean_ref, _ = compute_log_prob(
                                                transformer, pipeline, sample, j, embeds, pooled_embeds, config
                                            )
                                else:
                                    prev_sample_mean_ref = None

                        # Determine advantages based on reward_mode
                        reward_mode = config.train.get("reward_mode", "task_only")
                        if reward_mode == "kl_only":
                            # OPD: use step-wise KL reward as advantage
                            if prev_sample_mean_ref is not None and config.train.get("kl_scale", 0) != 0:
                                kl_reward = ((prev_sample_mean - prev_sample_mean_ref) ** 2).mean(dim=(1, 2, 3), keepdim=True) / (2 * std_dev_t ** 2)
                                kl_reward = kl_reward.squeeze(-1).squeeze(-1)
                                kl_norm = config.train.get("kl_norm", "none")
                                if kl_norm == "per_sample":
                                    kl_reward = (kl_reward - kl_reward.mean()) / (kl_reward.std() + 1e-4)
                                elif kl_norm == "per_timestep":
                                    kl_reward = (kl_reward - kl_reward.mean()) / (kl_reward.std() + 1e-4)
                                elif kl_norm == "global":
                                    kl_reward = (kl_reward - kl_reward.mean()) / (kl_reward.std() + 1e-4)
                                advantages = config.train.kl_scale * kl_reward
                            else:
                                advantages = torch.zeros_like(sample["advantages"][:, j])
                            if accelerator.is_main_process and i == 0 and j == 0:
                                wandb.log({"kl_reward": kl_reward.mean().detach().cpu()}, step=global_step)
                        else:
                            advantages = torch.clamp(
                                sample["advantages"][:, j],
                                -config.train.adv_clip_max,
                                config.train.adv_clip_max,
                            )

                        ratio = torch.exp(log_prob - sample["log_probs"][:, j])
                        unclipped_loss = -advantages * ratio
                        clipped_loss = -advantages * torch.clamp(
                            ratio,
                            1.0 - config.train.clip_range,
                            1.0 + config.train.clip_range,
                        )
                        policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))
                        if config.train.beta > 0 and prev_sample_mean_ref is not None:
                            kl_loss = ((prev_sample_mean - prev_sample_mean_ref) ** 2).mean(dim=(1,2,3), keepdim=True) / (2 * std_dev_t ** 2)
                            kl_loss = torch.mean(kl_loss)
                            loss = policy_loss + config.train.beta * kl_loss
                        else:
                            kl_loss = torch.tensor(0.0, device=accelerator.device)
                            loss = policy_loss

                        info["approx_kl"].append(0.5 * torch.mean((log_prob - sample["log_probs"][:, j]) ** 2))
                        info["clipfrac"].append(torch.mean((torch.abs(ratio - 1.0) > config.train.clip_range).float()))
                        info["clipfrac_gt_one"].append(torch.mean((ratio - 1.0 > config.train.clip_range).float()))
                        info["clipfrac_lt_one"].append(torch.mean((1.0 - ratio > config.train.clip_range).float()))
                        info["policy_loss"].append(policy_loss)
                        if config.train.beta > 0:
                            info["kl_loss"].append(kl_loss)
                        info["loss"].append(loss)

                        accelerator.backward(loss)
                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(
                                transformer.parameters(), config.train.max_grad_norm
                            )
                        optimizer.step()
                        optimizer.zero_grad()

                    if accelerator.sync_gradients:
                        info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                        info = accelerator.reduce(info, reduction="mean")
                        info.update({"epoch": epoch, "inner_epoch": inner_epoch})
                        if accelerator.is_main_process:
                            wandb.log(info, step=global_step)
                        global_step += 1
                        info = defaultdict(list)
                if config.train.ema:
                    ema.step(transformer_trainable_parameters, global_step)

        epoch += 1


if __name__ == "__main__":
    app.run(main)
