import ml_collections
import imp
import os

base = imp.load_source("base", os.path.join(os.path.dirname(__file__), "base.py"))

def compressibility():
    config = base.get_config()

    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")

    config.use_lora = True

    config.sample.batch_size = 8
    config.sample.num_batches_per_epoch = 4

    config.train.batch_size = 4
    config.train.gradient_accumulation_steps = 2

    # prompting
    config.prompt_fn = "general_ocr"

    # rewards
    config.reward_fn = {"jpeg_compressibility": 1}
    config.per_prompt_stat_tracking = True
    return config

def general_ocr_sd3_8gpu_opd():
    gpu_number = 8
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/ocr")
    config.run_name="opd_ocr"
    # sd3.5 medium
    config.pretrained.model = "path/to/stable-diffusion-3.5-medium"
    config.train.kl_ref_lora_path = "path/to/SD3.5M-FlowGRPO-Text"
    config.train.mar_lora = ""#path tou your mar teacher  

    config.sample.num_steps = 10
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5

    config.resolution = 512
    config.sample.train_batch_size = 8
    config.sample.num_image_per_prompt = 16
    config.sample.num_batches_per_epoch = int(16/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt))
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.sample.test_batch_size = 16 # 16 is a special design, the test set has a total of 1018, to make 8*16*n as close as possible to 1018, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    # kl loss
    config.train.beta = 0.04
    config.train.kl_reward_level="step_wise"
    config.train.kl_scale = -1
    config.train.reward_mode = "kl_only" # no outcome
    config.sample.global_std = True
    config.sample.same_latent = False
    config.train.ema = True
    # A large num_epochs is intentionally set here. Training will be manually stopped once sufficient
    config.save_freq = 60 # epoch
    config.use_lora = True
    config.eval_freq = 60
    config.save_dir = 'logs/ocr/sd3.5-M/opd'
    config.reward_fn = {
        "ocr": 1.0,
    }
    
    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    return config

def mix_opd_8gpu():
    gpu_number = 8
    config = base.get_config()

    config.pretrained.model = "path/to/stable-diffusion-3.5-medium"

    # Sampling
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5
    config.sample.eval_guidance_scale = 4.5
    config.resolution = 512
    config.sample.global_std = True
    config.sample.same_latent = False

    # Batch & dataloader
    config.sample.train_batch_size = 4
    config.sample.num_image_per_prompt = 16
    config.sample.num_batches_per_epoch = int(
        16/ (gpu_number * config.sample.train_batch_size / config.sample.num_image_per_prompt)
    )
    assert config.sample.num_batches_per_epoch % 2 == 0, \
        "num_batches_per_epoch must be even for gradient accumulation"
    config.sample.test_batch_size = 9

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch // 2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.learning_rate = 1e-4
    # KL ratio (global beta across all datasets)
    config.train.beta = 0.04
    config.train.kl_reward_level = "step_wise"
    config.train.kl_scale = -1
    config.train.reward_mode = "kl_only"  # no outcome reward, pure KL-based OPD
    config.train.ema = True

    # LoRA
    config.use_lora = True

    # Checkpointing
    config.save_freq = 60
    config.eval_freq = 60
    config.save_dir = 'logs/multi_opd/sd3.5-M/opd-mix'

    # Per-prompt stat tracking
    config.per_prompt_stat_tracking = True

    # =============================================================
    # Multi-dataset alternate training configuration
    # Each entry maps to its own kl_ref_lora_path (LoRA+base reference)
    # =============================================================
    config.alternate_datasets = [
        {
            "name": "ocr",
            "path": os.path.join(os.getcwd(), "dataset/ocr"),
            "prompt_fn": "general_ocr",
            "reward_fn": {"ocr": 1.0},
            "kl_ref_lora_path": "path/SD3.5M-FlowGRPO-Text",
            "epochs_per_cycle": 1,
            "test_batch_size": 9,
        },
        {
            "name": "geneval",
            "path": os.path.join(os.getcwd(), "dataset/geneval"),
            "prompt_fn": "geneval",
            "reward_fn": {"geneval": 1.0},
            "kl_ref_lora_path": "path/SD3.5M-FlowGRPO-GenEval",
            "epochs_per_cycle": 3,
            "test_batch_size": 11,
        },
        # {
        #     "name": "pickscore",
        #     "path": os.path.join(os.getcwd(), "dataset/pickscore"),
        #     "prompt_fn": "general_ocr",
        #     "reward_fn": {"pickscore": 1.0},
        #     "kl_ref_lora_path": "/SD3.5M-FlowGRPO-pick",
        #     "epochs_per_cycle": 1,
        #     "test_batch_size": 9,
        # },
    ]
    config.training_mode = "alternate"

    config.run_name = "mix_opd_8gpu"
    return config


def geneval_ocr_deqa_pickscore_32gpu():
    gpu_number = 32
    config = compressibility()
    config.pretrained.model = "path/to/stable-diffusion-3.5-medium"
    config.run_name="flow-grpo-mix"
    # sd3.5 medium

    # 采样步数
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5
    config.sample.eval_guidance_scale = 4.5

    config.resolution = 512

    # ============ 交替训练数据集配置 ============
    # 训练策略：每 2 个 epoch geneval 对应 1 个 epoch ocr
    # geneval 训练集: 50000 样本, test 集: 2211 样本
    # ocr 训练集: 19652 样本, test 集: 1017 样本
    config.alternate_datasets = [
        {
            "name": "geneval",
            "path": os.path.join(os.getcwd(), "dataset/geneval"),
            "prompt_fn": "geneval",
            "reward_fn": {"geneval": 0.8,"deqa":0.2},
            "epochs_per_cycle": 3, 
            "test_batch_size": 11, 
        },
        {
            "name": "ocr",
            "path": os.path.join(os.getcwd(), "dataset/ocr"),
            "prompt_fn": "general_ocr",
            "reward_fn": {"ocr": 0.8,"deqa":0.2},
            "epochs_per_cycle": 1,  
            "test_batch_size": 9,  
        },
                {
            "name": "pickscore",
            "path": os.path.join(os.getcwd(), "dataset/pickscore"),
            "prompt_fn": "general_ocr",
            "reward_fn": {"pickscore": 0.8,"deqa":0.2},
            "epochs_per_cycle": 1, 
            "test_batch_size": 9, 
        },]
    config.training_mode = "alternate"  # 标记为交替训练模式


    config.sample.train_batch_size = 3
    config.sample.num_image_per_prompt = 24
    config.sample.num_batches_per_epoch = int(48/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt))
    assert config.sample.num_batches_per_epoch % 2 == 0, \
        "num_batches_per_epoch must be even for gradient accumulation"

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch // 2  # = 2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.beta = 0.04
    config.train.learning_rate = 1e-4

    config.sample.global_std = True
    config.sample.same_latent = False
    config.train.ema = True

    # LoRA 配置
    config.use_lora = True

    config.save_freq = 60  # epoch
    config.eval_freq = 30  # 每 30 epoch 评估一次（两个数据集都会评估）
    config.save_dir = 'logs/geneval_ocr/flow-grpo-mix'


    config.per_prompt_stat_tracking = True
    return config

def get_config(name):
    return globals()[name]()
