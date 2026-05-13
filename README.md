# Flow-OPD: On-Policy Distillation for Flow Matching Models



<div align="center">

[![Project Page](https://img.shields.io/badge/🌐_Project_WebPage-green)](https://costaliya.github.io/Flow-OPD/)
[![Paper](https://img.shields.io/badge/📄_Paper-arXiv:2605.08063-red)](https://arxiv.org/abs/2605.08063)
[![Code](https://img.shields.io/badge/🚀_Code-GitHub-blue)](https://github.com/CostaliyA/Flow-OPD)
[![Model](https://img.shields.io/badge/🤗_Model-HuggingFace-yellow)](https://huggingface.co/CostaliyA/Flow-OPD)

> **Flow-OPD** integrates On-Policy Distillation into the Flow Matching pipeline, replacing sparse scalar rewards with dense, trajectory-level, multi-teacher vector field supervision. Evaluated on SD-3.5-Medium, Flow-OPD achieves **+18pt average improvement** over vanilla GRPO and surpasses individual teacher models on OCR and DeQA.

</div>

---
## 🚀 Quick Started
### 1. Environment Set Up
Clone this repository and install packages.
```bash
git clone https://github.com/CostaliyA/Flow-OPD.git
cd Flow_OPD
conda create -n flow_grpo python=3.10.16
pip install -e .
```

### 2. Model Download
To avoid redundant downloads and potential storage waste during multi-GPU training, please pre-download the required models in advance.

**Models**
* **SD3.5**: `stabilityai/stable-diffusion-3.5-medium`
* **Flux**: `black-forest-labs/FLUX.1-dev`

**Reward Models**
* **PickScore**:
  * `laion/CLIP-ViT-H-14-laion2B-s32B-b79K`
  * `yuvalkirstain/PickScore_v1`
* **CLIPScore**: `openai/clip-vit-large-patch14`
* **Aesthetic Score**: `openai/clip-vit-large-patch14`


### 3. Reward Preparation
The steps above only install the current repository. Since each reward model may rely on different versions, combining them in one Conda environment can cause version conflicts. To avoid this, we adopt a remote server setup inspired by ddpo-pytorch. You only need to install the specific reward model you plan to use.

#### GenEval
Please create a new Conda virtual environment and install the corresponding dependencies according to the instructions in [reward-server](https://github.com/yifan123/reward-server).

#### OCR
Please install paddle-ocr:
```bash
pip install paddlepaddle-gpu==2.6.2
pip install paddleocr==2.9.1
pip install python-Levenshtein
```
Then, pre-download the model using the Python command line:
```python
from paddleocr import PaddleOCR
ocr = PaddleOCR(use_angle_cls=False, lang="en", use_gpu=False, show_log=False)
```

#### Pickscore
PickScore requires no additional installation. Note that the original [pickscore](https://huggingface.co/datasets/yuvalkirstain/pickapic_v1) dataset corresponds to `dataset/pickscore` in this repository, containing some NSFW prompts. We strongly recommend using [pickapic\_v1\_no\_images\_training\_sfw](https://huggingface.co/datasets/CarperAI/pickapic_v1_no_images_training_sfw), the SFW version of the Pick-a-Pic dataset, which corresponds to `dataset/pickscore_sfw` in this repository.

#### DeQA
Please create a new Conda virtual environment and install the corresponding dependencies according to the instructions in [reward-server](https://github.com/yifan123/reward-server).

#### UnifiedReward
Since `sglang` may conflict with other environments, we recommend creating a new conda environment.
```bash
conda create -n sglang python=3.10.16
conda activate sglang
pip install "sglang[all]"
```
We use sglang to deploy the reward service. After installing sglang, please run the following command to launch UnifiedReward:
```bash
python -m sglang.launch_server --model-path CodeGoat24/UnifiedReward-7b-v1.5 --api-key flowgrpo --port 17140 --chat-template chatml-llava --enable-p2p-check --mem-fraction-static 0.85
```
#### ImageReward
Please install imagereward:
```bash
pip install image-reward
pip install git+https://github.com/openai/CLIP.git
```
#### QwenVL score
Please create a new Conda virtual environment with vllm:
```bash
pip install vllm
bash scripts/single_node/run_qwen_model.sh
```
and then change Line 130 (base_url) in rewards.py

### 4. Dataset Preparation

> **Note:** All training and evaluation prompts are located in the `dataset/` folder. Training prompts follow the format used in [flow-grpo](https://github.com/yifan123/flow_grpo), and evaluation prompts follow [T2I-CompBench](https://github.com/Karine-Huang/T2I-CompBench).

### 5. Start Training

#### 5.1 GRPO-mix
First, the GenEval rewarder and deqa services need to be deployed on other nodes.
```bash
# Master node
bash scripts/multi_node/sd3_mix.sh 0
# Other nodes
bash scripts/multi_node/sd3_mix.sh 1
bash scripts/multi_node/sd3_mix.sh 2
bash scripts/multi_node/sd3_mix.sh 3
```

#### 5.2 Flow-OPD
```bash
bash scripts/single_node/sd3_opd_example.sh
```
A local implementation example of Flow-OPD (single-teacher). It can be easily adapted into OPSD or Teacher-Student Learning.
Multi-teacher systems require interaction between nodes; this is still under review and will be open-sourced vert soon.

## 📊 Evaluation

This section describes how to evaluate your trained LoRA model on **T2I-CompBench**, based on the evaluation pipeline from [STAGE](https://github.com/krennic999/STAGE).

### 1. Generate Images

First, run `run_eval.sh` to generate images for all T2I-CompBench categories:

```bash
bash scripts/single_node/run_eval.sh
```

Modify `run_eval.sh` to set your LoRA path and output directory:

```bash
torchrun --nproc_per_node=8 scripts/eval_t2icompbench.py \
    --lora "path/to/your/lora" \
    --benchmark t2i_compbench \
    --output_dir ./eval_results/compbench_images
```

Images will be saved under `{output_dir}/{category}/samples/`.

### 2. Install T2I-CompBench

Clone the T2I-CompBench repository and install its dependencies:

```bash
git clone https://github.com/Karine-Huang/T2I-CompBench.git
cd T2I-CompBench
# Follow the installation instructions in their repository
```

### 3. Score Images

Set `T2I_COMP_CODE_ROOT` in `cal_t2i_compbench_value.sh` to point to the cloned T2I-CompBench folder:

```bash
T2I_COMP_CODE_ROOT="/path/to/T2I-CompBench"
```

Then run the scoring script:

```bash
bash cal_t2i_compbench_value.sh
```

Results for each category will be saved as txt files under the corresponding annotation directories.

---

## 🎯 Key Results

| Model | GenEval | OCR Acc. | DeQA | PickScore | Average |
|---|---|---|---|---|---|
| SD-3.5-M (base) | 0.63 | 0.59 | 4.07 | 21.64 | 0.72 |
| GRPO-Mix (best baseline) | 0.73 | 0.83 | 4.33 | 21.84 | 0.82 |
| **Flow-OPD (Merge Init)** | **0.92** | **0.94** | **4.35** | **23.08** | **0.90** |

- ✨ **+18pt** average improvement over base model
- 🚀 **+8pt** improvement over GRPO-Mix (best baseline)
- 📊 **0.92** GenEval score (base: 0.63)
- 📝 **0.94** OCR accuracy (base: 0.59)

---

## 🔬 Method Overview

Flow-OPD decouples expertise acquisition from model unification through a two-stage process:

1. **🧊 Cold Start Initialization** — SFT or Model Merging to initialize the student model
2. **👨‍🏫 Multi-Teacher On-Policy Distillation** — Dense vector field supervision from multiple teachers

The key innovations include:

- **⚡ On-Policy Sampling (SDE)**: Stochastic exploration via SDE for diverse trajectory sampling
- **🔀 Multi-Teacher Dense Labeling**: Each teacher (GenEval, OCR, DeQA, PickScore) acts as a Generative Reward Model returning a full vector field
- **🎨 MAR (Manifold Anchor Regularization)**: KL regularization from a frozen aesthetic teacher prevents aesthetic degradation

---

## 📋 Todo List
The code is being gradually open-sourced, optimized, and refactored. Please feel free to contact me if you have any questions.

### 🔄 In Progress

- [ ] Release full training code

### ✅ Completed

- [x] Release model weights ([HuggingFace](https://huggingface.co/CostaliyA/Flow-OPD))
- [x] Release paper ([arXiv](https://arxiv.org/abs/2605.08063))

---

## 🎨 Qualitative Results

### Overview

![Teaser](assets/teaser.png)

### Comparison

![Comparison](assets/compare.png)

### More Results (1/3)

![More Results 1](assets/more1.png)

### More Results (2/3)

![More Results 2](assets/more2.png)

### More Results (3/3)

![More Results 3](assets/more3.png)

---

## 📚 Citation

```bibtex
@misc{fang2026flowopdonpolicydistillationflow,
      title={Flow-OPD: On-Policy Distillation for Flow Matching Models},
      author={Zhen Fang and Wenxuan Huang and Yu Zeng and Yiming Zhao and Shuang Chen and Kaituo Feng and Yunlong Lin and Lin Chen and Zehui Chen and Shaosheng Cao and Feng Zhao},
      year={2026},
      eprint={2605.08063},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2605.08063},
}
```

---

## 🙏 Acknowledgements

This repo is based on [flow-grpo](https://github.com/yifan123/flow_grpo). We also build upon [STAGE](https://github.com/krennic999/STAGE) for T2I-CompBench evaluation. We thank the authors for their valuable contributions to the AIGC community.
