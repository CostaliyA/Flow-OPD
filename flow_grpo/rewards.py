from PIL import Image
import io
import numpy as np
import torch
from collections import defaultdict

def jpeg_incompressibility():
    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        images = [Image.fromarray(image) for image in images]
        buffers = [io.BytesIO() for _ in images]
        for image, buffer in zip(images, buffers):
            image.save(buffer, format="JPEG", quality=95)
        sizes = [buffer.tell() / 1000 for buffer in buffers]
        return np.array(sizes), {}

    return _fn


def jpeg_compressibility():
    jpeg_fn = jpeg_incompressibility()

    def _fn(images, prompts, metadata):
        rew, meta = jpeg_fn(images, prompts, metadata)
        return -rew/500, meta

    return _fn

def aesthetic_score():
    from flow_grpo.aesthetic_scorer import AestheticScorer

    scorer = AestheticScorer(dtype=torch.float32).cuda()

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8)
        else:
            images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            images = torch.tensor(images, dtype=torch.uint8)
        scores = scorer(images)
        return scores, {}

    return _fn

def clip_score(device):
    from flow_grpo.clip_scorer import ClipScorer

    scorer = ClipScorer(device=device)

    def _fn(images, prompts, metadata):
        if not isinstance(images, torch.Tensor):
            images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            images = torch.tensor(images, dtype=torch.uint8)/255.0
        scores = scorer(images, prompts)
        return scores, {}

    return _fn

def image_similarity_score(device):
    from flow_grpo.clip_scorer import ClipScorer

    scorer = ClipScorer(device=device).cuda()

    def _fn(images, ref_images):
        if not isinstance(images, torch.Tensor):
            images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            images = torch.tensor(images, dtype=torch.uint8)/255.0
        if not isinstance(ref_images, torch.Tensor):
            ref_images = [np.array(img) for img in ref_images]
            ref_images = np.array(ref_images)
            ref_images = ref_images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            ref_images = torch.tensor(ref_images, dtype=torch.uint8)/255.0
        scores = scorer.image_similarity(images, ref_images)
        return scores, {}

    return _fn

def pickscore_score(device):
    from flow_grpo.pickscore_scorer import PickScoreScorer

    scorer = PickScoreScorer(dtype=torch.float32, device=device)

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
            images = [Image.fromarray(image) for image in images]
        scores = scorer(prompts, images)
        return scores, {}

    return _fn

def imagereward_score(device):
    from flow_grpo.imagereward_scorer import ImageRewardScorer

    scorer = ImageRewardScorer(dtype=torch.float32, device=device)

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
            images = [Image.fromarray(image) for image in images]
        prompts = [prompt for prompt in prompts]
        scores = scorer(prompts, images)
        return scores, {}

    return _fn

def qwenvl_score(device):
    from flow_grpo.qwenvl import QwenVLScorer

    scorer = QwenVLScorer(dtype=torch.bfloat16, device=device)

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
            images = [Image.fromarray(image) for image in images]
        prompts = [prompt for prompt in prompts]
        scores = scorer(prompts, images)
        return scores, {}

    return _fn

def qwenvl_score_remote(device):
    from flow_grpo.qwenvl import QwenVLScorerRemote

    scorer = QwenVLScorerRemote(base_url="http://10.144.201.159:8001/v1",model="")

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
            images = [Image.fromarray(image) for image in images]
        prompts = [prompt for prompt in prompts]
        scores = scorer(prompts, images)
        return scores, {}

    return _fn
    
def ocr_score(device):
    from flow_grpo.ocr import OcrScorer

    scorer = OcrScorer()

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        scores = scorer(images, prompts)
        # change tensor to list
        return scores, {}

    return _fn

def video_ocr_score(device):
    from flow_grpo.ocr import OcrScorer_video_or_image

    scorer = OcrScorer_video_or_image()

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            if images.dim() == 4 and images.shape[1] == 3:
                images = images.permute(0, 2, 3, 1) 
            elif images.dim() == 5 and images.shape[2] == 3:
                images = images.permute(0, 1, 3, 4, 2)
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
        scores = scorer(images, prompts)
        # change tensor to list
        return scores, {}

    return _fn

def deqa_score_remote(device):
    """Submits images to DeQA and computes a reward.
    """
    import requests
    from requests.adapters import HTTPAdapter, Retry
    from io import BytesIO
    import pickle

    batch_size = 64
    url = "http://10.144.171.34:18086"
    sess = requests.Session()
    retries = Retry(
        total=1000, backoff_factor=1, status_forcelist=[500], allowed_methods=False
    )
    sess.mount("http://", HTTPAdapter(max_retries=retries))

    def _fn(images, prompts, metadata):
        del prompts
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        images_batched = np.array_split(images, np.ceil(len(images) / batch_size))
        all_scores = []
        for image_batch in images_batched:
            jpeg_images = []

            # Compress the images using JPEG
            for image in image_batch:
                img = Image.fromarray(image)
                buffer = BytesIO()
                img.save(buffer, format="JPEG")
                jpeg_images.append(buffer.getvalue())

            # format for LLaVA server
            data = {
                "images": jpeg_images,
            }
            data_bytes = pickle.dumps(data)

            # send a request to the llava server
            response = sess.post(url, data=data_bytes, timeout=120)
            response_data = pickle.loads(response.content)

            all_scores += response_data["outputs"]

        return all_scores, {}

    return _fn

def geneval_score(device):
    """Submits images to GenEval and computes a reward.
    """
    import requests
    from requests.adapters import HTTPAdapter, Retry
    from io import BytesIO
    import pickle

    batch_size = 64
    url = "http://10.144.169.109:18085"
    sess = requests.Session()
    retries = Retry(
        total=1000, backoff_factor=1, status_forcelist=[500], allowed_methods=False
    )
    sess.mount("http://", HTTPAdapter(max_retries=retries))

    def _fn(images, prompts, metadatas, only_strict):
        del prompts
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        images_batched = np.array_split(images, np.ceil(len(images) / batch_size))
        metadatas_batched = np.array_split(metadatas, np.ceil(len(metadatas) / batch_size))
        all_scores = []
        all_rewards = []
        all_strict_rewards = []
        all_group_strict_rewards = []
        all_group_rewards = []
        for image_batch, metadata_batched in zip(images_batched, metadatas_batched):
            jpeg_images = []

            # Compress the images using JPEG
            for image in image_batch:
                img = Image.fromarray(image)
                buffer = BytesIO()
                img.save(buffer, format="JPEG")
                jpeg_images.append(buffer.getvalue())

            # format for LLaVA server
            data = {
                "images": jpeg_images,
                "meta_datas": list(metadata_batched),
                "only_strict": only_strict,
            }
            data_bytes = pickle.dumps(data)

            # send a request to the llava server
            response = sess.post(url, data=data_bytes, timeout=120)
            response_data = pickle.loads(response.content)

            all_scores += response_data["scores"]
            all_rewards += response_data["rewards"]
            all_strict_rewards += response_data["strict_rewards"]
            all_group_strict_rewards.append(response_data["group_strict_rewards"])
            all_group_rewards.append(response_data["group_rewards"])
        all_group_strict_rewards_dict = defaultdict(list)
        all_group_rewards_dict = defaultdict(list)
        for current_dict in all_group_strict_rewards:
            for key, value in current_dict.items():
                all_group_strict_rewards_dict[key].extend(value)
        all_group_strict_rewards_dict = dict(all_group_strict_rewards_dict)

        for current_dict in all_group_rewards:
            for key, value in current_dict.items():
                all_group_rewards_dict[key].extend(value)
        all_group_rewards_dict = dict(all_group_rewards_dict)

        return all_scores, all_rewards, all_strict_rewards, all_group_rewards_dict, all_group_strict_rewards_dict

    return _fn

def unifiedreward_score_remote(device):
    """Submits images to DeQA and computes a reward.
    """
    import requests
    from requests.adapters import HTTPAdapter, Retry
    from io import BytesIO
    import pickle

    batch_size = 64
    url = "http://10.144.200.49:17140"
    sess = requests.Session()
    retries = Retry(
        total=1000, backoff_factor=1, status_forcelist=[500], allowed_methods=False
    )
    sess.mount("http://", HTTPAdapter(max_retries=retries))

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        images_batched = np.array_split(images, np.ceil(len(images) / batch_size))
        prompts_batched = np.array_split(prompts, np.ceil(len(prompts) / batch_size))

        all_scores = []
        for image_batch, prompt_batch in zip(images_batched, prompts_batched):
            jpeg_images = []

            # Compress the images using JPEG
            for image in image_batch:
                img = Image.fromarray(image)
                buffer = BytesIO()
                img.save(buffer, format="JPEG")
                jpeg_images.append(buffer.getvalue())

            # format for LLaVA server
            data = {
                "images": jpeg_images,
                "prompts": prompt_batch
            }
            data_bytes = pickle.dumps(data)

            # send a request to the llava server
            response = sess.post(url, data=data_bytes, timeout=120)
            print("response: ", response)
            print("response: ", response.content)
            response_data = pickle.loads(response.content)

            all_scores += response_data["outputs"]

        return all_scores, {}

    return _fn

def unifiedreward_score_sglang(device):
    from openai import OpenAI
    import base64
    from io import BytesIO
    import re 

    def pil_image_to_base64(image):
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        encoded_image_text = base64.b64encode(buffered.getvalue()).decode("utf-8")
        base64_qwen = f"data:image;base64,{encoded_image_text}"
        return base64_qwen

    def _extract_scores(text_outputs):
        scores = []
        pattern = r"Final Score:\s*([1-5](?:\.\d+)?)"
        for text in text_outputs:
            match = re.search(pattern, text)
            if match:
                try:
                    scores.append(float(match.group(1)))
                except ValueError:
                    scores.append(0.0)
            else:
                scores.append(0.0)
        return scores

    client = OpenAI(base_url="http://10.144.200.49:17140/v1", api_key="flowgrpo")

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC

        images = [Image.fromarray(image).resize((512, 512)) for image in images]

        text_outputs = []
        for prompt, img in zip(prompts, images):
            question = (
                f"<image>\nYou are given a text caption and a generated image based on that caption. "
                f"Your task is to evaluate this image based on two key criteria:\n"
                f"1. Alignment with the Caption: Assess how well this image aligns with the provided caption. "
                f"Consider the accuracy of depicted objects, their relationships, and attributes as described in the caption.\n"
                f"2. Overall Image Quality: Examine the visual quality of this image, including clarity, detail preservation, "
                f"color accuracy, and overall aesthetic appeal.\n"
                f"Based on the above criteria, assign a score from 1 to 5 after 'Final Score:'.\n"
                f"Your task is provided as follows:\n"
                f"Text Caption: [{prompt}]"
            )
            images_base64 = pil_image_to_base64(img)
            response = client.chat.completions.create(
                model="UnifiedReward-7b-v1.5",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": images_base64},
                            },
                            {
                                "type": "text",
                                "text": question,
                            },
                        ],
                    },
                ],
                temperature=0,
            )
            text_outputs.append(response.choices[0].message.content)

        score = _extract_scores(text_outputs)
        score = [sc / 5.0 for sc in score]
        return score, {}

    return _fn

def multi_score(device, score_dict):
    from concurrent.futures import ThreadPoolExecutor

    score_functions = {
        "deqa": deqa_score_remote,
        "ocr": ocr_score,
        "video_ocr": video_ocr_score,
        "imagereward": imagereward_score,
        "pickscore": pickscore_score,
        "qwenvl": qwenvl_score_remote,
        "aesthetic": aesthetic_score,
        "jpeg_compressibility": jpeg_compressibility,
        "unifiedreward": unifiedreward_score_sglang,
        "geneval": geneval_score,
        "clipscore": clip_score,
        "image_similarity": image_similarity_score,
    }
    score_fns = {}
    for score_name, weight in score_dict.items():
        score_fns[score_name] = score_functions[score_name](device) if 'device' in score_functions[score_name].__code__.co_varnames else score_functions[score_name]()

    # OCR/PaddleOCR uses OpenCV internally, which is not thread-safe,
    # so we run it serially outside the thread pool.
    LOCAL_SERIAL_SCORES = {"ocr", "video_ocr"}

    def _fn(images, prompts, metadata, ref_images=None, only_strict=True):
        # Run thread-unsafe OCR scores first, then parallel HTTP/compute scores
        score_details = {}

        # Serial: OCR (not thread-safe)
        for score_name in score_dict:
            if score_name not in LOCAL_SERIAL_SCORES:
                continue
            weight = score_dict[score_name]
            if score_name == "video_ocr":
                scores, _ = score_fns[score_name](images, prompts, metadata)
            else:
                scores, _ = score_fns[score_name](images, prompts, metadata)
            score_details[score_name] = scores
            weighted_scores = [weight * score for score in scores]
            if 'avg' not in score_details:
                score_details['avg'] = weighted_scores
            else:
                score_details['avg'] = [total + w for total, w in zip(score_details['avg'], weighted_scores)]

        # Parallel: everything else (HTTP-based remote + local GPU scores)
        parallel_names = [n for n in score_dict if n not in LOCAL_SERIAL_SCORES]
        if parallel_names:
            max_workers = min(len(parallel_names), 8)
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {}
                for score_name in parallel_names:
                    weight = score_dict[score_name]
                    if score_name == "geneval":
                        fut = pool.submit(score_fns[score_name], images, prompts, metadata, only_strict)
                    elif score_name == "image_similarity":
                        fut = pool.submit(score_fns[score_name], images, ref_images)
                    else:
                        fut = pool.submit(score_fns[score_name], images, prompts, metadata)
                    futures[score_name] = (fut, weight)

                for score_name, (fut, weight) in futures.items():
                    result = fut.result()
                    if score_name == "geneval":
                        scores, rewards, strict_rewards, group_rewards, group_strict_rewards = result
                        score_details['accuracy'] = rewards
                        score_details['strict_accuracy'] = strict_rewards
                        for key, value in group_strict_rewards.items():
                            score_details[f'{key}_strict_accuracy'] = value
                        for key, value in group_rewards.items():
                            score_details[f'{key}_accuracy'] = value
                    else:
                        scores, _ = result
                    score_details[score_name] = scores
                    weighted_scores = [weight * score for score in scores]
                    if 'avg' not in score_details:
                        score_details['avg'] = weighted_scores
                    else:
                        score_details['avg'] = [total + w for total, w in zip(score_details['avg'], weighted_scores)]

        return score_details, {}

    return _fn



def geneval_ocr_multi_score(device, geneval_weight=0.5, ocr_weight=0.5):
    """
    GenEval 和 OCR 联合训练的奖励函数。
    
    根据每张图像对应的任务类型（geneval 或 ocr）计算相应的奖励，
    然后将奖励加权求和得到最终奖励。
    
    支持两种图像输入模式:
    - Tensor/numpy 数组: 原始图像数据
    - URL 列表: 图像的 URL 地址
    
    Args:
        device: 计算设备
        geneval_weight: GenEval 奖励权重
        ocr_weight: OCR 奖励权重
    """
    geneval_fn = geneval_score(device)
    ocr_fn = ocr_score(device)
    
    def _fn(images, prompts, metadata, task_types, only_strict=True):
        """
        计算联合奖励。
        
        Args:
            images: 生成的图像 (Tensor、List 或 URL 列表)
            prompts: 文本提示列表
            metadata: 元数据列表
            task_types: 任务类型列表，值为 "geneval" 或 "ocr"，可以为 None
            only_strict: 是否只使用严格奖励（仅对 GenEval 有效）
            
        Returns:
            score_details: 包含各类型奖励详情的字典
        """
        # 检测是否为 URL 模式
        is_url_mode = False
        if isinstance(images, (list, tuple)) and len(images) > 0:
            if isinstance(images[0], str) and (images[0].startswith('http://') or images[0].startswith('https://')):
                is_url_mode = True
        
        # 处理图像格式
        if is_url_mode:
            # URL 模式：保持原样传递
            images_list = images
            images_np = images  # URL 列表直接传递
        elif isinstance(images, torch.Tensor):
            images_np = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images_np = images_np.transpose(0, 2, 3, 1)  # NCHW -> NHWC
            images_list = [Image.fromarray(image) for image in images_np]
        else:
            images_list = images
            images_np = np.array([np.array(img) for img in images])
        
        batch_size = len(images_np) if not is_url_mode else len(images)
        
        # 处理 task_types 为 None 的情况
        if task_types is None:
            # 默认全部视为 geneval 任务
            task_types = ['geneval'] * batch_size
        
        # 初始化奖励数组
        geneval_rewards = [0.0] * batch_size
        ocr_rewards = [0.0] * batch_size
        total_scores = [0.0] * batch_size
        
        # 按任务类型分组
        geneval_indices = [i for i, tt in enumerate(task_types) if tt == 'geneval']
        ocr_indices = [i for i, tt in enumerate(task_types) if tt == 'ocr']
        
        # 添加默认的奖励键
        score_details = {
            'avg': [0.0] * batch_size,
            'geneval': [0.0] * batch_size,
            'ocr': [0.0] * batch_size,
            'accuracy': [0.0] * batch_size,
            'strict_accuracy': [0.0] * batch_size,
            'ocr_accuracy': [0.0] * batch_size,
        }
        
        # 计算 GenEval 奖励
        if geneval_indices:
            if is_url_mode:
                geneval_images = [images[i] for i in geneval_indices]
            else:
                geneval_images = images_np[geneval_indices]
            geneval_prompts = [prompts[i] for i in geneval_indices]
            geneval_metadata = [metadata[i] for i in geneval_indices]
            
            ge_scores, ge_rewards, ge_strict_rewards, group_rewards, group_strict_rewards = geneval_fn(
                geneval_images, geneval_prompts, geneval_metadata, only_strict
            )
            
            # 填入 GenEval 奖励
            for idx, ge_idx in enumerate(geneval_indices):
                geneval_rewards[ge_idx] = ge_rewards[idx]
                score_details['geneval'][ge_idx] = ge_scores[idx]
                score_details['accuracy'][ge_idx] = ge_rewards[idx]
                score_details['strict_accuracy'][ge_idx] = ge_strict_rewards[idx]
            
            # 添加 GenEval 类型的详细奖励
            for key, value in group_strict_rewards.items():
                score_details[f'{key}_strict_accuracy'] = [0.0] * batch_size
                for idx, ge_idx in enumerate(geneval_indices):
                    score_details[f'{key}_strict_accuracy'][ge_idx] = value[idx]
            for key, value in group_rewards.items():
                score_details[f'{key}_accuracy'] = [0.0] * batch_size
                for idx, ge_idx in enumerate(geneval_indices):
                    score_details[f'{key}_accuracy'][ge_idx] = value[idx]
        
        # 计算 OCR 奖励
        if ocr_indices:
            if is_url_mode:
                ocr_images = [images[i] for i in ocr_indices]
            else:
                ocr_images = images_np[ocr_indices]
            ocr_prompts = [prompts[i] for i in ocr_indices]
            ocr_metadata = [metadata[i] for i in ocr_indices]
            
            ocr_reward_list, _ = ocr_fn(ocr_images, ocr_prompts, ocr_metadata)
            
            # 填入 OCR 奖励
            for idx, ocr_idx in enumerate(ocr_indices):
                ocr_rewards[ocr_idx] = ocr_reward_list[idx]
                score_details['ocr'][ocr_idx] = ocr_reward_list[idx]
                score_details['ocr_accuracy'][ocr_idx] = ocr_reward_list[idx]
        
        # 计算加权总奖励（根据任务类型选择对应的奖励）
        for i in range(batch_size):
            if task_types[i] == 'geneval':
                total_scores[i] = geneval_rewards[i]
                score_details['avg'][i] = total_scores[i]
            elif task_types[i] == 'ocr':
                total_scores[i] = ocr_rewards[i]
                score_details['avg'][i] = total_scores[i]
        
        return score_details, {}
    
    return _fn
def main():
    import torchvision.transforms as transforms

    image_paths = [
        "nasa.jpg",
    ]

    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert to tensor
    ])

    images = torch.stack([transform(Image.open(image_path).convert('RGB')) for image_path in image_paths])
    prompts=[
        'A astronaut’s glove floating in zero-g with "NASA 2049" on the wrist',
    ]
    metadata = {}  # Example metadata
    score_dict = {
        "unifiedreward": 1.0
    }
    # Initialize the multi_score function with a device and score_dict
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scoring_fn = multi_score(device, score_dict)
    # Get the scores
    scores, _ = scoring_fn(images, prompts, metadata)
    # Print the scores
    print("Scores:", scores)


if __name__ == "__main__":
    main()
