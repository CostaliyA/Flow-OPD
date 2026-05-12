from PIL import Image
import torch
import re
import base64
from io import BytesIO
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

def pil_image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    encoded_image_text = base64.b64encode(buffered.getvalue()).decode("utf-8")
    base64_qwen = f"data:image;base64,{encoded_image_text}"
    return base64_qwen

def extract_scores(output_text):
    scores = []
    for text in output_text:
        match = re.search(r'<Score>(\d+)</Score>', text)
        if match:
            scores.append(float(match.group(1))/5)
        else:
            scores.append(0)
    return scores

def extract_quality_scores(output_text):
    scores = []
    for text in output_text:
        match = re.search(r'<QualityScore>(\d+)</QualityScore>', text)
        if match:
            scores.append(float(match.group(1))/5)
        else:
            scores.append(0)
    return scores

def extract_instruction_scores(output_text):
    scores = []
    for text in output_text:
        match = re.search(r'<InstructionScore>(\d+)</InstructionScore>', text)
        if match:
            scores.append(float(match.group(1))/5)
        else:
            scores.append(0)
    return scores

def extract_overall_scores(output_text):
    scores = []
    for text in output_text:
        match = re.search(r'<OverallScore>(\d+)</OverallScore>', text)
        if match:
            scores.append(float(match.group(1))/5)
        else:
            scores.append(0)
    return scores

class QwenVLScorer(torch.nn.Module):
    """Qwen-VL scorer (local). Supports two modes:
    - quality-only: prompt is None or empty → only image quality
    - quality+instruction: prompt is provided → evaluates both in a single call
    """
    def __init__(self, device="cuda", dtype=torch.bfloat16):
        super().__init__()
        self.device = device
        self.dtype = dtype

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct",
            torch_dtype=self.dtype,
            attn_implementation="flash_attention_2",
            device_map=None,
        ).to(self.device)
        self.model.requires_grad_(False)
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", use_fast=True)

        self._quality_only_task = '''Your role is to evaluate the aesthetic quality score of given images.
1. Bad: Extremely blurry, underexposed with significant noise, indiscernible subjects, and chaotic composition.
2. Poor: Noticeable blur, poor lighting, washed-out colors, and awkward composition with cut-off subjects.
3. Fair: In focus with adequate lighting, dull colors, decent composition but lacks creativity.
4. Good: Sharp, good exposure, vibrant colors, thoughtful composition with a clear focal point.
5. Excellent: Exceptional clarity, perfect exposure, rich colors, masterful composition with emotional impact.

Please first provide a detailed analysis of the evaluation process, including the criteria for judging aesthetic quality, within the <Thought> tag. Then, give a final score from 1 to 5 within the <Score> tag.
<Thought>
[Analyze the evaluation process in detail here]
</Thought>
<Score>X</Score>'''

        self._quality_instruction_task = '''Your role is to evaluate the generated image on three dimensions: aesthetic quality, instruction following, and an overall score that prioritizes instruction adherence.

=== Aesthetic Quality ===
1. Bad: Extremely blurry, underexposed with significant noise, indiscernible subjects, and chaotic composition.
2. Poor: Noticeable blur, poor lighting, washed-out colors, and awkward composition with cut-off subjects.
3. Fair: In focus with adequate lighting, dull colors, decent composition but lacks creativity.
4. Good: Sharp, good exposure, vibrant colors, thoughtful composition with a clear focal point.
5. Excellent: Exceptional clarity, perfect exposure, rich colors, masterful composition with emotional impact.

=== Instruction Following ===
Instruction: {prompt}
1. Bad: The image completely ignores or contradicts the instruction.
2. Poor: The image only vaguely relates to the instruction, missing most key elements.
3. Fair: The image partially follows the instruction, but misses or distorts some important elements.
4. Good: The image mostly follows the instruction with minor deviations.
5. Excellent: The image perfectly and faithfully follows the instruction.

=== Overall Score ===
The overall score should PRIMARILY reflect instruction adherence. An image that perfectly follows the instruction but has slightly lower aesthetic quality should score higher than a beautiful image that ignores the instruction.
1 = Bad, 2 = Poor, 3 = Fair, 4 = Good, 5 = Excellent

Please first provide a detailed analysis of all three dimensions within the <Thought> tag. Then, give three scores from 1 to 5 within the <QualityScore>, <InstructionScore>, and <OverallScore> tags.
<Thought>
[Analyze the aesthetic quality, instruction adherence, and overall impression in detail here]
</Thought>
<QualityScore>X</QualityScore>
<InstructionScore>Y</InstructionScore>
<OverallScore>Z</OverallScore>'''

    def _build_messages(self, image_base64, prompt=None):
        task = self._quality_only_task if prompt is None else self._quality_instruction_task.format(prompt=prompt)
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_base64},
                    {"type": "text", "text": task},
                ],
            },
        ]

    def _call_task(self, images, prompts, task_template):
        messages = []
        for img, p in zip(images, prompts):
            image_base64 = pil_image_to_base64(img)
            messages.append(self._build_messages(image_base64, p))
        texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages
        ]
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)
        generated_ids = self.model.generate(**inputs, max_new_tokens=2048)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        return self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

    @torch.no_grad()
    def __call__(self, prompts, images):
        if prompts is None or all(p is None for p in prompts):
            output_texts = self._call_task(images, [None] * len(images), self._quality_only_task)
            return extract_scores(output_texts)

        output_texts = self._call_task(images, prompts, self._quality_instruction_task)
        rewards = []
        for out in output_texts:
            o = extract_overall_scores([out])[0]
            rewards.append(o)
        return rewards

# Usage example


class QwenVLScorerRemote:
    """Qwen-VL scorer that calls a remote vLLM server via OpenAI-compatible API.

    Two modes (determined by whether prompts is None/empty):
    - quality-only: prompts is None or empty → only evaluates image quality
    - quality+instruction: prompts is provided → evaluates both in a single call
    """
    def __init__(
        self,
        base_url="http://127.0.0.1:8001/v1",
        api_key="EMPTY",
        model="Qwen/Qwen2-VL-7B-Instruct",
    ):
        from openai import OpenAI

        self.model = model
        self.client = OpenAI(base_url=base_url, api_key=api_key)

        self._quality_only_task = '''Your role is to evaluate the aesthetic quality score of given images.
1. Bad: Extremely blurry, underexposed with significant noise, indiscernible subjects, and chaotic composition.
2. Poor: Noticeable blur, poor lighting, washed-out colors, and awkward composition with cut-off subjects.
3. Fair: In focus with adequate lighting, dull colors, decent composition but lacks creativity.
4. Good: Sharp, good exposure, vibrant colors, thoughtful composition with a clear focal point.
5. Excellent: Exceptional clarity, perfect exposure, rich colors, masterful composition with emotional impact.

Please first provide a detailed analysis of the evaluation process, including the criteria for judging aesthetic quality, within the <Thought> tag. Then, give a final score from 1 to 5 within the <Score> tag.
<Thought>
[Analyze the evaluation process in detail here]
</Thought>
<Score>X</Score>'''

        self._quality_instruction_task = '''Your role is to evaluate the generated image on three dimensions: aesthetic quality, instruction following, and an overall score that prioritizes instruction adherence.

=== Aesthetic Quality ===
1. Bad: Extremely blurry, underexposed with significant noise, indiscernible subjects, and chaotic composition.
2. Poor: Noticeable blur, poor lighting, washed-out colors, and awkward composition with cut-off subjects.
3. Fair: In focus with adequate lighting, dull colors, decent composition but lacks creativity.
4. Good: Sharp, good exposure, vibrant colors, thoughtful composition with a clear focal point.
5. Excellent: Exceptional clarity, perfect exposure, rich colors, masterful composition with emotional impact.

=== Instruction Following ===
Instruction: {prompt}
1. Bad: The image completely ignores or contradicts the instruction.
2. Poor: The image only vaguely relates to the instruction, missing most key elements.
3. Fair: The image partially follows the instruction, but misses or distorts some important elements.
4. Good: The image mostly follows the instruction with minor deviations.
5. Excellent: The image perfectly and faithfully follows the instruction.

=== Overall Score ===
The overall score should PRIMARILY reflect instruction adherence. An image that perfectly follows the instruction but has slightly lower aesthetic quality should score higher than a beautiful image that ignores the instruction.
1 = Bad, 2 = Poor, 3 = Fair, 4 = Good, 5 = Excellent
**Be extremely strict.**
Please first provide a detailed analysis of all three dimensions within the <Thought> tag. Then, give three scores from 1 to 5 within the <QualityScore>, <InstructionScore>, and <OverallScore> tags.
<Thought>
[Analyze the aesthetic quality, instruction adherence, and overall impression in detail here]
</Thought>
<QualityScore>X</QualityScore>
<InstructionScore>Y</InstructionScore>
<OverallScore>Z</OverallScore>'''

    def _score_single(self, image, prompt=None):
        task = self._quality_only_task if prompt is None else self._quality_instruction_task.format(prompt=prompt)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": pil_image_to_base64(image)}},
                    {"type": "text", "text": task},
                ],
            }
        ]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0,
            max_tokens=2048,
        )
        return response.choices[0].message.content

    def __call__(self, prompts, images):
        if prompts is None or all(p is None for p in prompts):
            output_texts = [self._score_single(img) for img in images]
            return extract_scores(output_texts)

        rewards = []
        for img, p in zip(images, prompts):
            if p is None:
                out = self._score_single(img)
                rewards.append(extract_scores([out])[0])
            else:
                out = self._score_single(img, p)
                o = extract_overall_scores([out])[0]
                q = extract_quality_scores([out])[0]
                i = extract_instruction_scores([out])[0]
                rewards.append(o)
               # print(f"[QwenVL] quality={q}, instruction={i}, overall={o}")

        return rewards


if __name__ == "__main__":
    from PIL import Image

    img = Image.open("/mnt/tidal-alsh01/dataset/redone/zengyu/fz/code/flux-dev.png").convert("RGB")

    # mode 1: quality only (prompts=None)
    scorer = QwenVLScorerRemote(base_url="http://10.146.231.196:8001/v1", model="")
    scores = scorer(None, [img])
    print(f"quality-only scores: {scores}")

    # mode 2: quality + instruction following
    scores = scorer(["a dog"], [img])
    print(f"quality+instruction scores: {scores}")