from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from typing import Optional, List, Dict, Any
import torch
from PIL import Image
import base64
from io import BytesIO
import re
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info  # 引入 process_vision_info


class VLLM:
    def __init__(
            self,
            model: str,
            max_model_len: int = 10000,
            gpu_memory_utilization: float = 0.9,
            tensor_parallel_size: Optional[int] = None,
            enable_lora: bool = False,
            trust_remote_code: bool = True,
            limit_mm_per_prompt: Optional[dict] = None
    ):
        # 初始化 LLM
        if tensor_parallel_size is None:
            tensor_parallel_size = torch.cuda.device_count()

        self.llm = LLM(
            model=model,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            enable_lora=enable_lora,
            trust_remote_code=trust_remote_code,
            limit_mm_per_prompt=limit_mm_per_prompt
        )

        # 初始化多模态处理器
        self.is_multimodal = False
        try:
            self.processor = AutoProcessor.from_pretrained(model)
            self.is_multimodal = True
        except:
            self.processor = None

        # 获取 tokenizer
        self.tokenizer = self.llm.get_tokenizer()

    def _encode_image(self, image_path: str) -> str:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            return base64.b64encode(buffered.getvalue()).decode()

    def _process_input(self, input_str: str) -> Dict[str, Any]:
        # 解析是否包含 <image> 标签
        image_tags = re.findall(r'<image>(.*?)</image>', input_str)
        images = [self._encode_image(path) for path in image_tags]
        text = re.sub(r'<image>.*?</image>', '', input_str).strip()
        return {"text": text, "images": images, "is_multimodal": bool(images)}

    def _create_text_prompt(self, text: str, system_prompt: str, enable_thinking: bool) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking
        )

    def _create_mm_prompt(self, input_data: Dict[str, Any], system_prompt: str) -> Dict[str, Any]:
        if not self.processor:
            raise ValueError("Multi-modal processing requires a valid processor")

        # 构建消息列表
        messages = [{"role": "system", "content": system_prompt}]
        content = []
        for img_b64 in input_data["images"]:
            content.append({"type": "image", "image": f"data:image/jpeg;base64,{img_b64}"})
        if input_data["text"]:
            content.append({"type": "text", "text": input_data["text"]})
        messages.append({"role": "user", "content": content})

        # 使用 process_vision_info 提取 vision inputs
        image_inputs, _ = process_vision_info(messages)
        mm_data = {"image": image_inputs} if image_inputs else {}

        # 构建文本 prompt
        text_prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        return {"prompt": text_prompt, "multi_modal_data": mm_data}

    def generate(
            self,
            inputs: List[str],
            system_prompt: str = "You are a helpful assistant.",
            enable_thinking: bool = False,
            temperature: float = 0.8,
            max_tokens: int = 10000,
            top_p: float = 0.95,
            lora_path: Optional[str] = None
    ) -> List[str]:
        processed = [self._process_input(s) for s in inputs]
        text_prompts, mm_prompts = [], []

        for data in processed:
            if data["is_multimodal"]:
                if not self.is_multimodal:
                    raise ValueError("Multi-modal input detected but model not initialized with multi-modal support.")
                mm_prompts.append(self._create_mm_prompt(data, system_prompt))
            else:
                text_prompts.append(self._create_text_prompt(data["text"], system_prompt, enable_thinking))

        results = []
        # 文本生成
        if text_prompts:
            outs = self.llm.generate(
                prompts=text_prompts,
                sampling_params=SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens),
                lora_request=LoRARequest("adapter", 1, lora_path) if lora_path else None
            )
            results.extend([o.outputs[0].text for o in outs])

        # 多模态生成
        if mm_prompts and self.is_multimodal:
            prompts = [p for p in mm_prompts if p]
            outs = self.llm.generate(
                prompts,
                sampling_params=SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens),
                lora_request=LoRARequest("adapter", 1, lora_path) if lora_path else None
            )
            results.extend([o.outputs[0].text for o in outs])
        return results

