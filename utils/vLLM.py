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


# from vllm import LLM, SamplingParams
# from vllm.lora.request import LoRARequest
# from typing import Optional, List, Union, Dict, Any
# import torch
# from PIL import Image
# import base64
# from io import BytesIO
# import re
#
#
# class VLLM:
#     def __init__(
#             self,
#             model: str,
#             max_model_len: int = 10000,
#             gpu_memory_utilization: float = 0.9,
#             tensor_parallel_size: int = None,
#             enable_lora: bool = False,
#             trust_remote_code: bool = True,
#             limit_mm_per_prompt: Optional[dict] = None
#     ):
#         """
#         Initialize the VLLMGenerator with the specified model and configurations.
#
#         :param model: Name of the model to load
#         :param max_model_len: Maximum length of the model's context
#         :param gpu_memory_utilization: Fraction of GPU memory to utilize
#         :param tensor_parallel_size: Number of GPUs to use for tensor parallelism
#         :param enable_lora: Whether to enable LoRA support
#         :param trust_remote_code: Whether to trust remote code for custom models
#         :param limit_mm_per_prompt: Limits for multi-modal inputs per prompt
#         """
#         if tensor_parallel_size is None:
#             tensor_parallel_size = torch.cuda.device_count()
#
#         self.llm = LLM(
#             model=model,
#             max_model_len=max_model_len,
#             gpu_memory_utilization=gpu_memory_utilization,
#             tensor_parallel_size=tensor_parallel_size,
#             enable_lora=enable_lora,
#             trust_remote_code=trust_remote_code,
#             limit_mm_per_prompt=limit_mm_per_prompt
#         )
#
#         # Initialize processor for multi-modal models
#         self.is_multimodal = False
#         try:
#             from transformers import AutoProcessor
#             self.processor = AutoProcessor.from_pretrained(model)
#             self.is_multimodal = True
#         except:
#             self.processor = None
#
#         # Get tokenizer for text-only processing
#         self.tokenizer = self.llm.get_tokenizer()
#
#     def _encode_image(self, image_path: str) -> str:
#         """Encode image to base64 string"""
#         with Image.open(image_path) as img:
#             img = img.convert("RGB")
#             buffered = BytesIO()
#             img.save(buffered, format="JPEG")
#             return base64.b64encode(buffered.getvalue()).decode()
#
#     def _process_input(self, input_str: str) -> dict:
#         """
#         Process input string to extract text and images.
#         Supports image tags like <image>path/to/image.jpg</image> in the text.
#
#         :param input_str: Input string potentially containing image tags
#         :return: Dictionary with processed content
#         """
#         # Find all image tags in the input
#         image_tags = re.findall(r'<image>(.*?)</image>', input_str)
#         images = [self._encode_image(img_path) for img_path in image_tags]
#
#         # Remove image tags to get clean text
#         text = re.sub(r'<image>.*?</image>', '', input_str).strip()
#
#         return {
#             "text": text,
#             "images": images,
#             "is_multimodal": bool(images)  # 明确标记是否为多模态输入
#         }
#
#     def _create_text_prompt(self, text: str, system_prompt: str, enable_thinking: bool) -> str:
#         """
#         Create text-only prompt using tokenizer
#
#         :param text: Input text
#         :param system_prompt: System prompt
#         :param enable_thinking: Whether to enable thinking mode
#         :return: Formatted prompt string
#         """
#         messages = [
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": text}
#         ]
#         return self.tokenizer.apply_chat_template(
#             messages,
#             tokenize=False,
#             add_generation_prompt=True,
#             enable_thinking=enable_thinking
#         )
#
#     def _create_mm_prompt(self, input_data: dict, system_prompt: str) -> Dict[str, Any]:
#         """
#         Create multi-modal prompt using processor
#
#         :param input_data: Processed input data with images and text
#         :param system_prompt: System prompt
#         :return: Dictionary with prompt and multi-modal data
#         """
#         if not self.processor:
#             raise ValueError("Multi-modal processing requires a valid processor")
#
#         messages = [{"role": "system", "content": system_prompt}]
#
#         # Build content list with images and text
#         content = []
#         for img in input_data["images"]:
#             content.append({
#                 "type": "image",
#                 "image": f"data:image/jpeg;base64,{img}",
#             })
#
#         if input_data["text"]:
#             content.append({
#                 "type": "text",
#                 "text": input_data["text"],
#             })
#
#         messages.append({
#             "role": "user",
#             "content": content
#         })
#
#         # Process with processor for multi-modal models
#         text_prompt = self.processor.apply_chat_template(
#             messages,
#             tokenize=False,
#             add_generation_prompt=True
#         )
#
#         # Prepare multi-modal inputs
#         mm_data = {"image": [img["image"] for img in content if img["type"] == "image"]}
#
#         return {
#             "prompt": text_prompt,
#             "multi_modal_data": mm_data
#         }
#
#     def generate(
#             self,
#             inputs: List[str],
#             system_prompt: str = "You are a helpful assistant.",
#             enable_thinking: bool = False,
#             temperature: float = 0.8,
#             max_tokens: int = 10000,
#             top_p: float = 0.95,
#             lora_path: Optional[str] = None
#     ) -> List[str]:
#         """
#         Generate responses for multiple inputs.
#         Supports both pure text and text with embedded images.
#
#         For text with images, use <image>path/to/image.jpg</image> tags in the input string.
#
#         :param inputs: List of input strings (can contain <image> tags)
#         :param system_prompt: System prompt to prepend to each input
#         :param enable_thinking: Whether to enable thinking mode
#         :param temperature: Temperature parameter for sampling
#         :param max_tokens: Maximum number of tokens to generate
#         :param top_p: Top-p parameter for sampling
#         :param lora_path: Path to the LoRA adapter (optional)
#         :return: List of generated response texts
#         """
#         # Process all inputs to extract text and images
#         processed_inputs = [self._process_input(input_str) for input_str in inputs]
#
#         # Separate text-only and multi-modal inputs
#         text_prompts = []
#         mm_prompts = []
#
#         for input_data in processed_inputs:
#             if input_data["is_multimodal"]:
#                 if not self.is_multimodal:
#                     raise ValueError(
#                         "Multi-modal input detected but model is not initialized with multi-modal support. "
#                         "Please check your model initialization."
#                     )
#                 try:
#                     mm_prompts.append(
#                         self._create_mm_prompt(input_data, system_prompt)
#                     )
#                 except Exception as e:
#                     print(f"Error creating multi-modal prompt: {e}")
#                     mm_prompts.append(None)
#             else:
#                 text_prompts.append(
#                     self._create_text_prompt(input_data["text"], system_prompt, enable_thinking)
#                 )
#
#         results = []
#
#         # Process text-only prompts
#         if text_prompts:
#             text_outputs = self.llm.generate(
#                 prompts=[p for p in text_prompts if p is not None],
#                 sampling_params=SamplingParams(
#                     temperature=temperature,
#                     top_p=top_p,
#                     max_tokens=max_tokens
#                 ),
#                 lora_request=LoRARequest("adapter", 1, lora_path) if lora_path is not None else None
#             )
#             # Map results back to original order
#             text_results = [output.outputs[0].text if output else "" for output in text_outputs]
#             results.extend(text_results)
#
#         # Process multi-modal prompts
#         if mm_prompts and self.is_multimodal:
#             valid_mm_prompts = [p for p in mm_prompts if p is not None]
#             mm_outputs = self.llm.generate(
#                 valid_mm_prompts,
#                 sampling_params=SamplingParams(
#                     temperature=temperature,
#                     top_p=top_p,
#                     max_tokens=max_tokens
#                 ),
#                 lora_request=LoRARequest("adapter", 1, lora_path) if lora_path is not None else None
#             )
#             # Map results back to original order
#             mm_results = [output.outputs[0].text if output else "" for output in mm_outputs]
#             results.extend(mm_results)
#
#         return results
