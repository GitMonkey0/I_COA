from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from typing import Optional, List
import torch
from PIL import Image
import base64
from io import BytesIO
import re


class VLLM:
    def __init__(
            self,
            model: str,
            max_model_len: int = 10000,
            gpu_memory_utilization: float = 0.9,
            tensor_parallel_size: int = None,
            enable_lora: bool = False,
            trust_remote_code: bool = True,
            limit_mm_per_prompt: Optional[dict] = None
    ):
        """
        Initialize the VLLMGenerator with the specified model and configurations.

        :param model: Name of the model to load
        :param max_model_len: Maximum length of the model's context
        :param gpu_memory_utilization: Fraction of GPU memory to utilize
        :param tensor_parallel_size: Number of GPUs to use for tensor parallelism
        :param enable_lora: Whether to enable LoRA support
        :param trust_remote_code: Whether to trust remote code for custom models
        :param limit_mm_per_prompt: Limits for multi-modal inputs per prompt
        """
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

        # Initialize processor for multi-modal models
        self.is_multimodal = False
        try:
            from transformers import AutoProcessor
            self.processor = AutoProcessor.from_pretrained(model)
            self.is_multimodal = True
        except:
            self.processor = None

    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64 string"""
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            return base64.b64encode(buffered.getvalue()).decode()

    def _process_input(self, input_str: str) -> dict:
        """
        Process input string to extract text and images.
        Supports image tags like <image>path/to/image.jpg</image> in the text.

        :param input_str: Input string potentially containing image tags
        :return: Dictionary with processed content
        """
        # Find all image tags in the input
        image_tags = re.findall(r'<image>(.*?)</image>', input_str)
        images = [self._encode_image(img_path) for img_path in image_tags]

        # Remove image tags to get clean text
        text = re.sub(r'<image>.*?</image>', '', input_str).strip()

        return {
            "text": text,
            "images": images
        }

    def _create_prompt(self, input_data: dict, system_prompt: str, enable_thinking: bool) -> dict:
        """
        Create prompt structure based on input content.

        :param input_data: Processed input data (text + images)
        :param system_prompt: System prompt to prepend
        :param enable_thinking: Whether to enable thinking mode
        :return: Dictionary with prompt structure
        """
        messages = [{"role": "system", "content": system_prompt}]

        content = []
        # Add images first if they exist
        if input_data["images"]:
            for img in input_data["images"]:
                content.append({
                    "type": "image",
                    "image": f"data:image/jpeg;base64,{img}",
                })

        # Add text if it exists
        if input_data["text"]:
            content.append({
                "type": "text",
                "text": input_data["text"],
            })

        messages.append({
            "role": "user",
            "content": content
        })

        # Get the text prompt using chat template
        tokenizer = self.llm.get_tokenizer()
        text_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking
        )

        # Prepare multi-modal data if needed
        mm_data = {}
        if self.is_multimodal and input_data["images"]:
            mm_data = {"image": [img["image"] for img in content if img["type"] == "image"]}

        return {
            "prompt": text_prompt,
            "multi_modal_data": mm_data
        }

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
        """
        Generate responses for multiple inputs.
        Supports both pure text and text with embedded images.

        For text with images, use <image>path/to/image.jpg</image> tags in the input string.

        :param inputs: List of input strings (can contain <image> tags)
        :param system_prompt: System prompt to prepend to each input
        :param enable_thinking: Whether to enable thinking mode
        :param temperature: Temperature parameter for sampling
        :param max_tokens: Maximum number of tokens to generate
        :param top_p: Top-p parameter for sampling
        :param lora_path: Path to the LoRA adapter (optional)
        :return: List of generated response texts
        """
        # Process all inputs to extract text and images
        processed_inputs = [self._process_input(input_str) for input_str in inputs]

        # Create prompts for all inputs
        prompts = []
        for input_data in processed_inputs:
            prompts.append(self._create_prompt(input_data, system_prompt, enable_thinking))

        # Separate pure text prompts from multi-modal prompts
        text_prompts = []
        mm_prompts = []
        for prompt in prompts:
            if prompt["multi_modal_data"]:
                mm_prompts.append(prompt)
            else:
                text_prompts.append(prompt["prompt"])

        results = []

        # Process pure text prompts
        if text_prompts:
            text_outputs = self.llm.generate(
                prompts=text_prompts,
                sampling_params=SamplingParams(
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens
                ),
                lora_request=LoRARequest("adapter", 1, lora_path) if lora_path is not None else None
            )
            results.extend([output.outputs[0].text for output in text_outputs])

        # Process multi-modal prompts
        if mm_prompts and self.is_multimodal:
            mm_outputs = self.llm.generate(
                mm_prompts,
                sampling_params=SamplingParams(
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens
                ),
                lora_request=LoRARequest("adapter", 1, lora_path) if lora_path is not None else None
            )
            results.extend([output.outputs[0].text for output in mm_outputs])

        return results

# from vllm import LLM, SamplingParams
# from vllm.lora.request import LoRARequest
# from typing import Optional, Dict, List
# import torch
#
# class VLLM:
#     def __init__(
#         self,
#         model: str,
#         max_model_len: int = 10000,
#         gpu_memory_utilization: float = 0.9,
#         tensor_parallel_size: int = None,
#         enable_lora: bool = False
#     ):
#         """
#         Initialize the VLLMGenerator with the specified model and configurations.
#
#         :param model: Name of the model to load.
#         :param max_model_len: Maximum length of the model's context.
#         :param gpu_memory_utilization: Fraction of GPU memory to utilize.
#         :param tensor_parallel_size: Number of GPUs to use for tensor parallelism.
#         :param enable_lora: Whether to enable LoRA support.
#         """
#         if tensor_parallel_size is None:
#             tensor_parallel_size = torch.cuda.device_count()
#         self.llm = LLM(
#             model=model,
#             max_model_len=max_model_len,
#             gpu_memory_utilization=gpu_memory_utilization,
#             tensor_parallel_size=tensor_parallel_size,
#             enable_lora=enable_lora
#         )
#
#     def _create_prompts(self, inputs: List[str], system_prompt: str, enable_thinking: bool = False) -> List[str]:
#         """
#         Create chat prompts for each input using the system prompt.
#
#         :param inputs: List of input prompts.
#         :param system_prompt: System prompt to prepend to each input.
#         :param enable_thinking: Whether to enable thinking mode in the chat template.
#         :return: List of formatted prompts.
#         """
#         tokenizer = self.llm.get_tokenizer()
#         prompts = []
#         for data in inputs:
#             message = [
#                 {"role": "system", "content": system_prompt},
#                 {"role": "user", "content": data}
#             ]
#             prompt = tokenizer.apply_chat_template(
#                 message,
#                 tokenize=False,
#                 add_generation_prompt=True,
#                 enable_thinking=enable_thinking
#             )
#             prompts.append(prompt)
#         return prompts
#
#     def _generate_initial_responses(self, prompts: List[str], temperature: float, max_tokens: int, top_p: float, lora_path: Optional[str]) -> List[str]:
#         """
#         Generate initial responses for the provided prompts.
#
#         :param prompts: List of formatted prompts.
#         :param temperature: Temperature parameter for sampling.
#         :param max_tokens: Maximum number of tokens to generate.
#         :param top_p: Top-p parameter for sampling.
#         :param lora_path: Path to the LoRA adapter (optional).
#         :return: List of generated response texts.
#         """
#         outputs = self.llm.generate(
#             prompts=prompts,
#             sampling_params=SamplingParams(
#                 temperature=temperature,
#                 top_p=top_p,
#                 max_tokens=max_tokens
#             ),
#             lora_request=LoRARequest("adapter", 1, lora_path) if lora_path is not None else None
#         )
#         return [output.outputs[0].text for output in outputs]
#
#     def generate(
#         self,
#         inputs: List[str],
#         system_prompt: str = "You are a helpful assistant.",
#         enable_thinking: bool = False,
#         temperature: float = 0.8,
#         max_tokens: int = 10000,
#         top_p: float = 0.95,
#         lora_path: Optional[str] = None
#     ) -> List[str]:
#         """
#         Parallelly generate responses for multiple inputs.
#
#         :param inputs: List of input prompts.
#         :param system_prompt: System prompt to prepend to each input.
#         :param enable_thinking: Whether to enable thinking mode in the chat template.
#         :param temperature: Temperature parameter for sampling.
#         :param max_tokens: Maximum number of tokens to generate.
#         :param top_p: Top-p parameter for sampling.
#         :param lora_path: Path to the LoRA adapter (optional).
#         :return: List of generated response texts.
#         """
#         prompts = self._create_prompts(inputs, system_prompt, enable_thinking)
#         results = self._generate_initial_responses(prompts, temperature, max_tokens, top_p, lora_path)
#
#         return results