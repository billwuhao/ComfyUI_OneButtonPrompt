import random
import re
import os
import torch
from PIL import Image
import numpy as np
from qwen_vl_utils import process_vision_info
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoProcessor,
    BitsAndBytesConfig,
    set_seed,
)

node_dir = os.path.dirname(os.path.abspath(__file__))
comfy_path = os.path.dirname(os.path.dirname(node_dir))
llm_model_path = os.path.join(comfy_path, "models", "LLM")


def hash_seed(seed):
    import hashlib
    # Convert the seed to a string and then to bytes
    seed_bytes = str(seed).encode('utf-8')
    # Create a SHA-256 hash of the seed bytes
    hash_object = hashlib.sha256(seed_bytes)
    # Convert the hash to an integer
    hashed_seed = int(hash_object.hexdigest(), 16)
    # Ensure the hashed seed is within the acceptable range for set_seed
    return hashed_seed % (2**32)


################################### Qwen VLM ###################################
def tensor_to_pil(image_tensor, batch_index=0) -> Image:
    # Convert tensor of shape [batch, height, width, channels] at the batch_index to PIL Image
    image_tensor = image_tensor[batch_index].unsqueeze(0)
    i = 255.0 * image_tensor.cpu().numpy()
    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8).squeeze())
    return img

class QwenVLRun:
    def __init__(self):
        self.model_checkpoint = None
        self.processor = None
        self.model = None
        # self.device = (
        #     torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        # )
        # self.bf16_support = (
        #     torch.cuda.is_available()
        #     and torch.cuda.get_device_capability(self.device)[0] >= 8
        # )

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"default": "", "multiline": True}),
                "model": (
                    [
                        "Qwen2.5-VL-3B-Instruct",
                        # "Qwen2.5-VL-7B-Instruct",
                        # "Qwen2.5-VL-7B-Instruct-bnb-4bit"
                    ],
                    {"default": "Qwen2.5-VL-3B-Instruct"},
                ),
                # "quantization": (
                #     ["none", "4bit", "8bit"],
                #     {"default": "none"},
                # ),
                "keep_model_loaded": ("BOOLEAN", {"default": False}),
                "temperature": (
                    "FLOAT",
                    {"default": 0.7, "min": 0, "max": 1, "step": 0.1},
                ),
                "max_new_tokens": (
                    "INT",
                    {"default": 512, "min": 128, "max": 2048, "step": 1},
                ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "image": ("IMAGE",),
                "next_image": ("IMAGE",),
                "video": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "inference"
    CATEGORY = "MW-OneButtonPrompt"

    def inference(
        self,
        text,
        model,
        # quantization,
        keep_model_loaded,
        temperature,
        max_new_tokens,
        seed,
        image=None,
        next_image =None,
        video=None,
    ):
        
        set_seed(hash_seed(seed))
        self.model_checkpoint = os.path.join(llm_model_path, model)

        if self.processor is None:
            # Define min_pixels and max_pixels:
            # Images will be resized to maintain their aspect ratio
            # within the range of min_pixels and max_pixels.
            min_pixels = 256*28*28
            max_pixels = 1024*28*28 

            self.processor = AutoProcessor.from_pretrained(
                self.model_checkpoint,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )

        if self.model is None:
            # Load the model on the available device(s)
            # if quantization == "4bit":
            #     quantization_config = BitsAndBytesConfig(
            #         load_in_4bit=True,
            #         bnb_4bit_quant_type="nf4",  # 与 Unsloth 默认量化类型一致
            #         bnb_4bit_compute_dtype=torch.bfloat16
            #     )
            # elif quantization == "8bit":
            #     quantization_config = BitsAndBytesConfig(
            #         load_in_8bit=True,
            #     )
            # else:
            #     quantization_config = None

            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_checkpoint,
                torch_dtype="auto", # torch.bfloat16 if self.bf16_support else torch.float16,
                device_map="auto",
                # quantization_config=quantization_config,
            )

        with torch.no_grad():
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text},
                    ],
                }
            ]

            if video is not None:
                video_imgs = [tensor_to_pil(video, i) for i in range(video.size(0))]
                messages[0]["content"].insert(0, {"type": "video", "video": video_imgs})

            # 处理图像输入
            elif image is not None:
                if next_image is not None:
                    pil_next_image = tensor_to_pil(next_image)
                    messages[0]["content"].insert(0, {
                        "type": "image",
                        "image": pil_next_image,
                    })
                
                pil_image = tensor_to_pil(image)
                messages[0]["content"].insert(0, {
                    "type": "image",
                    "image": pil_image,
                })

            # 准备输入
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
                **video_kwargs,
            ).to("cuda")

            # 推理
            try:
                generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                result = self.processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                    temperature=temperature,
                )
            except Exception as e:
                print(f"Error during model inference: {str(e)}")
                raise

            if not keep_model_loaded:
                del self.processor
                del self.model
                self.processor = None
                self.model = None
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

            return result

################################### Qwen LLM ###################################
class QwenLLMRun:
    def __init__(self):
        self.model_checkpoint = None
        self.tokenizer = None
        self.model = None
        # self.device = (
        #     torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        # )
        # self.bf16_support = (
        #     torch.cuda.is_available()
        #     and torch.cuda.get_device_capability(self.device)[0] >= 8
        # )

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (
                    [
                        "Qwen2.5-3B-Instruct",
                        "Qwen2.5-3B-Instruct-Flux",
                        # "Qwen2.5-7B-Instruct",
                        # "Qwen2.5-7B-Instruct-Uncensored-Flux",
                    ],
                    {"default": "Qwen2.5-3B-Instruct-Flux"},
                ),
                "system": ("STRING", {
                        "multiline": True, 
                        "default": "Act like a prompt engineer for Stable Diffusion. You need to give me the most accturate prompt for my input. Don't introduce your message, give ONLY the prompt. Prompt must be around 150 words."}),
                "text": ("STRING", {"multiline": True, "default": "Give me a prompt for Stable Diffusion"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "max_new_tokens": ("INT", {"default": 512, "min": 128, "max": 2048, "step": 1}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": 50, "min": 0}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}),
                "keep_model_loaded": ("BOOLEAN", {"default": False}),
                # "attention": (['flash_attention_2', 'sdpa', 'eager'], {"default": 'sdpa'}),
                # "quantization": (
                #     ["none", "4bit", "8bit"],
                #     {"default": "none"},
                # ),
            }
        }
    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate"
    CATEGORY = "MW-OneButtonPrompt"


    def generate(self, 
                 model, 
                #  attention, 
                #  quantization, 
                 system, 
                 text, 
                 seed, 
                 max_new_tokens, 
                 temperature, 
                 top_k, 
                 top_p, 
                 keep_model_loaded
        ):
        set_seed(hash_seed(seed))
        self.model_checkpoint = os.path.join(llm_model_path, model)

        if self.model is None:
            # Load the model on the available device(s)
            # if quantization == "4bit":
            #     quantization_config = BitsAndBytesConfig(
            #         load_in_4bit=True,
            #         bnb_4bit_quant_type="nf4",  # 与 Unsloth 默认量化类型一致
            #         bnb_4bit_compute_dtype=torch.bfloat16
            #     )
            # elif quantization == "8bit":
            #     quantization_config = BitsAndBytesConfig(
            #         load_in_8bit=True,
            #     )
            # else:
            #     quantization_config = None

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_checkpoint,
                torch_dtype="auto", # torch.bfloat16 if self.bf16_support else torch.float16,
                device_map="auto",
                # attn_implementation=attention,
                # quantization_config = quantization_config,
            )

        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)
        
        messages = [
            {
                "role": "system", 
                "content": system
            },
            {
                "role": "user", 
                "content": text
            }
        ]
            
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            model_inputs.input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            attention_mask=model_inputs.attention_mask,
            pad_token_id=self.tokenizer.eos_token_id
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        if not keep_model_loaded: 
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        return (response, )


################################## DeepSeekRone ##################################
class DeepseekRun:
    ds_model_path = os.path.join(llm_model_path, "DeepScaleR-1.5B-Preview")
    model_paths = {
        "DeepScaleR-1.5B-Preview": ds_model_path, # 可以添加更多模型和路径
    }
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (list(cls.model_paths.keys()), {"tooltip": "models are expected to be in Comfyui/models/LLM folder"}),
                "user_prompt": ("STRING", {"default": "", "multiline": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "max_tokens": ("INT", {"default": 1000, "min": 0, "max": 0xffffffffffffffff}),
                "temperature": ("FLOAT", {"default": 1, "min": 0, "max": 2}),
                "top_k": ("INT", {"default": 50, "min": 0, "max": 101}),
                "top_p": ("FLOAT", {"default": 1, "min": 0, "max": 1}),
                "unload_model": ("BOOLEAN", {
                    "default": False, 
                    "tooltip": "If True, unload the model from memory after execution. Next execution will reload the model."}), # Added unload_model input
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("Prompt", "Think")
    FUNCTION = "dsgen"
    CATEGORY = "MW-OneButtonPrompt"

    _model_cache = {}

    def dsgen(self, model, user_prompt, seed=0, temperature=1.0, max_tokens=1000, top_k=25, top_p=1.0, unload_model=False):
        
        if seed:
            set_seed(hash_seed(seed))

        match = re.search(r'{(.*?)}', user_prompt)

        if match:
            content_in_brackets = match.group(1)

            # 按 "|" 拆分
            options = content_in_brackets.split('|')
            # 随机选择一个
            chosen_option = random.choice(options).strip() # 使用 strip() 去除可能存在的首尾空格
            # 替换原字符串 "{}" 中的内容
            user_prompt = user_prompt.replace(match.group(0), chosen_option)
        else:
            user_prompt = user_prompt

        messages = [
            {"role": "user", "content": user_prompt},
        ]

        model_dict = self.load_model(model)

        if model_dict is None: # Check if model loading failed
            return ("Error loading model. Check console.", )
        
        device = model_dict["device"] # 获取设备信息
        dsmodel = model_dict["dsmodel"]
        tokenizer = model_dict["tokenizer"]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        # 构造输入时直接使用 device
        model_inputs = tokenizer([text], return_tensors="pt").to(device) 

        generated_ids = dsmodel.generate(
            **model_inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=True
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        responses = response.split("</think>")
        if len(responses) < 2:
            think = response.replace("<think>", "").strip()
            answer = ""
        else:
            think = responses[0].replace("<think>", "").strip()
            answer = responses[1].strip()

        if unload_model: # Check if unload_model is True
            self.unload_model_from_cache(model) # Unload the model if requested
            print(f"DeepseekRun: Model '{model}' unloaded from cache.") # Inform user model is unloaded

        return (answer, think,)

    def load_model(self, model):
        model_path = self.model_paths.get(model)
        if model in self._model_cache:
            return self._model_cache[model]

        try:
            dsmodel = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",  # 自动分配
                torch_dtype="auto",
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            tokenizer = AutoTokenizer.from_pretrained(model_path)

            # 获取实际使用的设备
            device = next(dsmodel.parameters()).device

            model_info = {
                "dsmodel": dsmodel,
                "tokenizer": tokenizer,
                "device": device
            }
            self._model_cache[model] = model_info
            print(f"DeepseekRun: Model '{model}' loaded to cache.") # Inform user model is loaded
            return model_info
        except Exception as e:
            print(f"DeepseekRun: Error loading model {model} from {model_path}: {e}")
            return None
                
    def unload_model_from_cache(self, model): # New method to unload model
        if model in self._model_cache:
            model_info = self._model_cache.pop(model) # Remove model from cache
            del model_info # Optionally delete model_info dictionary to release references (may not be strictly needed in Python)
            torch.cuda.empty_cache() # Clear CUDA cache to try and free VRAM
            print(f"DeepseekRun: Model '{model}' removed from cache and CUDA cache cleared.") # Inform user of unload action
        else:
            print(f"DeepseekRun: Model '{model}' not found in cache, cannot unload.") # Inform user if model was not in cache
