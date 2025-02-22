import random
import re
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


class DeepseekRun:
    node_dir = os.path.dirname(os.path.abspath(__file__))
    comfy_path = os.path.dirname(os.path.dirname(node_dir))
    model_path = os.path.join(comfy_path, "models", "LLM")
    ds_model_path = os.path.join(model_path, "DeepScaleR-1.5B-Preview")
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

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("STRING",)
    FUNCTION = "dsgen"
    CATEGORY = "MW-OneButtonPrompt"

    _model_cache = {}

    def dsgen(self, model, user_prompt, seed=0, temperature=1.0, max_tokens=1000, top_k=25, top_p=1.0, unload_model=False):
        
        if seed:
            set_seed(self.hash_seed(seed))

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
        response = re.sub(r'<think>[\s\S]*?</think>', '', response).strip()

        if unload_model: # Check if unload_model is True
            self.unload_model_from_cache(model) # Unload the model if requested
            print(f"DeepseekRun: Model '{model}' unloaded from cache.") # Inform user model is unloaded

        return (response, )

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

    def hash_seed(self, seed):
        import hashlib
        # Convert the seed to a string and then to bytes
        seed_bytes = str(seed).encode('utf-8')
        # Create a SHA-256 hash of the seed bytes
        hash_object = hashlib.sha256(seed_bytes)
        # Convert the hash to an integer
        hashed_seed = int(hash_object.hexdigest(), 16)
        # Ensure the hashed seed is within the acceptable range for set_seed
        return hashed_seed % (2**32)


NODE_CLASS_MAPPINGS = {
    "DeepseekRun": DeepseekRun
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DeepseekRun": "Deepseek Run"
}