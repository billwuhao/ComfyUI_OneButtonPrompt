import random
import os
import re
from collections.abc import Callable
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
import folder_paths
from comfy import model_management, model_patcher


def create_path_dict(paths: list[str], predicate: Callable[[Path], bool] = lambda _: True) -> dict[str, str]:
    """
    Creates a flat dictionary of the contents of all given paths: ``{name: absolute_path}``.

    Non-recursive.  Optionally takes a predicate to filter items.  Duplicate names overwrite (the last one wins).

    Args:
        paths (list[str]):
            The paths to search for items.
        predicate (Callable[[Path], bool]): 
            (Optional) If provided, each path is tested against this filter.
            Returns ``True`` to include a path.

            Default: Include everything
    """

    flattened_paths = [item for path in paths for item in Path(path).iterdir() if predicate(item)]

    return {item.name: str(item.absolute()) for item in flattened_paths}


class DeepseekRun:
    @classmethod
    def INPUT_TYPES(s):
        
        all_llm_paths = folder_paths.get_folder_paths("LLM")
        s.model_paths = create_path_dict(all_llm_paths, lambda x: x.is_dir())

        return {
            "required": {
                "model": ([*s.model_paths], {"tooltip": "models are expected to be in Comfyui/models/LLM folder"}),
                "user_prompt": ("STRING", {"default": "", "multiline": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "max_tokens": ("INT", {"default": 1000, "min": 0, "max": 0xffffffffffffffff}),
                "temperature": ("FLOAT", {"default": 1, "min": 0, "max": 2}),
                "top_k": ("INT", {"default": 20, "min": 0, "max": 101}),
                "top_p": ("FLOAT", {"default": 1, "min": 0, "max": 1}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("STRING",)
    FUNCTION = "dsgen"
    CATEGORY = "MW-OneButtonPrompt"

    _model_cache = {}

    def load_model(self, model):
        model_path = DeepseekRun.model_paths.get(model) 
        
        if model in self._model_cache:
            return self._model_cache[model]
        
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
        
        self._model_cache[model] = {
            "model": dsmodel,
            "tokenizer": tokenizer,
            "device": device
        }

        return {
            "model": dsmodel,
            "tokenizer": tokenizer,
            "device": device
        }
                
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

    def dsgen(self, model, user_prompt, seed=0, temperature=1.0, max_tokens=1000, top_k=25, top_p=1.0, **kwargs):
        
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
            {"role": "user", "content": user_prompt.format(**kwargs)},
        ]

            # 单次加载模型
        model_dict = self.load_model(model)
        
        device = model_dict["device"] # 获取设备信息
        dsmodel = model_dict["model"]
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
        return (response, )

# ------------------------------------

def process_txt_file(txtfile: str):
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Script directory
    file_path = os.path.join(script_dir, "./txtfiles/", txtfile)

    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            processed_lines = [line.split("//")[0].strip() for line in lines if line.strip() and not line.lstrip().startswith("//")]
            if processed_lines:
                return processed_lines
            else:
                file_path = os.path.join(script_dir, "./txtfiles/", "example_" + txtfile)
    else:
        file_path = os.path.join(script_dir, "./txtfiles/", "example_" + txtfile)

    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        processed_lines = [line.split("//")[0].strip() for line in lines if line.strip() and not line.lstrip().startswith("//")]
        return processed_lines

humans = process_txt_file("humans.txt")
others = process_txt_file("others.txt")
poses = process_txt_file("poses.txt")
styles = process_txt_file("styles.txt")
test_prompts = process_txt_file("test.txt")
subject = ["human", "other", "dual_subject", "None"]

def generate_prompt(subject: str, pose: bool, style: bool, lora_trigger_or_prefix: str, refresh: bool, test: bool, seed: int):

    if refresh:
        global humans, others, poses, styles
        
        humans = process_txt_file("humans.txt")
        others = process_txt_file("others.txt")
        poses = process_txt_file("poses.txt")
        styles = process_txt_file("styles.txt")

    if seed > 0:
        random.seed(seed)

    prompt_human = random.choice(humans)
    prompt_other = random.choice(others)
    prompt_pose = random.choice(poses)
    prompt_style = random.choice(styles)

    if subject == "human":
        prompt_subject = prompt_human

    if subject == "other":
        prompt_subject = prompt_other

    if subject == "dual_subject":
        prompt_subject = prompt_human + ", " + prompt_other

    if subject == "None":
        prompt_subject = ""

    if pose == True:
        if prompt_subject:
            prompt_subject = prompt_subject + ", " + prompt_pose
        else:
            prompt_subject = prompt_pose

    if lora_trigger_or_prefix:
        if lora_trigger_or_prefix.strip():
            lora_trigger_or_prefix = lora_trigger_or_prefix.strip() + ", "

    if style == True:
        prompt = lora_trigger_or_prefix + prompt_subject + ", " + prompt_style
    else:
        prompt = lora_trigger_or_prefix + prompt_subject
    
    if test:
        if not test_prompts:
            prompt = prompt
        else:
            prompt = lora_trigger_or_prefix + test_prompts.pop(0)

    return prompt

class OneButtonPromptFlux:

    CATEGORY = "MW-OneButtonPrompt"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "fluxprompt"
    
    @classmethod
    def INPUT_TYPES(s):
               
        return {
            "required": {
                "refresh": ("BOOLEAN", {"default": False}),
                },
            "optional": {
                "subject": (subject, {
                    "default": "human", "tooltip": "'dual_subject' including both. 'None' will be no subject."
                }),
                "pose": ("BOOLEAN", {"default": False, "tooltip": "The pose of any subject."}),
                "style": ("BOOLEAN", {"default": False}),
                "lora_trigger_or_prefix": ("STRING", {
                    "multiline": False, 
                    "default": "", "tooltip": "Lora trigger words or custom prefix."
                }),
                "test": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            },
        }

    def fluxprompt(
            self, 
            subject: str = "human", 
            pose: bool = False, 
            style: bool = False,
            lora_trigger_or_prefix: str = "",
            refresh: bool = False,
            test: bool = False,
            seed: int = 0):

        return (generate_prompt(subject, pose, style, lora_trigger_or_prefix, refresh, test, seed),)

NODE_CLASS_MAPPINGS = {
    "OneButtonPromptFlux": OneButtonPromptFlux,
    "DeepseekRun": DeepseekRun
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OneButtonPromptFlux": "One Button Prompt Flux",
    "DeepseekRun": "Deepseek Run"
}