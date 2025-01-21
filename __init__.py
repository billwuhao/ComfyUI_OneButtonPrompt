import random
import os

def process_txt_file(txtfile: str):
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Script directory
    file_path = os.path.join(script_dir, "./txtfiles/", txtfile)

    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            processed_lines = [line.strip() for line in lines if line.strip() and not line.lstrip().startswith("//")]
            if processed_lines:
                return processed_lines
            else:
                file_path = os.path.join(script_dir, "./txtfiles/", "example_" + txtfile)
    else:
        file_path = os.path.join(script_dir, "./txtfiles/", "example_" + txtfile)

    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        processed_lines = [line.strip() for line in lines if line.strip() and not line.lstrip().startswith("//")]
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
    
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OneButtonPromptFlux": "One Button Prompt Flux",
}