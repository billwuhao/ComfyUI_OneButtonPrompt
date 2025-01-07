import random
import os

def process_txt_file(txtfile: str):
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Script directory
    file_path = os.path.join(script_dir, "./txtfiles/", txtfile)

    if not os.path.exists(file_path):
        file_path = os.path.join(script_dir, "./txtfiles/", "example_" + txtfile)

    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    processed_lines = [line.strip() for line in lines if line.strip()]
    return processed_lines

humans = process_txt_file("humans.txt")
others = process_txt_file("others.txt")
human_poses = process_txt_file("human_poses.txt")
styles = process_txt_file("styles.txt")
subject = ["human", "other", "dual_subject"]

def generate_prompt(subject: str, human_pose: bool, style: bool, lora_trigger_or_prefix: str, refresh: bool, seed: int):

    if refresh:
        global humans, others, human_poses, styles
        
        humans = process_txt_file("humans.txt")
        others = process_txt_file("others.txt")
        human_poses = process_txt_file("human_poses.txt")
        styles = process_txt_file("styles.txt")

    if seed > 0:
        random.seed(seed)

    prompt_human = random.choice(humans)
    prompt_other = random.choice(others)
    prompt_human_pose = random.choice(human_poses)
    prompt_style = random.choice(styles)

    if subject == "human":
        prompt_subject = random.choice(humans)
        if human_pose == True:
            prompt_subject = prompt_subject + ", " + prompt_human_pose

    if subject == "other":
        prompt_subject = prompt_other

    if subject == "dual_subject":
        if human_pose == True:
            prompt_subject = prompt_human + ", " + prompt_human_pose + ", " + prompt_other
        else:
            prompt_subject = prompt_human + ", " + prompt_other

    if lora_trigger_or_prefix:
        if lora_trigger_or_prefix.strip():
            lora_trigger_or_prefix = lora_trigger_or_prefix.strip() + ", "

    if style == True:
        prompt = lora_trigger_or_prefix + prompt_subject + ", " + prompt_style
    else:
        prompt = lora_trigger_or_prefix + prompt_subject

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
                "subject": (subject, {
                    "default": "human", "tooltip": "'dual_subject' including both"
                }),
                },
            "optional": {
                "human_pose": ("BOOLEAN", {"default": False, "tooltip": "Effective when selecting 'human'"}),
                "style": ("BOOLEAN", {"default": True}),
                "lora_trigger_or_prefix": ("STRING", {
                    "multiline": False, 
                    "default": "", "tooltip": "Lora trigger words or custom prefix."
                }),
                "refresh": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            },
        }

    def fluxprompt(
            self, 
            subject: str = "human", 
            human_pose: bool = False, 
            style: bool = True,
            lora_trigger_or_prefix: str = "",
            refresh: bool = False,
            seed: int = 0):

        return (generate_prompt(subject, human_pose, style, lora_trigger_or_prefix, refresh, seed),)


NODE_CLASS_MAPPINGS = {
    "OneButtonPromptFlux": OneButtonPromptFlux,
    
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OneButtonPromptFlux": "One Button Prompt Flux",
}