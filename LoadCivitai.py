from PIL import Image, ImageSequence, ImageOps
import torch
import requests
from io import BytesIO
import os
import re
import numpy as np
import json
import random
from transformers import set_seed

def get_image_data_from_url(url, proxies=None):
    """
    Checks if a URL is likely an image and returns the image data if it is.

    Args:
        url (str): The URL to check.

    Returns:
        bytes or None:
        - Image data (bytes) if the URL is likely an image.
        - None if the URL is not likely an image or if there was an error.
    """
    try:
        response = requests.get(url, stream=True, allow_redirects=True, timeout=10, proxies=proxies)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)

        content_type = response.headers.get('Content-Type', '').lower()

        if content_type.startswith('image/'):
            image_content = response.content # 显式读取 response.content 到内存
            if image_content is None:
                return None
            return Image.open(BytesIO(image_content))  # Return the image data as bytes
        else:
            return None  # Not an image content type

    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL: {url}. Error: {e}")
        return None  # Error during request
    
def pil2tensor(img):
    output_images = []
    output_masks = []
    for i in ImageSequence.Iterator(img):
        i = ImageOps.exif_transpose(i)
        if i.mode == 'I':
            i = i.point(lambda i: i * (1 / 255))
        image = i.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        if 'A' in i.getbands():
            mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
            mask = 1. - torch.from_numpy(mask)
        else:
            mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
        output_images.append(image)
        output_masks.append(mask.unsqueeze(0))

    if len(output_images) > 1:
        output_image = torch.cat(output_images, dim=0)
        output_mask = torch.cat(output_masks, dim=0)
    else:
        output_image = output_images[0]
        output_mask = output_masks[0]

    return (output_image, output_mask)

def process_prompt(prompt):
    result = re.sub(r'[sS]core_\w+\s*,?\s*|<.+?>\s*,?\s*', '', prompt, flags=re.DOTALL)
    # 清理多余的空格和换行
    result = re.sub(r'\s+', ' ', result).strip()
    return result

def get_random_imginfo(data: dict, img_type: str, proxies=None):
    """
    Selects and returns a random key-value pair from a dictionary.

    Args:
        data (dict): The input dictionary.
        img_type (str): The type of image to return.
    Returns:
        tuple or None: A tuple containing the (key, value) of a randomly
                       selected item from the dictionary. Returns None if
                       the dictionary is empty.
    """
    if img_type == "Img+Prompt" or img_type == "OnlyPrompt":
        data = data["Prompt"]
    else:
        data = {**data["Prompt"], **data["NoPrompt"]}

    keys = list(data)
    if img_type == "OnlyPrompt":
        prompt = process_prompt(data[random.choice(keys)][1])
        return None, prompt
    else:
        for i in range(6):
            random_url = random.choice(keys)
            imageinfo = get_image_data_from_url(random_url, proxies=proxies)
            prompt = process_prompt(data[random_url][1])
            if imageinfo:
                return imageinfo, prompt
        else:
            raise ValueError("Failed to find a valid image URL after 6 attempts. Or the proxy settings are incorrect.")


class LoadImageInfoFromCivitai:
    node_dir = os.path.dirname(os.path.abspath(__file__))
    jsonfile_path = os.path.join(node_dir, "txtfiles")
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "output_type": (["Img", "Img+Prompt", "OnlyPrompt"], {"default": "OnlyPrompt"}),
                "nsfw": ("BOOLEAN", {
                    "default": False, 
                    "tooltip": "When True, need `civit_nsfw.json` in the `txtfiles` folder, otherwise it is invalid"}),
                "proxy": ("STRING", {
                    "multiline": False,
                    "default": "http://127.0.0.1:None", 
                    "tooltip": "When load Img, if unable to access Civitai site, proxy needs to be filled in"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    CATEGORY = "MW-OneButtonPrompt"
    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("Image", "Mask", "Prompt")
    FUNCTION = "load"
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

    def load(self, output_type, nsfw, proxy, seed=0):
        if seed:
            set_seed(self.hash_seed(seed))

        proxy = None if proxy == "http://127.0.0.1:None" else proxy
        data = self.load_data_form_json(nsfw)

        img, prompt = get_random_imginfo(data, output_type, proxies={"https": proxy,"http": proxy,})

        if img is None: 
            img = Image.new('RGB', (512, 512), color=(0, 0, 0))
            # return (None, None, prompt) 
        
        img_out, mask_out = pil2tensor(img)

        return (img_out, mask_out, prompt)
    
        
    def load_data_form_json(self,nsfw):
        if nsfw:
            file_path = self.jsonfile_path + "/civit_nsfw.json"
            if not os.path.exists(file_path):
                file_path = self.jsonfile_path + "/civit_sfw.json"
        else:
            file_path = self.jsonfile_path + "/civit_sfw.json"
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data
        except Exception as e:
            print(f"Error loading JSON file: {file_path}. Error: {e}")
            raise
