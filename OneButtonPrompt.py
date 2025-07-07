import random
import os
import numpy as np
import json
import re
from PIL import Image, ImageSequence, ImageOps
import torch
import requests
from io import BytesIO


node_dir = os.path.dirname(os.path.abspath(__file__))
prompts_file_path = os.path.join(node_dir, "files", "prompts")
imgs_file_path = os.path.join(node_dir, "files", "images")
imgs_prompts_path = os.path.join(node_dir, "files", "prompts-images")


def get_prompts_from_txtfile(txtfile: str):
    with open(txtfile, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        processed_lines = [line.split("//")[0].strip() for line in lines if line.strip() and not line.lstrip().startswith("//")]
        if processed_lines:
            return processed_lines
        else:
            return [""]
        

def get_imageurls_from_mdfile(mdfile: str):
    with open(mdfile, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        processed_lines = [line.strip().strip("![]()") for line in lines if line.lstrip().startswith("![]")]
        if processed_lines:
            return processed_lines
        else:
            return []
        

def get_imageurls_prompts_from_jsonfile(jsonfile: str):
    with open(jsonfile, 'r', encoding='utf-8') as file:
        data = json.load(file)
        return data


def find_exact_word(text, word):
    return bool(re.search(rf"\b{word}\b", text))


def search_word_from_prompts(data: dict, word: str):
    processed_data = {}
    keys = list(data.keys())
    string_values = [data[key][1] for key in keys] 

    arr = np.array(string_values)
    vectorized_find = np.vectorize(find_exact_word)
    mask = vectorized_find(arr, word)
    processed_data = {keys[i]: data[keys[i]] for i in range(len(keys)) if mask[i]}
    
    return processed_data


def find_files_by_type(folder_path, file_extension):
    matching_files = []
    for filename in os.listdir(folder_path):
        if filename.endswith(file_extension):
            matching_files.append(filename)

    return matching_files



def get_image_data_from_url(url, load_time, proxies=None):
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
            image_content = response.content
            import time
            time.sleep(load_time)
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


class LoadPrompt:
    txt_list = find_files_by_type(prompts_file_path, ".txt")
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "txt_1": (cls.txt_list,),
                "txt_2": (cls.txt_list + ["None"],),
                "txt_3": (cls.txt_list + ["None"],),
                "txt_4": (cls.txt_list + ["None"],),
                "txt_5": (cls.txt_list + ["None"],),
                },
            "optional": {
                # "refresh": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            },
        }


    CATEGORY = "🎤MW/MW-OneButtonPrompt"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "loadprompt"
    
    def loadprompt(
            self, 
            txt_1: str, 
            txt_2: str, 
            txt_3: str, 
            txt_4: str, 
            txt_5: str,
            # refresh: bool = False,
            seed: int = 0
            ):
        
        if seed != 0:
            random.seed(seed)
        
        PROMPTS = []
        prompt1 = get_prompts_from_txtfile(os.path.join(prompts_file_path, txt_1))
        PROMPTS.append(prompt1)
        if txt_2 != "None":
            prompt2 = get_prompts_from_txtfile(os.path.join(prompts_file_path, txt_2))
            PROMPTS.append(prompt2)
        if txt_3 != "None":
            prompt3 = get_prompts_from_txtfile(os.path.join(prompts_file_path, txt_3))
            PROMPTS.append(prompt3)
        if txt_4 != "None":
            prompt4 = get_prompts_from_txtfile(os.path.join(prompts_file_path, txt_4))
            PROMPTS.append(prompt4)
        if txt_5 != "None":
            prompt5 = get_prompts_from_txtfile(os.path.join(prompts_file_path, txt_5))
            PROMPTS.append(prompt5)

        np_data = np.array(PROMPTS, dtype=object)
        lengths = np.array([len(sublist) for sublist in np_data])
        random_indices = np.random.randint(0, lengths)
        result = [sublist[index] for sublist, index in zip(np_data, random_indices)]

        return ("; ".join(result),)


class LoadImageFromURL:
    md_list = find_files_by_type(imgs_file_path, ".md")
    def __init__(self):
        self.images = set()
        self.md_1 = None
        self.md_2 = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "md_1": (cls.md_list,),
                "md_2": (cls.md_list + ["None"],),
                },
            "optional": {
                "in_order": ("BOOLEAN", {"default": False}),
                "load_time": ("FLOAT", {"default": 2, "min": 0.0, "max": 10, "step": 0.5}),
                # "refresh": ("BOOLEAN", {"default": False}),
                "proxy": ("STRING", {"default": "http://127.0.0.1:None"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            },
        }

    CATEGORY = "🎤MW/MW-OneButtonPrompt"
    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("Image", "Imgurl",)
    FUNCTION = "loadimage"
    
    def loadimage(
            self, 
            md_1: str, 
            md_2: str, 
            in_order: bool = False,
            load_time: float = 2,
            # refresh: bool = False,
            proxy: str = "http://127.0.0.1:None",
            seed: int = 0
            ):
        
        if seed != 0:
            random.seed(seed)

        if self.md_1 is None:
            self.md_1 = md_1

        if self.md_2 is None:
            self.md_2 = md_2

        if self.images == set() or self.md_1 != md_1 or self.md_2 != md_2:
            self.md_1 = md_1
            self.md_2 = md_2
            self.images = set()
            imgurl1 = get_imageurls_from_mdfile(os.path.join(imgs_file_path, md_1))
            imgurl2 = get_imageurls_from_mdfile(os.path.join(imgs_file_path, md_2)) if md_2 != "None" else []
            self.images = set(imgurl1 + imgurl2)

        if len(self.images) == 0:
            raise ValueError("No image URL found.")

        if in_order:
            imgurl = self.images.pop()
            print(f"---------\n{len(self.images)}") 
        else:
            imgurl = random.choice(list(self.images))

        if proxy.strip() in ["http://127.0.0.1:None", ""]:
            proxy = None
        else:
            proxies = {
                "http": proxy,
                "https": proxy,
            }
        img = get_image_data_from_url(imgurl, load_time, proxies=proxies)
        if img is None:
            raise ValueError("Failed to load image from URL. Please check the URL or proxy settings.")
        img = img.convert("RGBA")
        img, _ = pil2tensor(img)

        return (img, imgurl)


class LoadImageAndPromptFromURL:
    json_list = find_files_by_type(imgs_prompts_path, ".json")
    def __init__(self):
        self.images_prompts = {}
        self.json_1 = None
        self.json_2 = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_1": (cls.json_list,),
                "json_2": (cls.json_list + ["None"],),
                },
            "optional": {
                "in_order": ("BOOLEAN", {"default": False}),
                "load_time": ("FLOAT", {"default": 2, "min": 0.0, "max": 10, "step": 0.5}),
                # "refresh": ("BOOLEAN", {"default": False}),
                "proxy": ("STRING", {"default": "http://127.0.0.1:None"}),
                "on_search": ("BOOLEAN", {"default": False}),
                "search_for": ("STRING", {"default": "cat"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            },
        }


    CATEGORY = "🎤MW/MW-OneButtonPrompt"
    RETURN_TYPES = ("IMAGE", "STRING", "STRING",)
    RETURN_NAMES = ("Image", "Prompt", "Imgurl",)
    FUNCTION = "loadimageprompt"
    
    def loadimageprompt(
            self, 
            json_1: str, 
            json_2: str, 
            in_order: bool = False,
            load_time: float = 2,
            # refresh: bool = False,
            proxy: str = "http://127.0.0.1:None",
            on_search: bool = False,
            search_for: str = "cat",
            seed: int = 0
            ):
        
        if seed != 0:
            random.seed(seed)
        
        if self.json_1 is None:
            self.json_1 = json_1

        if self.json_2 is None:
            self.json_2 = json_2

        if self.images_prompts == {} or self.json_1 != json_1 or self.json_2 != json_2:
            self.json_1 = json_1
            self.json_2 = json_2
            imgs_prompts1 = get_imageurls_prompts_from_jsonfile(os.path.join(imgs_prompts_path, json_1))
            imgs_prompts2 = get_imageurls_prompts_from_jsonfile(os.path.join(imgs_prompts_path, json_2)) if json_2 != "None" else {}
            self.images_prompts = {}
            self.images_prompts.update(imgs_prompts1)
            self.images_prompts.update(imgs_prompts2)

        if len(self.images_prompts) == 0:
            raise ValueError("No image URL found.")

        if on_search:
            word = search_for.strip()
            self.images_prompts = search_word_from_prompts(self.images_prompts, word)
            if len(self.images_prompts) == 0:
                raise ValueError("No prompt found with the search word.")

        if in_order:
            imgurl, info = self.images_prompts.popitem()
            prompt = info[1]
            print(f"---------\n{len(self.images_prompts)}")
        else:
            imgurl = random.choice(list(self.images_prompts.keys()))
            prompt = self.images_prompts[imgurl][1]

        if proxy.strip() in ["http://127.0.0.1:None", ""]:
            proxy = None
        else:
            proxies = {
                "http": proxy,
                "https": proxy,
            }
        img = get_image_data_from_url(imgurl, load_time, proxies=proxies)
        if img is None:
            raise ValueError("Failed to load image from URL. Please check the URL or proxy settings.")
        img = img.convert("RGBA")
        img, _ = pil2tensor(img)

        return (img, prompt, imgurl)


class StringEditNodeOBP:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}), 
             },
            "optional": {
                "paste_edit_text": ("STRING", {"multiline": True, "default": "",}),
                "split_tag": ("STRING", {"default": "",}),
                "form_right": ("BOOLEAN", {"default": False}),
             },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("text", "split_1", "split_2", "edited_text", "merged_text")
    FUNCTION = "edit_text"
    CATEGORY = "🎤MW/MW-OneButtonPrompt" 
    OUTPUT_NODE = False

    def edit_text(self, text, split_tag="", paste_edit_text="", form_right=False):
        str_1, str_2, str_3, str_4, str_5 = text, "", "", paste_edit_text, ""
        str_5 = text + paste_edit_text
        
        if split_tag != "" and text.find(split_tag) > 0:
            if form_right:
                parts = text.rsplit(split_tag, 1)
                str_2, str_3 = parts[0].strip(), parts[1].strip()
            else:
                parts = text.split(split_tag, 1)
                str_2, str_3 = parts[0].strip(), parts[1].strip()

        return (str_1, str_2, str_3, str_4, str_5)      

NODE_CLASS_MAPPINGS = {
    "LoadPrompt": LoadPrompt,
    "LoadImageFromURL": LoadImageFromURL,
    "LoadImageAndPromptFromURL": LoadImageAndPromptFromURL,
    "StringEditNodeOBP": StringEditNodeOBP,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadPrompt": "Load Prompt",
    "LoadImageFromURL": "Load Image From URL",
    "LoadImageAndPromptFromURL": "Load Image And Prompt From URL",
    "StringEditNodeOBP": "String Edit"
}