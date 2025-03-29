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


PROMPTS = None
IMAGES =  None
IMAGES_PROMPTS = None


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
                "refresh": ("BOOLEAN", {"default": False}),
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
            refresh: bool = False,
            seed: int = 0
            ):
        
        if seed != 0:
            random.seed(seed)
        
        global PROMPTS
        if refresh:
            PROMPTS = None
        
        if PROMPTS is None:
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
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "md_1": (cls.md_list,),
                "md_2": (cls.md_list + ["None"],),
                },
            "optional": {
                "load_time": ("FLOAT", {"default": 2, "min": 0.0, "max": 10, "step": 0.5}),
                "refresh": ("BOOLEAN", {"default": False}),
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
            load_time: float = 2,
            refresh: bool = False,
            proxy: str = "http://127.0.0.1:None",
            seed: int = 0
            ):
        
        if seed != 0:
            random.seed(seed)
        
        global IMAGES
        if refresh:
            IMAGES = None
        
        if IMAGES is None:
            IMAGES = []
            imgurl1 = get_imageurls_from_mdfile(os.path.join(imgs_file_path, md_1))
            IMAGES.extend(imgurl1)
            if md_2 != "None":
                imgurl2 = get_imageurls_from_mdfile(os.path.join(imgs_file_path, md_2))
                IMAGES.extend(imgurl2)
                if len(IMAGES) == 0:
                    raise ValueError("No image URL found.")
        
        imgurl = random.choice(IMAGES)
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
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_1": (cls.json_list,),
                "json_2": (cls.json_list + ["None"],),
                },
            "optional": {
                "load_time": ("FLOAT", {"default": 2, "min": 0.0, "max": 10, "step": 0.5}),
                "refresh": ("BOOLEAN", {"default": False}),
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
            load_time: float = 2,
            refresh: bool = False,
            proxy: str = "http://127.0.0.1:None",
            on_search: bool = False,
            search_for: str = "cat",
            seed: int = 0
            ):
        
        if seed != 0:
            random.seed(seed)
        
        global IMAGES_PROMPTS
        if refresh:
            IMAGES_PROMPTS = None
        
        if IMAGES_PROMPTS is None:
            IMAGES_PROMPTS = {}
            imgs_prompts1 = get_imageurls_prompts_from_jsonfile(os.path.join(imgs_prompts_path, json_1))
            IMAGES_PROMPTS.update(imgs_prompts1)
            if json_2 != "None":
                imgs_prompts2 = get_imageurls_prompts_from_jsonfile(os.path.join(imgs_prompts_path, json_2))
                IMAGES_PROMPTS.update(imgs_prompts2)
            if len(IMAGES_PROMPTS) == 0:
                raise ValueError("There is no content in the JSON file.")
            
        if on_search:
            word = search_for.strip()
            search_imgs_prompts = search_word_from_prompts(IMAGES_PROMPTS, word)
            if len(search_imgs_prompts) == 0:
                raise ValueError("No prompt found with the search word.")
            else:
                imgurl = random.choice(list(search_imgs_prompts.keys()))
                prompt = search_imgs_prompts[imgurl][1]
        else:
            imgurl = random.choice(list(IMAGES_PROMPTS.keys()))
            prompt = IMAGES_PROMPTS[imgurl][1]

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
    


NODE_CLASS_MAPPINGS = {
    "LoadPrompt": LoadPrompt,
    "LoadImageFromURL": LoadImageFromURL,
    "LoadImageAndPromptFromURL": LoadImageAndPromptFromURL,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadPrompt": "Load Prompt",
    "LoadImageFromURL": "Load Image From URL",
    "LoadImageAndPromptFromURL": "Load Image And Prompt From URL",
}