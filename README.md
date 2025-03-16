[中文](README-CN.md) | [English](README.md)

![One Button Prompt for Flux in ComfyUI](https://github.com/billwuhao/ComfyUI_OneButtonPrompt_Flux/blob/master/images/example.png)

---

# A node for one click generation of prompts (for image and video generation, etc.) in Comfyui

## Summary

This is a node for generating Flux prompts with one click in ComfyUI.

## 📣 Updates

[2025-03-01]⚒️: 

- Supports Qwen2.5 language and visual models, can generate prompts, and can describe images and videos.
  1. Generate prompts:
  ![](https://github.com/billwuhao/ComfyUI_OneButtonPrompt_Flux/blob/master/images/3bflux.png)
  2. Two images are used to generate the video prompt:
  ![](https://github.com/billwuhao/ComfyUI_OneButtonPrompt_Flux/blob/master/images/nextimage.png)
  3. Describe video (high memory usage, please load fewer frames):
  ![](https://github.com/billwuhao/ComfyUI_OneButtonPrompt_Flux/blob/master/images/video3b.png)

- Now it is possible to load `onlyprompt` from JSON files on C site without the need for internet connection.
  ![](https://github.com/billwuhao/ComfyUI_OneButtonPrompt_Flux/blob/master/images/onlyprompt.png)

Download the following 3 models to `models/LLM`:
- [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
- [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)
- [Qwen2.5-3B-Instruct-Flux](https://huggingface.co/mrkrak3n/Qwen2.5-3B-Instruct-Flux)

[2025-02-20]⚒️: Support [C](https://civitai.com/images) Station images and prompts.

![](https://github.com/billwuhao/ComfyUI_OneButtonPrompt_Flux/blob/master/images/2025-02-23_00-40-23.png)

Default use [Civitai](https://civitai.com/images). If you need to use images from other websites, please modify the file `\ComfyUI_OneButtonPrompt_Flux\txtfiles\civit_sfw.json` yourself. I will periodically update the `civit_sfw.json` file

If you need nsfw images, please create a new `civit_nsfw.json` file in the `ComfyUI_OneButtonPrompt_Flux\txtfiles\` folder, which should be in the same format as the content of the `civit_sfw.json` file.

Collaborate with Deepseek r1 to optimize and enhance prompts.

![](https://github.com/billwuhao/ComfyUI_OneButtonPrompt_Flux/blob/master/images/2025-02-23_01-14-08.png)

Reverse inference prompts.

![](https://github.com/billwuhao/ComfyUI_OneButtonPrompt_Flux/blob/master/images/2025-02-23_01-37-50.png)

[2025-02-19] ⚒️: Support local DeepSeek R1.

![](https://github.com/billwuhao/ComfyUI_OneButtonPrompt_Flux/blob/master/images/2025-02-19_10-32-16.png)

[DeepScaleR-1.5B-Preview](https://hf-mirror.com/agentica-org/DeepScaleR-1.5B-Preview) Exceeding the performance of OpenAI's O1 Preview. Manually download the model and all configuration files to `ComfyUI\models\LLM\DeepScaleR-1.5B-Preview` to use

-Custom theme usage:
![](https://github.com/billwuhao/ComfyUI_OneButtonPrompt_Flux/blob/master/images/2025-02-19_19-22-49.png)

-Advanced automatic prompt usage:
![](https://github.com/billwuhao/ComfyUI_OneButtonPrompt_Flux/blob/master/images/2025-02-19_19-19-26.png)

This way, `{ a beautiful girl | a handsome man | a sexy woman | a naughty child | a future interstellar animal | a transportation  vehicle| a space vehicle | A fantastic building | a landscape of lake and mountain | a Chinese landscape| a sea landscape| a city landscape}` will set the theme and automatically randomly select a theme to generate prompt.

-Leave more usage options to you to develop on your own.

-Generate effect:

![](https://github.com/billwuhao/ComfyUI_OneButtonPrompt_Flux/blob/master/images/2025-02-19_20-00-01.png)

**Tip**: To solve the problem of OOM, you can replace Flux with sdxl or SD15 and adjust the DeepSeek R1 prompt. Think on your own.

[2025-01-21]⚒️: 

- `txt` file support `//` comments: The line for comments will be ignored.

- Add more prompts (human and poses):

![](https://github.com/billwuhao/ComfyUI_OneButtonPrompt_Flux/blob/master/images/2025-01-25_22-47-23.png)
![](https://github.com/billwuhao/ComfyUI_OneButtonPrompt_Flux/blob/master/images/2025-01-25_22-49-40.png)
![](https://github.com/billwuhao/ComfyUI_OneButtonPrompt_Flux/blob/master/images/2025-01-25_22-58-00.png)

[2025-01-10]⚒️: 

- human_pose → pose: Any subject can set pose. Please promptly change the file name `human_pose` to `pose`.

- Add 4 `add_` files to record the good prompt words I have found. You can also share your prompt words and submit an PR.

- Add `None` to `subject` to make it optional, so that custom subject can be fixed in `lora_trigger_or_prefix` to generate images under the same subject and win prizes.

- To add a testing mode, please put the prompts you need to test into the file `test.txt`(The first time you need to create a new one), and then turn on the testing mode to test each prompt in sequence. **Note**: During testing mode, all options except for the custom `lora_trigger_or_prefix` are invalid. f you are running the test automatically, after all prompts are completed, switch to a random generated image.

  ![](https://github.com/billwuhao/ComfyUI_OneButtonPrompt_Flux/blob/master/images/2025-01-10_12-07-54.png)

## Usage

- **If "human" is selected**, it will generate only the human-related prompt.
- **If "other" is selected**, it will generate only the other-related prompt.
- **If "dual_subject" is selected**, it will include both of the above.
- **If "pose" is enabled**, it will generate a subject pose prompt.
- **If "style" is enabled**, it will generate a style prompt.
- **lora_trigger_or_prefix** are optional and can be customized.
- If you have modified the custom file during the generation images, please enable **refresh** to update it, without having to restart the software
- **If the seed is fixed**, it will generate the same fixed prompt.

![alt text](https://github.com/billwuhao/ComfyUI_OneButtonPrompt_Flux/blob/master/images/image-1.png)

**Example Prompt**:  

(MW:1.2), a beautiful girl, fall asleep, A beautiful lovely alpaca, naive art

- `(MW:1.2)` is the **Lora trigger word**.
- `a beautiful girl` is the **human**.
- `fall asleep` is the **pose**.
- `A beautiful lovely alpaca` is the **other**.
- `naive art` is the **style**.

## Custom Subjects, Poses, and Styles

1. Create four files under the folder `ComfyUI\custom_nodes\ComfyUI_OneButtonPrompt_Flux\txtfiles`.

   ![alt text](https://github.com/billwuhao/ComfyUI_OneButtonPrompt_Flux/blob/master/images/image.png)

2. Input the corresponding content into each file to generate your custom subject, pose, and style.

3. The `prompt_words.xlsx` file contains prompts I gathered a long time ago when experimenting with **Disco Diffusion** and **SD 1.5**. You can refer to it for ideas.

**Note**: Be sure to create new files and modify them freely; this will not affect the node's functionality or future updates.

## Automatic Image Generation

If you want to automatically generate images while your computer is running, select `Queue (On Change)` and click Start. Workflow in `workflow_example`.

![alt text](https://github.com/billwuhao/ComfyUI_OneButtonPrompt_Flux/blob/master/images/image-2.png)

## Using with Prompt Enhancer

The enhancer of prompt has a poor effect in some cases, completely destroying the style and artistic conception of the prompt. It is recommended to enable it during the lottery

ComfyUI-Fluxpromptenhancer: https://github.com/marduk191/ComfyUI-Fluxpromptenhancer

## Thank you

Thank you to the amazing [OneButtonPrompt](https://github.com/AIrjen/OneButtonPrompt) for inspiration.