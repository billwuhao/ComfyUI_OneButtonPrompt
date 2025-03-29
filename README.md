[中文](README-CN.md) | [English](README.md)

![One Button Prompt for Flux in ComfyUI](https://github.com/billwuhao/ComfyUI_OneButtonPrompt_Flux/blob/master/images/example.png)
![](https://github.com/billwuhao/ComfyUI_OneButtonPrompt_Flux/blob/master/images/2025-02-19_20-00-01.png.png)

---

# A node in comfyui for one-click assisted prompt generation (for image and video generation, etc.).

(https://github.com/billwuhao/ComfyUI_OneButtonPrompt_Flux/blob/master/images/2025-03-30_03-22-41.png)

## Updates

[2025-03-30]⚒️: Completely refactored, released v1.0.0. I apologize for any inconvenience caused. If you have any suggestions or comments, please feel free to submit them in RP. Large language models are too powerful, so I decided to refactor this project. Only starting prompts are provided. Combined with large language models, they can be used for polishing, optimization, and generating various prompts (currently only suitable for image and video generation).

The starting prompt folder is located in the `ComfyUI_OneButtonPrompt_Flux/files` directory, categorized as `images` (.txt files), `prompts` (.md files), and `prompts-images` (.json files). You can freely customize files within these folders, but the content format must be consistent. Custom file names must start with `nsfw` or `custom` to avoid being affected by updates.

(https://github.com/billwuhao/ComfyUI_OneButtonPrompt_Flux/blob/master/images/2025-03-30_03-52-46.png)

`prompts-images` supports searching prompts by specific words. For example, entering `cat` or `dog` will return prompts containing `cat` or `dog`.

## Acknowledgments

(OneButtonPrompt)(https://github.com/AIrjen/OneButtonPrompt)