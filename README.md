![One Button Prompt for Flux in ComfyUI](https://github.com/billwuhao/ComfyUI_OneButtonPrompt_Flux/blob/master/images/example.png)

---

# One Button Prompt for Flux in ComfyUI

## Summary

This is a node for generating Flux prompts with one click in ComfyUI.

## Usage

- **If "human" is selected**, it will generate only the human-related prompt.
- **If "other" is selected**, it will generate only the other-related prompt.
- **If "dual_subject" is selected**, it will include both of the above.
- **If "human_pose" is enabled**, it will generate a human pose prompt.
- **If "style" is enabled**, it will generate a style prompt.
- **Lora trigger words/prefixes** are optional and can be customized.
- If you have modified the custom file during the generation images, please enable **refresh** to update it, without having to restart the software
- **If the seed is fixed**, it will generate the same fixed prompt.

![alt text](https://github.com/billwuhao/ComfyUI_OneButtonPrompt_Flux/blob/master/images/image-1.png)

**Example Prompt**:  

(MW:1.2), a beautiful girl, fall asleep, A beautiful lovely alpaca, naive art

- `(MW:1.2)` is the **Lora trigger word**.
- `a beautiful girl` is the **human**.
- `fall asleep` is the **human_pose**.
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

ComfyUI-Fluxpromptenhancer: https://github.com/marduk191/ComfyUI-Fluxpromptenhancer

---

## 概述

这是一个在 comfyui 中一键生成 flux 提示的节点.

## 用法

- 主体选择 human，则只生成 human 提示
- 选择 other，则只生成 other 提示
- 选择 dual_subject，则包含上述两者
- 如果开启 human_pose，将生成人物姿态提示
- 如果开启 style，将生成 style 提示
- lora 触发词/前缀 是可选的，可自定义输入
- 如果你在运行过程中修改了自定义文件, 请开启 refresh 刷新一下, 而不用重启软件.
- 如果固定种子，将只生成一次固定的提示

![alt text](https://github.com/billwuhao/ComfyUI_OneButtonPrompt_Flux/blob/master/images/image-1.png)

(MW:1.2), a beautiful girl, fall asleep, A beautiful lovely alpaca, naive art

- `(MW:1.2)`, 是 lora 触发词
- `a beautiful girl`, 是 human
- `fall asleep`, 是 human_pose
- `A beautiful lovely alpaca`, 是 other
- `naive art`, 是 style

## 自定义主体，姿态和风格

1. 在文件夹 `ComfyUI\custom_nodes\ComfyUI_OneButtonPrompt_Flux\txtfiles` 下面新建 4 个文件

   ![alt text](https://github.com/billwuhao/ComfyUI_OneButtonPrompt_Flux/blob/master/images/image.png)

2. 分别输入相应的内容, 即可生成自定义的主体, 姿态, 风格

3. `prompt_words.xlsx` 文件里是我很早的时候, 玩 disco diffusion 和 sd1.5 时整理的提示, 你可以参考.

注意: 一定要新建文件, 然后随意修改, 并不会影响节点使用和更新.

## 自动生图

如果你想挂着电脑自动生图, 选择 `Queue (On Change)` 点击开始即可. 工作流在 `workflow_example` 里.

![alt text](https://github.com/billwuhao/ComfyUI_OneButtonPrompt_Flux/blob/master/images/image-2.png)

## 结合 Prompt Enhancer 一起使用

ComfyUI-Fluxpromptenhancer: https://github.com/marduk191/ComfyUI-Fluxpromptenhancer

# Thank you

Thank you to the amazing [OneButtonPrompt](https://github.com/AIrjen/OneButtonPrompt) for inspiration.