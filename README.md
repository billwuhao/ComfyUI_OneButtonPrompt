![One Button Prompt for Flux in ComfyUI](https://github.com/billwuhao/ComfyUI_OneButtonPrompt_Flux/blob/master/images/example.png)

---

# One Button Prompt for Flux in ComfyUI

## Summary

This is a node for generating Flux prompts with one click in ComfyUI.

## 📣 Updates

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

---

## 概述

这是一个在 comfyui 中一键生成 flux 提示的节点.

## 📣 更新

[2025-01-21]⚒️: 

- “txt” 文件支持 “//” 注释：注释行将被忽略。

- 添加更多 prompts (人物和姿态):

![](https://github.com/billwuhao/ComfyUI_OneButtonPrompt_Flux/blob/master/images/2025-01-25_22-47-23.png)
![](https://github.com/billwuhao/ComfyUI_OneButtonPrompt_Flux/blob/master/images/2025-01-25_22-49-40.png)
![](https://github.com/billwuhao/ComfyUI_OneButtonPrompt_Flux/blob/master/images/2025-01-25_22-58-00.png)


[2025-01-10]⚒️: 

- `human_pose` → `pose`：任何主体都可以设置姿势。请立即将文件名 `human_pose` 更改为 `pose`.

- 新增 4 个 `add_` 文件, 用来记录我发现的好的提示词, 你也可以将你的提示词分享, 并提交 PR.

- `subject` 新增 `None`, 将 `subject` 变成可选的, 以便在 `lora_trigger_or_prefix` 固定自定义主体, 来生成同一主体下的图片, 以抽大奖.

- 增加测试模式, 请将你需要测试的提示词, 放入文件 `test.txt`(第一次需要新建一个), 然后开启测试模式, 依次进行每条提示词的测试. **注意**: 测试模式时, 除了自定义的 `lora_trigger_or_prefix` 其他所有选项都无效. 如果你是挂机测试, 所有提示测试完后, 转随机生图.

  ![](https://github.com/billwuhao/ComfyUI_OneButtonPrompt_Flux/blob/master/images/2025-01-10_12-07-54.png)

## 用法

- 主体选择 human，则只生成 human 提示
- 选择 other，则只生成 other 提示
- 选择 dual_subject，则包含上述两者
- 如果开启 pose，将生成主体的姿态提示
- 如果开启 style，将生成 style 提示
- lora 触发词/前缀 是可选的，可自定义输入
- 如果你在运行过程中修改了自定义文件, 请开启 refresh 刷新一下, 而不用重启软件.
- 如果固定种子，将只生成一次固定的提示

![alt text](https://github.com/billwuhao/ComfyUI_OneButtonPrompt_Flux/blob/master/images/image-1.png)

(MW:1.2), a beautiful girl, fall asleep, A beautiful lovely alpaca, naive art

- `(MW:1.2)`, 是 lora 触发词
- `a beautiful girl`, 是 human
- `fall asleep`, 是 pose
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

提示词增强在某些情况下效果很差, 完全将提示词的风格, 要表达的意境破坏了, 建议在抽奖时开启.

ComfyUI-Fluxpromptenhancer: https://github.com/marduk191/ComfyUI-Fluxpromptenhancer

# Thank you

Thank you to the amazing [OneButtonPrompt](https://github.com/AIrjen/OneButtonPrompt) for inspiration.