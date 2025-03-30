[中文](README-CN.md) | [English](README.md)

![One Button Prompt for Flux in ComfyUI](https://github.com/billwuhao/ComfyUI_OneButtonPrompt/blob/master/images/example.png)
![](https://github.com/billwuhao/ComfyUI_OneButtonPrompt/blob/master/images/2025-02-19_20-00-01.png)

---

## 一个在 comfyui 中一键辅助生成提示 (用于图像和视频生成等) 的节点.

![](https://github.com/billwuhao/ComfyUI_OneButtonPrompt/blob/master/images/2025-03-30_03-22-41.png)

## 📣 更新

[2025-03-30]⚒️: 完全重构, 发布 v1.0.0. 给您带了不变敬请谅解, 有建议或意见欢迎提RP. 大语言模型太强了, 所以我决定重构这个项目. 只提供启动提示, 配合大语言模型, 可进行润色, 优化, 生成各类提示 (目前只有图像和视频生成合适).

启动提示文件夹放在 `ComfyUI_OneButtonPrompt/files` 目录下, 分类为 `images`(.txt 文件), `prompts`(.md 文件) 和 `prompts-images`(.json 文件). 你可以在其中随意自定义文件, 内容格式需一致, 自定义文件名必须以 `nsfw` 或者 `custom` 开头, 都不会受到更新影响.

![](https://github.com/billwuhao/ComfyUI_OneButtonPrompt/blob/master/images/2025-03-30_03-52-46.png)

`prompts-images` 支持指定词语搜索提示, 例如输入  `cat` 或者 `dog`, 会返得到包含 `cat` 或者 `dog` 的提示.

## 感谢

[OneButtonPrompt](https://github.com/AIrjen/OneButtonPrompt)