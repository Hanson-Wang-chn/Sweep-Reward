# Allow multiple images input as current image

## 任务

- 请在 config/config.yaml 中添加 preprocess_current 参数，当其为 true 时，保存 VLM 的输入图片到日志目录中。注意当有多张 current 图片输入时，要分别保存每一次VLM调用时发送给VLM的图片。该功能为调试使用。
- 完善 README，标记为 1.4 版本。
- 注意不要修改无关的文件或代码。
