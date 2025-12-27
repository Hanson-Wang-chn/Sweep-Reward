# Allow multiple images input as current image

## 任务

- 请帮我完善命令行参数的格式。允许 --current 后面输入一张或多张图片。
- 当输入多张图片时：
  - DINO只计算一次goal的值，然后分别与每一个current的值比较。
  - 使用多线程并发调用VLM（通过gate之后）
  - 把所有图片的运行结果汇总输出（print到终端 & 存储为json）
  - 所有的任务要服从 config/config.yaml 的配置。单张图片时的参数和多张图片时的参数需要分开控制（仅visualize、调试相关部分，其他部分不需要分开）。注意，输入文件目录（batch模式）相当于一次输入多张图片，使用多张图片的config。
- 其他逻辑保持不变。
- 完善 README，标记为 1.3 版本。
- 注意不要修改无关的文件或代码。
