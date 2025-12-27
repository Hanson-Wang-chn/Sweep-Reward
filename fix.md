# Add `generate_pseudo_label.py` to Sweep-Reward Project

## 任务

- 请帮我编写`generate_pseudo_label.py`的代码。这是一个预处理模块，能从数据集中生成goal image。
  - 输入一张积木的真实照片，类似`example/example_current.png`。
  - 输出一张二值图片，类似`example/example_goal.png`。
  - 需要用当前项目中的方法，把真实照片中的mask提取出来，并进行预处理（开闭操作），然后保存二值图像到指定目录中。
- 需要支持命令行参数。
- 注意不要修改任何其他文件。
