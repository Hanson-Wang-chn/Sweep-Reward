# Add `generate_pseudo_label.py` to Sweep-Reward Project

## 任务

- 当前项目代码面临一个问题，DINO和VLM比较的是一张真实图片和一张渲染图片的差异，每一次比较出来相似度都很低。
- 请参考`generate_pseudo_label.py`中的代码的处理方式，在输入DINO和VLM之前，先把current image（真实照片）中的mask提取出来，然后用config.yaml中的方式渲染。这样DINO和VLM就是在比较两张同相同方法渲染后照片的相似度。
- 注意这里不要进行开闭操作预处理（`generate_pseudo_label.py`中是有的）。
- 注意不要修改任何其他文件。
