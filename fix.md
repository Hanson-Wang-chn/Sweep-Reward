# Add Metrics to Sweep-Reward

## 任务

- 完善代码的 Metrics 部分：
  - 在 Geometric Layer 中添加下面几个指标：
    - 基于 Earth Mover's Distance 的 Sinkhorn Divergence
  - 在 Semantic Layer 中添加下面几个指标：
    - LPIPS (Learned Perceptual Image Patch Similarity)
    - DISTS (Deep Image Structure and Texture Similarity)
  - 注意上述指标都是对**两张相同大小的二值图像**进行比较，可以参考现有的相关代码。
- 对代码结构、配置文件等进行必要的优化，确保新代码可以正常运行。
- 更新 project.md，加入新 Metrics，风格参考原来的写法（比如，需要加入数学公式，总体比较简洁）。
- 更新 README.md，标记为 Version 2.0。对于当前写得不够清楚的部分，可以整体完善。同时需要修改 requirements.txt 等必要的文件。

## 具体要求

- 调整配置文件的格式。我希望每一个具体指标的权重都需要出现在 config/config.yaml 中的 ensemble/weights 字段下（比如 IoU 和 F1 应该分开来写，不能总体写为 geometric）。如果某一项的权重设置为0，则**在代码中跳过该指标的计算**，以节约算力和时间。
- 尽可能优化代码，在保证性能的前提下，提高效率，降低延迟。
- 代码必须结构良好，不同的 Metrics 模块化，逻辑清晰，便于维护，有必要的注释。
- 代码运行一台 Ubuntu 22.04 + 拥有 16GB VRAM 的 NVIDIA 显卡的**无头**服务器上。当前环境中安装了基于 CUDA 11.8 的 torch 2.7.1，**务必处理好 torch 版本问题**。所有的模型都尽可能使用 pytorch 来计算。
