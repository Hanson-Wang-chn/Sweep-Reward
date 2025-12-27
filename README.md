# Sweep-Reward: 多模态集成评估模块

**Version 1.3.1**

## 项目概述

本项目是 **Sweep to Shapes** 任务的评估模块，用于评估双臂机器人将乐高积木扫成指定字母形状（如 "Z"、"E"、"N"）的任务完成质量。

该模块作为系统的"裁判"和"奖励函数生成器"，输入当前裁切后的 RGB 图像和二值化目标图像，输出一个 **[0, 1] 浮点数评分**，用于 RL 奖励计算、任务成功判定及 Failure Detection 触发。

## 功能特性

- **颜色分割**：基于 HSV 空间的红色乐高积木分割
- **形态学处理**：闭运算填补空隙，开运算去除噪点
- **几何层评估**：Elastic IoU 和 F1-Score
- **轮廓层评估**：双向 Chamfer Distance
- **语义层评估**：DINOv2 嵌入相似度（使用二值图像输入）
- **感知层评估**：VLM (GPT-4o) Chain-of-Thought 评分（使用二值图像输入）
- **加权门控机制**：节省计算资源，低分时跳过高成本模块

## 目录结构

```
Sweep-Reward/
├── config/
│   └── config.yaml          # 配置文件（所有超参数）
├── src/
│   ├── __init__.py
│   ├── preprocessor.py      # 颜色分割、形态学处理
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── geometric.py     # Elastic IoU、F1
│   │   ├── contour.py       # Chamfer Distance
│   │   └── semantic.py      # DINOv2 Wrapper
│   ├── vlm_client.py        # OpenRouter API Handler
│   └── evaluator.py         # 集成逻辑（主入口）
├── utils/
│   ├── __init__.py
│   └── visualization.py     # 可视化工具
├── example/
│   ├── example_current.png  # 示例当前图像
│   └── example_goal.png     # 示例目标图像
├── logs/
│   └── vis_eval/            # 可视化输出目录
├── main.py                  # 测试脚本
├── generate_pseudo_label.py # 伪标签生成工具
├── requirements.txt         # 依赖列表
└── README.md               # 本文件
```

## 安装

### 环境要求

- Python >= 3.11
- CUDA（推荐，用于 DINOv2 加速）

### 安装依赖

```bash
# 创建虚拟环境（推荐）
conda create -n sweep-reward python=3.11 -y
conda activate sweep-reward

# 安装合适的pytorch（示例）
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118

# 安装依赖
pip install -r requirements.txt
```

### 配置 API Key（可选）

如果需要使用 VLM 评估（GPT-4o），请设置 OpenRouter API Key：

```bash
export OPENROUTER_API_KEY="your-api-key-here"
```

## 使用方法

### 命令行运行

```bash
# 基础运行（使用默认示例图片）
python main.py

# 指定单张输入图片
python main.py --current path/to/current.png --goal path/to/goal.png

# 指定多张输入图片
python main.py --current img1.png img2.png img3.png --goal path/to/goal.png

# 指定图片目录（批量处理）
python main.py --current path/to/image_dir/ --goal path/to/goal.png

# 混合输入（多张图片 + 目录）
python main.py --current img1.png path/to/dir/ img2.png --goal path/to/goal.png

# 启用可视化输出
python main.py --visualize

# 跳过 VLM 评估（无需 API Key）
python main.py --skip-vlm

# 跳过 DINOv2 评估（无需下载模型）
python main.py --skip-dino

# 仅运行基础指标（无需 DINOv2 和 VLM，推荐用于快速测试）
python main.py --basic-only --visualize

# 指定配置文件
python main.py --config config/config.yaml

# 设置并发 VLM 调用的最大线程数（默认为 4）
python main.py --current path/to/dir/ --goal goal.png --max-workers 8

# 完整示例（包含所有模块）
python main.py \
    --current example/example_current.png \
    --goal example/example_goal.png \
    --output ./logs/vis_eval \
    --visualize

# 批量评估示例
python main.py \
    --current data/batch_images/ \
    --goal example/example_goal.png \
    --output ./logs/batch_eval \
    --max-workers 4

# 快速测试示例（仅基础指标）
python main.py \
    --current example/example_current.png \
    --goal example/example_goal.png \
    --output ./logs/vis_eval \
    --visualize \
    --basic-only
```

### 生成伪标签（Goal 图像）

使用 `generate_pseudo_label.py` 从真实乐高照片生成二值化目标图像（伪标签）：

```bash
# 单张图片处理
python generate_pseudo_label.py --input data/end-0.png --output data/goal-0.png

# 批量处理目录
python generate_pseudo_label.py --input_dir data/raw --output_dir data/goals

# 指定配置文件
python generate_pseudo_label.py --input data/photo.png --config config/config.yaml
```

**功能说明**：
- 从真实乐高照片中提取红色积木区域
- 使用 HSV 颜色分割识别红色乐高
- 应用形态学操作（闭运算填补空隙、开运算去除噪点）
- 输出二值化图像（0/255），可直接作为评估模块的目标图像

### 作为 Python 模块使用

```python
import yaml
import cv2
import numpy as np
from src.evaluator import Evaluator

# 加载配置
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# 初始化评估器
evaluator = Evaluator(config)

# 加载图片
current_image = cv2.imread("example/example_current.png")
current_image = cv2.cvtColor(current_image, cv2.COLOR_BGR2RGB)

goal_mask = cv2.imread("example/example_goal.png")
goal_mask = cv2.cvtColor(goal_mask, cv2.COLOR_BGR2RGB)

# 设置目标（只需执行一次，会缓存 DINOv2 嵌入）
evaluator.set_goal(goal_mask)

# 执行评估
result = evaluator.evaluate(current_image)

# 获取结果
print(f"总分: {result['total_score']:.4f}")
print(f"详细分数: {result['details']}")
print(f"是否通过门控: {result['gating_passed']}")
```

### 批量评估

```python
# 方式一：使用 evaluate_batch 方法
images = [load_image(path) for path in image_paths]
results = evaluator.evaluate_batch(images, goal_mask)

for i, result in enumerate(results):
    print(f"Image {i}: score = {result['total_score']:.4f}")

# 方式二：命令行批量评估（推荐，支持并发 VLM 调用）
# python main.py --current img1.png img2.png img3.png --goal goal.png
# python main.py --current path/to/image_dir/ --goal goal.png --max-workers 4
```

## 配置说明

配置文件 `config/config.yaml` 包含所有可调参数：

### 系统设置

```yaml
system:
  device: "cuda"            # 计算设备: "cuda" 或 "cpu"
  seed: 42                  # 随机种子
  image_size: [224, 224]    # 输入图像标准化尺寸
```

### 预处理参数

```yaml
preprocess:
  color_segmentation:       # 红色 HSV 分割范围
    hsv_range_1:
      lower: [0, 100, 70]
      upper: [10, 255, 255]
    hsv_range_2:
      lower: [170, 100, 70]
      upper: [180, 255, 255]
  morphology:               # 形态学操作参数
    closing:
      kernel_size: 5
      iterations: 2
    opening:
      kernel_size: 3
      iterations: 1
```

### 评估指标权重

```yaml
ensemble:
  gating:
    enable: true
    threshold: 0.4          # 门控阈值
  weights:
    geometric: 0.35         # 几何层权重
    contour: 0.25           # 轮廓层权重
    semantic: 0.20          # 语义层权重
    perceptual: 0.20        # 感知层权重
```

## 输出格式

评估结果为字典格式：

```python
{
    "total_score": 0.82,           # 最终加权得分 [0, 1]
    "details": {
        "iou": 0.75,               # Elastic IoU
        "f1": 0.78,                # F1-Score
        "chamfer": 0.88,           # Chamfer 相似度
        "dino": 0.91,              # DINOv2 相似度
        "vlm": 0.80                # VLM 评分
    },
    "gating_passed": True,         # 是否通过门控
    "raw_metrics": {...}           # 原始指标详情
}
```

## 评估流程

1. **预处理**：颜色分割 → 形态学处理 → 生成预测掩膜
2. **几何评估**：计算 Elastic IoU 和 F1-Score
3. **轮廓评估**：计算双向 Chamfer Distance
4. **门控检查**：若几何分数低于阈值，直接返回（节省计算）
5. **语义评估**：使用 DINOv2 计算嵌入相似度
6. **感知评估**：调用 VLM API 进行 CoT 评分
7. **加权融合**：计算最终得分

## 可视化输出

启用 `--visualize` 后，会在输出目录生成以下文件：

- `evaluation_result.png`：综合评估结果可视化
- `mask_comparison.png`：预测掩膜与目标掩膜对比
- `metrics_*.json`：详细评估指标 JSON 文件

## 注意事项

1. **首次运行**：DINOv2 模型会自动下载（约 1.2GB），请确保网络畅通
2. **GPU 内存**：DINOv2 ViT-Large 模型需要约 3GB 显存
3. **API 费用**：VLM 评估会产生 OpenRouter API 费用，可使用 `--skip-vlm` 跳过
4. **目标缓存**：`set_goal()` 会缓存目标的 DINOv2 嵌入，避免重复计算

## 更新日志

### Version 1.3.1
- **VLM 输入图片保存**：新增 `save_vlm_imgs` 调试选项
  - 在 `config/config.yaml` 的 `debug` 部分设置 `save_vlm_imgs: true` 启用
  - 启用后会将每次 VLM 调用时发送给 VLM 的图片保存到日志目录
  - 多图片评估时，每张 current 图片的 VLM 输入会分别保存，文件名包含时间戳和图片索引
  - 用于调试 VLM 的输入和输出

### Version 1.3
- **多图片输入支持**：`--current` 参数现在支持多张图片或目录输入
  - 可以指定多张图片：`--current img1.png img2.png img3.png`
  - 可以指定目录：`--current path/to/image_dir/`
  - 可以混合使用：`--current img1.png path/to/dir/ img2.png`
- **并发 VLM 调用**：多图片模式下使用多线程并发调用 VLM API，大幅提升批量评估效率
  - 新增 `--max-workers` 参数控制并发线程数（默认为 4）
  - DINO 目标嵌入只计算一次，复用于所有图片的对比
- **分离单/多图片配置**：在 `config.yaml` 中区分单张图片和多张图片模式的调试设置
  - `debug.single_image`: 单张图片模式的可视化和 JSON 保存设置
  - `debug.multi_image`: 批量模式的可视化和 JSON 保存设置（默认关闭可视化以提高效率）
- **汇总输出**：批量评估结果汇总输出到终端和 JSON 文件
  - 终端显示：每张图片的分数和统计摘要（均值、最大、最小、标准差）
  - JSON 输出：`batch_results_<timestamp>.json` 包含完整的评估结果和统计信息

### Version 1.2
- **移除渲染功能**：DINO 和 VLM 的输入改为黑白二值图像（224x224），不再渲染为彩色图像
- **简化配置**：移除 `config.yaml` 中的 `goal_render` 配置项（foreground_color、background_color）
- **代码优化**：
  - `src/metrics/semantic.py`：使用 `mask_to_binary_image()` 替代 `render_goal_image()`
  - `src/vlm_client.py`：VLM 比较图像改为二值图像对比
  - `src/evaluator.py`：移除渲染图像的保存

### Version 1.1
- 添加伪标签生成模块 `generate_pseudo_label.py`
- 添加渲染可视化功能

### Version 1.0
- 初始版本

## 许可证

本项目仅供研究使用。
