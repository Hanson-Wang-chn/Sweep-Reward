# Sweep-Reward: 多模态集成评估模块

## 项目概述

本项目是 **Sweep to Shapes** 任务的评估模块，用于评估双臂机器人将乐高积木扫成指定字母形状（如 "Z"、"E"、"N"）的任务完成质量。

该模块作为系统的"裁判"和"奖励函数生成器"，输入当前裁切后的 RGB 图像和二值化目标图像，输出一个 **[0, 1] 浮点数评分**，用于 RL 奖励计算、任务成功判定及 Failure Detection 触发。

## 功能特性

- **颜色分割**：基于 HSV 空间的红色乐高积木分割
- **形态学处理**：闭运算填补空隙，开运算去除噪点
- **几何层评估**：Elastic IoU 和 F1-Score
- **轮廓层评估**：双向 Chamfer Distance
- **语义层评估**：DINOv2 嵌入相似度
- **感知层评估**：VLM (GPT-4o) Chain-of-Thought 评分
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
├── requirements.txt         # 依赖列表
└── README.md               # 本文件
```

## 安装

### 环境要求

- Python >= 3.8
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

# 指定输入图片
python main.py --current path/to/current.png --goal path/to/goal.png

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

# 完整示例（包含所有模块）
python main.py \
    --current example/example_current.png \
    --goal example/example_goal.png \
    --output ./logs/vis_eval \
    --visualize

# 快速测试示例（仅基础指标）
python main.py \
    --current example/example_current.png \
    --goal example/example_goal.png \
    --output ./logs/vis_eval \
    --visualize \
    --basic-only
```

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
# 批量评估多张图片
images = [load_image(path) for path in image_paths]
results = evaluator.evaluate_batch(images, goal_mask)

for i, result in enumerate(results):
    print(f"Image {i}: score = {result['total_score']:.4f}")
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

## 许可证

本项目仅供研究使用。
