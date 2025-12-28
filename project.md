# Multi-modal Ensemble Evaluation Module

## 1. 项目概况

### 1.1 项目背景与内容
本项目属于具身智能（Embodied AI）中的非刚体/颗粒介质操控（Granular Media Manipulation）领域。任务名为 **“Sweep to Shapes”**，即通过双臂机器人使用毛刷将散落的乐高积木扫成指定的字母形状（如 "Z", "E", "N"）。

### 1.2 整体 Pipeline
系统采用分层策略架构：
1.  **输入**：自然语言指令 + 视觉观测。
2.  **High-Level Policy (HLP)**：VLM 输出 Primitives（如 `<Sweep> <Box>...`）。
3.  **Low-Level Policy (LLP)**：VLA 根据 Primitives 执行具体动作序列。
4.  **Evaluation Module (本模块)**：在动作执行后介入，对当前积木状态与目标形状进行质量评估。

### 1.3 本模块职责
本模块作为系统的“裁判”和“奖励函数生成器”，输入 **当前裁切后的 RGB 图像** ($I_{curr}$) 和 **二值化目标图像** ($M_{goal}$)，输出一个 **$[0, 1]$ 浮点数评分**。该评分将用于 RL 奖励计算、任务成功判定及 Failure Detection 触发。（如果输入图片不是224*224的格式，需要先进行缩放）

---

## 2. 多模态集成评估模块详细设计

### 2.1 预处理流程 (Preprocessing Pipeline)

此阶段旨在从含噪的 RGB 图像中提取干净的积木分布掩膜。

**输入**：
*   $I_{curr}$: $224 \times 224 \times 3$ (RGB, uint8)，已校正裁切。
*   $M_{goal}$: $224 \times 224$ (Binary, 0/1)，目标形状掩膜。

#### 2.1.1 颜色分割 (Color Segmentation)
针对红色乐高积木，采用 HSV 空间阈值分割。由于红色在 HSV 色环上跨越 $0^{\circ}$，需处理双区间。

*   **逻辑**：
    $$
    M_{raw}(x,y) = \begin{cases} 
    1, & \text{if } H(x,y) \in [0, h_1] \cup [h_2, 180] \land S(x,y) > s_{th} \land V(x,y) > v_{th} \\
    0, & \text{otherwise}
    \end{cases}
    $$
*   **鲁棒性设计**：使用 `cv2.inRange` 分别提取两个红色区间后取并集。

#### 2.1.2 形态学操作 (Morphological Operations)
积木堆积时存在内部缝隙，需通过形态学操作将其融合为连通域，并去除背景噪点。

*   **流程**：
    1.  **闭运算 (Closing)**：先膨胀 (Dilation) 后腐蚀 (Erosion)。填补积木间的微小空隙。
        *   Kernel: $5 \times 5$ Ellipse.
    2.  **开运算 (Opening)**：先腐蚀后膨胀。去除背景中游离的单个积木噪点。
        *   Kernel: $3 \times 3$ Ellipse.
    3.  **高斯模糊 (Gaussian Blur)**（用于生成软掩膜）：
        *   $\sigma = 1.5$，用于平滑边缘锯齿。
*   **输出**：预测掩膜 $M_{pred}$ (Binary, 0/1)。

---

### 2.2 评估核心算法 (Evaluation Core)

#### 2.2.1 几何层 (Geometric Layer)

**1. Elastic IoU (弹性交并比)**
为了解决机械臂执行误差导致的整体刚性位移（平移/旋转）对评分的过度惩罚，引入弹性搜索。

*   **定义**：在小范围内搜索最佳变换参数 $\theta^*, t_x^*, t_y^*$ 以最大化 IoU。
    $$
    \text{ElasticIoU} = \max_{\theta, t_x, t_y} \frac{|T(M_{pred}; \theta, t_x, t_y) \cap M_{goal}|}{|T(M_{pred}; \theta, t_x, t_y) \cup M_{goal}|}
    $$
    其中 $T$ 为仿射变换。
*   **搜索范围**：$\theta \in [-10^{\circ}, 10^{\circ}]$, $t_x, t_y \in [-10px, 10px]$。使用Coarse-to-Fine Grid Search。

**2. F1-Score**
兼顾形状的覆盖率（Recall）和溢出率（Precision）。
$$
\text{Precision} = \frac{|M_{pred} \cap M_{goal}|}{|M_{pred}| + \epsilon}, \quad \text{Recall} = \frac{|M_{pred} \cap M_{goal}|}{|M_{goal}| + \epsilon}
$$
$$
S_{geo} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall} + \epsilon}
$$

**3. Sinkhorn Divergence (EMD 近似)**
在几何层加入基于熵正则的地球移动距离，衡量两个二值分布之间的运输代价。
1. 将掩膜下采样为概率分布 $a, b$。
2. 构造像素网格坐标 $X$，计算代价矩阵 $C = ||X_i - X_j||_2$。
3. 使用 Sinkhorn-Knopp 迭代求解：
$$
\mathcal{W}_\varepsilon(a, b) = \min_{\pi \in \Pi(a, b)} \langle \pi, C \rangle + \varepsilon \mathrm{KL}(\pi)
$$
4. Sinkhorn Divergence:
$$
D_{\text{sink}}(a, b) = \mathcal{W}_\varepsilon(a, b) - \tfrac{1}{2}\mathcal{W}_\varepsilon(a, a) - \tfrac{1}{2}\mathcal{W}_\varepsilon(b, b)
$$
5. 映射到相似度：
$$
S_{\text{sink}} = \exp(-\lambda \cdot D_{\text{sink}})
$$
其中 $\lambda$ 由配置文件 `normalization_lambda` 控制。

#### 2.2.2 轮廓层 (Contour Layer)

**双向 Chamfer Distance (倒角距离)**
关注边缘细节的平滑度和拓扑一致性。

*   **预处理**：对 $M_{pred}$ 和 $M_{goal}$ 分别进行 Canny 边缘检测，得到点集 $P$ 和 $G$。
*   **距离计算**：
    $$
    D_{chamfer}(P, G) = \frac{1}{|P|} \sum_{p \in P} \min_{g \in G} ||p - g||_2 + \frac{1}{|G|} \sum_{g \in G} \min_{p \in P} ||g - p||_2
    $$
    *工程实现：使用 `cv2.distanceTransform` 加速计算。*
*   **标准化 (Normalization)**：将距离映射到 $[0, 1]$ 相似度分数。
    $$
    S_{contour} = e^{-\lambda \cdot D_{chamfer}(P, G)}
    $$
    推荐 $\lambda = 0.1$ (需根据像素尺度调整)。

#### 2.2.3 语义特征层 (Semantic Layer)

**DINOv2 Embedding Similarity**
利用 DINOv2 对物体几何结构的敏感性，评估高层视觉相似度。

*   **输入**：
    *   $M_{pred}$: 当前观测图像的二值预测掩膜。
    *   $M_{goal}$: 二值化目标图像
*   **模型**：`dinov2_vitl14` (ViT-Large)。
*   **计算**：
    $$
    v_{curr} = \text{DINOv2}(M_{pred})_{[CLS]}, \quad v_{goal} = \text{DINOv2}(M_{goal})_{[CLS]}
$$
$$
S_{semantic} = \frac{v_{curr} \cdot v_{goal}}{||v_{curr}||_2 \cdot ||v_{goal}||_2} \quad (\text{Cosine Similarity})
$$
*注意：需将余弦相似度 [-1, 1] 归一化至 [0, 1]。*

**LPIPS (Learned Perceptual Image Patch Similarity)**
在二值图上计算感知距离，距离越小相似度越高。
$$
S_{\text{LPIPS}} = \exp(-\lambda_{\text{lpips}} \cdot d_{\text{LPIPS}})
$$
其中 $d_{\text{LPIPS}}$ 由预训练感知网络给出，$\lambda_{\text{lpips}}$ 为配置系数。

**DISTS (Deep Image Structure and Texture Similarity)**
同时关注结构与纹理信息，适合二值图的整体形态对齐。
$$
S_{\text{DISTS}} = \exp(-\lambda_{\text{dists}} \cdot d_{\text{DISTS}})
$$
同样对距离取指数映射以得到 $[0, 1]$ 相似度。

#### 2.2.4 感知层 (Perceptual Layer)

**VLM Scoring with CoT**
处理边缘情况（如断裂、堆叠过高），充当最终裁判。

*   **API**：OpenRouter (调用 GPT-4o 等)。
*   **输入**：拼接图像 (Binary Current Image |
Binary Goal Image)。
*   **Prompt 策略 (CoT)**：
    1.  **观察**：描述当前积木堆与目标的差异（断点、多余积木、形状畸变）。
    2.  **推理**：分析这些缺陷是否严重影响字母的可读性。
    3.  **打分**：输出 JSON 格式 `{ "reasoning": "...", "score": 0.85 }`。
*   **输出**：解析 JSON 提取 $S_{perc}$。

---

### 2.3 集成机制 (The Ensemble Mechanism)

采用 **加权门控机制 (Weighted Gating Mechanism)** 以平衡精度与计算成本。

**流程逻辑**：

1.  **计算基础指标**：首先计算 $S_{geo}$ (F1/IoU) 和 $S_{contour}$。
2.  **早停门控 (Gating)**：
    *   如果 $S_{geo} < \tau_{fail}$ (如 0.4)，判定为完全失败，直接返回 $S_{geo}$，**不调用** DINOv2 和 VLM（节省算力/API Cost）。
3.  **全量计算**：若通过门控，计算 $S_{semantic}$ 和 $S_{perc}$。
4.  **加权融合**：
    $$
    S_{final} = w_1 \cdot S_{geo} + w_2 \cdot S_{contour} + w_3 \cdot S_{semantic} + w_4 \cdot S_{perc}
    $$
    其中 $\sum w_i = 1$。
    实现中对每个细分指标单独配置权重，权重设为 0 时会跳过该指标计算以节约算力。

---

## 3. 工程实施规范

### 3.1 推荐目录结构
```text
Sweep-Reward/
├── config/
│   └── config.yaml      # 所有超参数
├── src/
│   ├── __init__.py
│   ├── preprocessor.py       # Color Seg, Morphology
│   ├── metrics/
│   │   ├── geometric.py      # Elastic IoU, F1
│   │   ├── contour.py        # Chamfer Distance
│   │   └── semantic.py       # DINOv2 Wrapper
│   ├── vlm_client.py         # OpenRouter API Handler
│   └── evaluator.py          # Ensemble Logic (Main Entry)
├── utils/
│   └── visualization.py      # 调试用，可视化Mask和热力图
├── main.py              # 测试脚本
└── requirements.txt
```

### 3.2 配置文件示例 (`config/config.yaml`)
所有硬编码参数必须移至配置文件。

```yaml
# ==============================================================================
# Sweep to Shapes - 多模态集成评估模块配置文件
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. 系统基础设置 (System Settings)
# ------------------------------------------------------------------------------
system:
  device: "cuda"            # 计算设备: "cuda" 或 "cpu"
  seed: 42                  # 随机种子，保证 DINO/VLM 采样的一致性
  image_size: [224, 224]    # 输入图像的标准化尺寸 [H, W]

# ------------------------------------------------------------------------------
# 2. 预处理流水线 (Preprocessing Pipeline)
# 目标：从含噪 RGB 图像中提取干净的乐高积木 Mask
# ------------------------------------------------------------------------------
preprocess:
  # 是否对 current 图片应用形态学操作（开闭运算）
  # true: 对 current 图片使用与 goal 相同的形态学预处理
  # false: current 图片仅进行颜色分割，不使用形态学操作
  preprocess_current: true

  # 颜色分割：红色通常跨越 HSV 的 0 度，需要两个区间
  color_segmentation:
    hsv_range_1:
      lower: [0, 100, 70]   # [H, S, V]
      upper: [10, 255, 255]
    hsv_range_2:
      lower: [170, 100, 70]
      upper: [180, 255, 255]

  # 形态学操作：用于填补积木间隙(Closing)和去除噪点(Opening)
  morphology:
    closing:
      kernel_size: 5        # 闭运算核大小 (推荐 5x5)
      iterations: 1         # 执行次数，次数越多填补能力越强
      kernel_shape: "ellipse" # 核形状: "rect", "cross", "ellipse"
    opening:
      kernel_size: 3        # 开运算核大小 (推荐 3x3)
      iterations: 1         # 执行次数
      kernel_shape: "ellipse"

  # 平滑处理：生成软掩膜 (Soft Mask) 用于梯度计算
  smoothing:
    enable: true
    gaussian_kernel: 5      # 高斯模糊核大小 (必须是奇数)
    sigma: 1.5              # 标准差

# ------------------------------------------------------------------------------
# 3. 评估指标配置 (Metrics Configuration)
# ------------------------------------------------------------------------------
metrics:
  # --- A. 几何层 (Geometric Layer) ---
  geometric:
    # F1-Score 配置
    f1_score:
      epsilon: 1.0e-6       # 防止除零微小量

    # Elastic IoU (弹性 IoU) 配置
    elastic_iou:
      enable: true          # 是否启用弹性搜索（若 False 则计算标准 IoU）
      method: "grid_search"
      downsample_factor: 2  # 搜索时的降采样倍率 (1为原图)，越大越快但精度越低
      max_translation: 10   # 最大平移像素范围 (px)
      max_rotation: 10      # 最大旋转角度范围 (degrees)
      step_translation: 2   # 平移搜索步长 (px)
      step_rotation: 2      # 旋转搜索步长 (degrees)

  # --- B. 轮廓层 (Contour Layer) ---
  contour:
    chamfer_distance:
      edge_detection: "canny"
      canny_low_thresh: 50  # Canny 边缘检测低阈值
      canny_high_thresh: 150 # Canny 边缘检测高阈值
      normalization_lambda: 0.1 # 指数归一化系数: score = exp(-lambda * dist)
      truncate_distance: 20.0   # 距离截断阈值，防止离群点过度拉低分数

  # --- C. 语义特征层 (Semantic Layer) ---
  semantic:
    dinov2:
      model_repo: "facebookresearch/dinov2"
      model_name: "dinov2_vitl14" # 可选: vits14, vitb14, vitl14, vitg14
      layer: "cls"          # 提取特征层: "cls" 或 "patch" (本项目推荐 cls)
      resize_input: 224     # DINO 输入尺寸 (通常为 14 的倍数)

  # --- D. 感知层 (Perceptual Layer) ---
  perceptual:
    vlm:
      # model_name: "google/gemini-3-pro-preview"
      model_name: "openai/gpt-5.2"
      api_key_env_var: "OPENROUTER_API_KEY" # 环境变量名，不要直接写 Key
      base_url: "https://openrouter.ai/api/v1"

      # 请求参数
      temperature: 0.0      # 低温度保证评分稳定性
      max_tokens: 1024
      timeout: 30           # 请求超时时间 (秒)
      max_retries: 5        # 失败重试次数 All images are black-and-white masks. White pixels represent LEGO bricks filled material. Black pixels represent empty space.

      # 提示词模板 (CoT)
      system_prompt: |
        # Role
        You are a pragmatic robot evaluator assessing a "Lego Sweeping" task. You understand that sweeping granular objects (small Lego bricks) results in naturally rough edges and noise. **Perfection is not required.** Your goal is to judge if the robot has successfully formed the *semantic shape*.

        # Input
        - **Image 1**: The Goal Shape (Ideal binary mask).
        - **Image 2**: The Current Observation (Actual state of bricks).

        # Context & Lenience Guidelines
        - **Granular Material**: The objects are small bricks. Straight lines will inevitably look jagged or "pixelated." **Do not penalize for jagged edges.**
        - **Stray Bricks**: A moderate amount of scattered bricks (noise) outside the main shape is acceptable and expected. **Ignore isolated background noise.**
        - **Focus**: Prioritize **Topological Correctness** (e.g., "Does it look like the letter?") over **Geometric Precision** (e.g., "Are the lines perfectly straight?").

        # Evaluation Criteria (0.1 - 0.9)
        Please assign a score based on the following relaxed standards:

        - **0.1 (Unrecognizable)**: The bricks are piled randomly. No coherent shape is visible.
        - **0.3 (Attempted but Failed)**: You can guess what the robot tried to do, but major parts are missing (e.g., an 'E' missing the middle bar) or the shape is broken into disconnected islands.
        - **0.5 (Passable)**: The shape is clearly recognizable as the target letter/symbol. It may be significantly thicker/thinner than the goal, or have 1-2 moderate gaps, but the identity is unambiguous.
        - **0.7 (Good Success)**: The shape matches the goal's topology perfectly. The strokes are connected. There might be some fuzzy edges or a few scattered bricks nearby, but the main structure is solid.
        - **0.9 (Excellent)**: The shape is distinct, correctly oriented, and topologically complete. Even if the borders are wavy or not perfectly aligned with the goal mask pixels, visually, it is a great result for a robot.

        # Reasoning Steps
        1. **Identify**: Can you instantly recognize the shape in Image 2 as the shape in Image 1 without guessing? If yes, start from score 0.5.
        2. **Topology Check**: Are all necessary strokes present and connected? (e.g., "Z" has top, diagonal, bottom). If yes, boost score to 0.7+.
        3. **Noise Tolerance**: Is the noise distracting? If the main shape is prominent enough to ignore the noise, maintain the high score.

        # Output Format
        {
          "reasoning": "Brief justification focusing on recognizability and topology...",
          "score": <float between 0.1 and 0.9>
        }



# ------------------------------------------------------------------------------
# 4. 集成机制 (Ensemble Mechanism)
# ------------------------------------------------------------------------------
ensemble:
  # 门控机制：节省算力
  gating:
    enable: true
    threshold: 0.4     # 如果基础几何得分 (Geometric Score) 低于此阈值，直接返回失败，不调用 DINO/VLM

  # 各模块权重 (总和应为 1.0)
  weights:
    geometric: 0.35    # IoU/F1: 负责基础重合度
    contour: 0.25      # Chamfer: 负责边缘细节
    semantic: 0.20     # DINO: 负责结构与布局
    perceptual: 0.20   # VLM: 负责人类观感与Corner Case

# ------------------------------------------------------------------------------
# 5. 调试与日志 (Debugging & Logging)
# ------------------------------------------------------------------------------
debug:
  # 单张图片模式设置
  single_image:
    enable_visualization: true  # 是否保存处理过程中的图片 (Mask, Edges, Heatmaps)
    save_metrics_to_json: true  # 将每次评估的详细指标保存为 JSON

  # 多张图片/批量模式设置
  multi_image:
    enable_visualization: false # 批量模式下通常关闭可视化以提高效率
    save_metrics_to_json: true  # 保存汇总的 JSON 结果

  vis_output_dir: "./logs"
  log_level: "INFO"           # "DEBUG", "INFO", "WARNING", "ERROR"

  # 向后兼容的默认设置（当未指定模式时使用）
  enable_visualization: true
  save_metrics_to_json: true

  # VLM 调试设置
  save_vlm_imgs: true        # 是否保存 VLM 的输入图片（调试用）
```

### 3.3 编码与数据传递规范

1.  **数据类型**：
    *   图像数据在模块间传递时，统一使用 **NumPy Array** (`np.uint8` for images, `np.float32` for masks)。
    *   涉及 GPU 计算（DINOv2, Chamfer Distance）时，在函数内部转为 `torch.Tensor`，计算后转回 CPU，避免显存泄漏。
2.  **延迟优化**：
    *   DINOv2 模型应在 `Evaluator` 类初始化时加载并常驻显存（Global Singleton），严禁在每次评估时重新加载模型。
    *   由于 $M_{goal}$ 在任务开始后是不变的，不要在每次 evaluate() 时都重新计算 $I_{goal\_render}$ 的 DINO Embedding。因此，在 Evaluator 初始化或 set_goal() 时计算一次 v_goal 并缓存。每次循环只计算 v_curr。
3.  **结果输出**：
    *   `Evaluator.evaluate()` 方法应返回一个字典，包含最终得分及各子项得分，便于 Log 记录和分析：
        ```python
        {
            "total_score": 0.82,
            "details": {
                "iou": 0.75,
                "chamfer": 0.88,
                "dino": 0.91,
                "vlm": 0.80
            },
            "gating_passed": True
        }
        ```
