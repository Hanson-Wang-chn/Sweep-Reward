"""
Semantic metrics using DINOv2 embeddings.
Evaluates high-level visual similarity between shapes.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple


class SemanticMetrics:
    """
    Semantic layer metrics: DINOv2 Embedding Similarity.
    Evaluates structural and layout similarity using vision transformers.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize semantic metrics with configuration.
        Loads DINOv2 model once and keeps it in memory.

        Args:
            config: Configuration dictionary.
        """
        self.config = config
        semantic_cfg = config.get("metrics", {}).get("semantic", {}).get("dinov2", {})

        # Model parameters
        self.model_repo = semantic_cfg.get("model_repo", "facebookresearch/dinov2")
        self.model_name = semantic_cfg.get("model_name", "dinov2_vitl14")
        self.layer = semantic_cfg.get("layer", "cls")
        self.resize_input = semantic_cfg.get("resize_input", 224)

        # Goal rendering parameters
        render_cfg = semantic_cfg.get("goal_render", {})
        self.fg_color = np.array(render_cfg.get("foreground_color", [200, 0, 0]))
        self.bg_color = np.array(render_cfg.get("background_color", [200, 200, 200]))

        # Device
        system_cfg = config.get("system", {})
        device_str = system_cfg.get("device", "cuda")
        self.device = torch.device(device_str if torch.cuda.is_available() else "cpu")

        # Color segmentation parameters (HSV ranges for red)
        preprocess_cfg = config.get("preprocess", {})
        color_seg = preprocess_cfg.get("color_segmentation", {})
        hsv1 = color_seg.get("hsv_range_1", {})
        hsv2 = color_seg.get("hsv_range_2", {})
        self.hsv_lower_1 = np.array(hsv1.get("lower", [0, 100, 70]))
        self.hsv_upper_1 = np.array(hsv1.get("upper", [10, 255, 255]))
        self.hsv_lower_2 = np.array(hsv2.get("lower", [170, 100, 70]))
        self.hsv_upper_2 = np.array(hsv2.get("upper", [180, 255, 255]))

        # Model (lazy loading)
        self._model = None

        # Cached goal embedding
        self._cached_goal_embedding: Optional[torch.Tensor] = None

    @property
    def model(self):
        """Lazy load DINOv2 model."""
        if self._model is None:
            self._model = torch.hub.load(
                self.model_repo, self.model_name, pretrained=True
            )
            self._model = self._model.to(self.device)
            self._model.eval()
        return self._model

    def render_goal_image(self, goal_mask: np.ndarray) -> np.ndarray:
        """
        Render binary goal mask as RGB image for DINOv2 processing.
        Creates a colored image that approximates the real Lego appearance.

        Args:
            goal_mask: Binary goal mask (H, W), values 0/1.

        Returns:
            RGB image (H, W, 3), uint8.
        """
        h, w = goal_mask.shape
        rendered = np.zeros((h, w, 3), dtype=np.uint8)

        # Set background color
        rendered[:, :] = self.bg_color

        # Set foreground color where mask is 1
        rendered[goal_mask > 0] = self.fg_color

        return rendered

    def extract_mask_from_image(self, image: np.ndarray) -> np.ndarray:
        """
        Extract binary mask from RGB image using HSV color segmentation.
        NOTE: No morphological operations (closing/opening) are applied here.

        Args:
            image: RGB image (H, W, 3), uint8.

        Returns:
            Binary mask (H, W), values 0/1.
        """
        # Convert RGB to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        # Create masks for both red ranges
        mask1 = cv2.inRange(hsv, self.hsv_lower_1, self.hsv_upper_1)
        mask2 = cv2.inRange(hsv, self.hsv_lower_2, self.hsv_upper_2)

        # Combine masks
        mask = cv2.bitwise_or(mask1, mask2)

        # Convert to binary 0/1
        binary_mask = (mask > 127).astype(np.uint8)

        return binary_mask

    def render_current_image(self, current_image: np.ndarray) -> np.ndarray:
        """
        Extract mask from current image and render it using the same method as goal.
        This ensures DINO compares two images rendered in the same way.

        Args:
            current_image: RGB image (H, W, 3), uint8.

        Returns:
            Rendered RGB image (H, W, 3), uint8.
        """
        # Extract mask using color segmentation only (no morphological operations)
        mask = self.extract_mask_from_image(current_image)

        # Render the mask using the same method as goal
        rendered = self.render_goal_image(mask)

        return rendered

    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess image for DINOv2 input.

        Args:
            image: RGB image (H, W, 3), uint8.

        Returns:
            Preprocessed tensor (1, 3, H, W), float32.
        """
        import cv2

        # Resize if needed
        h, w = image.shape[:2]
        if h != self.resize_input or w != self.resize_input:
            image = cv2.resize(
                image, (self.resize_input, self.resize_input),
                interpolation=cv2.INTER_LINEAR
            )

        # Convert to float and normalize
        image = image.astype(np.float32) / 255.0

        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std

        # Convert to tensor (H, W, C) -> (1, C, H, W)
        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.float().to(self.device)

        return tensor

    def extract_embedding(self, image: np.ndarray) -> torch.Tensor:
        """
        Extract DINOv2 CLS token embedding from image.

        Args:
            image: RGB image (H, W, 3), uint8.

        Returns:
            Embedding tensor (D,), normalized.
        """
        tensor = self.preprocess_image(image)

        with torch.no_grad():
            # DINOv2 forward pass
            features = self.model(tensor)

            # Get CLS token (first token)
            if isinstance(features, dict):
                embedding = features["x_norm_clstoken"]
            else:
                embedding = features

            # Flatten and normalize
            embedding = embedding.squeeze()
            embedding = F.normalize(embedding, p=2, dim=-1)

        return embedding

    def compute_cosine_similarity(
        self, emb1: torch.Tensor, emb2: torch.Tensor
    ) -> float:
        """
        Compute cosine similarity between two embeddings.

        Args:
            emb1: First embedding (D,).
            emb2: Second embedding (D,).

        Returns:
            Cosine similarity in [-1, 1].
        """
        similarity = torch.dot(emb1, emb2)
        return float(similarity.cpu())

    def normalize_similarity(self, cosine_sim: float) -> float:
        """
        Normalize cosine similarity from [-1, 1] to [0, 1].

        Args:
            cosine_sim: Cosine similarity in [-1, 1].

        Returns:
            Normalized score in [0, 1].
        """
        return (cosine_sim + 1.0) / 2.0

    def set_goal(self, goal_mask: np.ndarray) -> torch.Tensor:
        """
        Set and cache goal mask embedding.
        Should be called once at task start to avoid repeated computation.

        Args:
            goal_mask: Binary goal mask (H, W), values 0/1.

        Returns:
            Goal embedding tensor.
        """
        # Render goal mask as RGB
        goal_image = self.render_goal_image(goal_mask)

        # Extract and cache embedding
        self._cached_goal_embedding = self.extract_embedding(goal_image)

        return self._cached_goal_embedding

    def compute(
        self,
        current_image: np.ndarray,
        goal_mask: np.ndarray = None,
        use_cached_goal: bool = True,
    ) -> Dict[str, Any]:
        """
        Compute semantic similarity score.

        Args:
            current_image: Current RGB image (H, W, 3), uint8.
            goal_mask: Binary goal mask (H, W), values 0/1.
                       Only needed if use_cached_goal is False.
            use_cached_goal: Whether to use cached goal embedding.

        Returns:
            Dictionary containing:
                - dino_score: Normalized similarity score [0, 1]
                - cosine_similarity: Raw cosine similarity [-1, 1]
        """
        # Get goal embedding
        if use_cached_goal and self._cached_goal_embedding is not None:
            goal_embedding = self._cached_goal_embedding
        elif goal_mask is not None:
            goal_embedding = self.set_goal(goal_mask)
        else:
            raise ValueError(
                "Goal mask must be provided if no cached embedding exists"
            )

        # Render current image using the same method as goal
        # (extract mask via color segmentation, then render)
        rendered_current = self.render_current_image(current_image)

        # Extract rendered current image embedding
        current_embedding = self.extract_embedding(rendered_current)

        # Compute similarity
        cosine_sim = self.compute_cosine_similarity(current_embedding, goal_embedding)
        normalized_score = self.normalize_similarity(cosine_sim)

        return {
            "dino_score": normalized_score,
            "cosine_similarity": cosine_sim,
        }

    def clear_cache(self):
        """Clear cached goal embedding."""
        self._cached_goal_embedding = None
