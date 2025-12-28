"""
Semantic metrics using DINOv2 embeddings.
Evaluates high-level visual similarity between shapes.
"""

import logging
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, Set


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
        self.logger = logging.getLogger(__name__)
        semantic_root = config.get("metrics", {}).get("semantic", {})
        semantic_cfg = semantic_root.get("dinov2", {})

        # DINOv2 parameters
        self.dino_enable = semantic_cfg.get("enable", True)
        self.model_repo = semantic_cfg.get("model_repo", "facebookresearch/dinov2")
        self.model_name = semantic_cfg.get("model_name", "dinov2_vitl14")
        self.layer = semantic_cfg.get("layer", "cls")
        self.resize_input = semantic_cfg.get("resize_input", 224)

        # LPIPS parameters
        lpips_cfg = semantic_root.get("lpips", {})
        self.lpips_enable = lpips_cfg.get("enable", True)
        self.lpips_net = lpips_cfg.get("net", "alex")
        self.lpips_norm_lambda = lpips_cfg.get("normalization_lambda", 5.0)

        # DISTS parameters
        dists_cfg = semantic_root.get("dists", {})
        self.dists_enable = dists_cfg.get("enable", True)
        self.dists_norm_lambda = dists_cfg.get("normalization_lambda", 15.0)

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

        # Models (lazy loading)
        self._model = None  # DINOv2
        self._lpips_model = None
        self._dists_model = None
        self._lpips_load_failed = False
        self._dists_load_failed = False

        # Cached goal representations
        self._cached_goal_embedding: Optional[torch.Tensor] = None
        self._cached_goal_tensor_01: Optional[torch.Tensor] = None
        self._cached_goal_mask: Optional[np.ndarray] = None

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

    @property
    def lpips_model(self):
        """Lazy load LPIPS model."""
        if self._lpips_model is None and not self._lpips_load_failed:
            import lpips

            try:
                # Forward call handles normalization via `normalize=True`; constructor does not support it.
                self._lpips_model = lpips.LPIPS(net=self.lpips_net, verbose=False)
                self._lpips_model = self._lpips_model.to(self.device)
                self._lpips_model.eval()
            except Exception as exc:
                self._lpips_load_failed = True
                self.lpips_enable = False
                self.logger.warning("LPIPS model load failed, disabling LPIPS metric: %s", exc)
                self._lpips_model = None
        return self._lpips_model

    @property
    def dists_model(self):
        """Lazy load DISTS model."""
        if self._dists_model is None and not self._dists_load_failed:
            try:
                from DISTS_pytorch import DISTS

                self._dists_model = DISTS().to(self.device)
                self._dists_model.eval()
            except Exception as exc:
                self._dists_load_failed = True
                self.dists_enable = False
                self.logger.warning("DISTS model load failed, disabling DISTS metric: %s", exc)
                self._dists_model = None
        return self._dists_model

    def mask_to_binary_image(self, mask: np.ndarray) -> np.ndarray:
        """
        Convert binary mask to 3-channel black-white image for DINOv2 processing.

        Args:
            mask: Binary mask (H, W), values 0/1.

        Returns:
            RGB image (H, W, 3), uint8, black (0) and white (255).
        """
        # Convert to 0-255 range
        binary_255 = (mask * 255).astype(np.uint8)

        # Create 3-channel image
        binary_rgb = np.stack([binary_255, binary_255, binary_255], axis=-1)

        return binary_rgb

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

    def get_binary_image_from_current(self, current_image: np.ndarray) -> np.ndarray:
        """
        Extract mask from current image and convert to binary image.
        This ensures DINO compares two binary images in the same format.

        Args:
            current_image: RGB image (H, W, 3), uint8.

        Returns:
            Binary RGB image (H, W, 3), uint8, black (0) and white (255).
        """
        # Extract mask using color segmentation only (no morphological operations)
        mask = self.extract_mask_from_image(current_image)

        # Convert mask to binary image
        binary_image = self.mask_to_binary_image(mask)

        return binary_image

    def _prepare_binary_image(self, mask: np.ndarray) -> np.ndarray:
        """
        Ensure binary mask is converted to a 3-channel 0/255 image and resized.
        """
        image = self.mask_to_binary_image(mask)
        h, w = image.shape[:2]
        if h != self.resize_input or w != self.resize_input:
            image = cv2.resize(
                image, (self.resize_input, self.resize_input), interpolation=cv2.INTER_LINEAR
            )
        return image

    def _to_tensor_01(self, image: np.ndarray) -> torch.Tensor:
        """
        Convert a binary RGB image to a torch tensor in [0, 1].
        """
        tensor = torch.from_numpy(image.astype(np.float32) / 255.0)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)
        return tensor

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

    def set_goal(self, goal_mask: np.ndarray, metrics_to_cache: Optional[Set[str]] = None) -> None:
        """
        Cache goal representations for requested semantic metrics.
        """
        metrics = set(metrics_to_cache or {"dino", "lpips", "dists"})
        if not self.dino_enable:
            metrics.discard("dino")
        if not self.lpips_enable:
            metrics.discard("lpips")
        if not self.dists_enable:
            metrics.discard("dists")
        self._cached_goal_mask = goal_mask

        goal_image = self._prepare_binary_image(goal_mask)
        cache_tensors = {"lpips", "dists"} & metrics

        if "dino" in metrics and self.dino_enable:
            self._cached_goal_embedding = self.extract_embedding(goal_image)
        else:
            self._cached_goal_embedding = None

        if cache_tensors:
            goal_tensor_01 = self._to_tensor_01(goal_image)
            self._cached_goal_tensor_01 = goal_tensor_01
        else:
            self._cached_goal_tensor_01 = None

    def compute(
        self,
        pred_mask: Optional[np.ndarray] = None,
        goal_mask: Optional[np.ndarray] = None,
        metrics_to_compute: Optional[Set[str]] = None,
        use_cached_goal: bool = True,
        current_image: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Compute semantic similarity metrics on binary masks.

        Args:
            pred_mask: Predicted binary mask (H, W), values 0/1.
            goal_mask: Goal binary mask (H, W), values 0/1 (optional if cached).
            metrics_to_compute: Subset of semantic metrics to compute.
            use_cached_goal: Whether to reuse cached goal representations.
            current_image: Optional RGB image to derive mask if pred_mask is None.
        """
        requested = set(metrics_to_compute or {"dino", "lpips", "dists"})
        if not self.dino_enable:
            requested.discard("dino")
        if not self.lpips_enable:
            requested.discard("lpips")
        if not self.dists_enable:
            requested.discard("dists")
        results: Dict[str, Any] = {
            "dino_score": None,
            "cosine_similarity": None,
            "lpips_distance": None,
            "lpips_score": None,
            "dists_distance": None,
            "dists_score": None,
        }

        if pred_mask is None:
            if current_image is None:
                raise ValueError("Either pred_mask or current_image must be provided for semantic metrics.")
            pred_mask = self.extract_mask_from_image(current_image)

        if goal_mask is None:
            if use_cached_goal and self._cached_goal_mask is not None:
                goal_mask = self._cached_goal_mask
            else:
                raise ValueError("Goal mask must be provided if no cached goal is available.")

        # Prepare binary images/tensors
        pred_binary_image = self._prepare_binary_image(pred_mask)
        goal_binary_image = self._prepare_binary_image(goal_mask)

        goal_tensor_01 = None
        if ({"lpips", "dists"} & requested):
            goal_tensor_01 = self._cached_goal_tensor_01 if use_cached_goal else None
            if goal_tensor_01 is None:
                goal_tensor_01 = self._to_tensor_01(goal_binary_image)
                if use_cached_goal:
                    self._cached_goal_tensor_01 = goal_tensor_01

        # DINO
        if "dino" in requested and self.dino_enable:
            if use_cached_goal and self._cached_goal_embedding is not None:
                goal_embedding = self._cached_goal_embedding
            else:
                goal_embedding = self.extract_embedding(goal_binary_image)
                if use_cached_goal:
                    self._cached_goal_embedding = goal_embedding

            current_embedding = self.extract_embedding(pred_binary_image)
            cosine_sim = self.compute_cosine_similarity(current_embedding, goal_embedding)
            results["cosine_similarity"] = cosine_sim
            results["dino_score"] = self.normalize_similarity(cosine_sim)

        # LPIPS
        if "lpips" in requested and self.lpips_enable:
            lpips_model = self.lpips_model
            if lpips_model is None:
                self.logger.debug("LPIPS requested but unavailable; skipping.")
            else:
                pred_tensor = self._to_tensor_01(pred_binary_image)
                goal_tensor = goal_tensor_01 if goal_tensor_01 is not None else self._to_tensor_01(goal_binary_image)
                with torch.no_grad():
                    lpips_val = lpips_model(pred_tensor, goal_tensor, normalize=True)
                    if isinstance(lpips_val, torch.Tensor):
                        lpips_val = lpips_val.mean()
                    lpips_val = float(lpips_val.detach().cpu().item())
                results["lpips_distance"] = lpips_val
                results["lpips_score"] = float(np.exp(-self.lpips_norm_lambda * lpips_val))

        # DISTS
        if "dists" in requested and self.dists_enable:
            dists_model = self.dists_model
            if dists_model is None:
                self.logger.debug("DISTS requested but unavailable; skipping.")
            else:
                pred_tensor = self._to_tensor_01(pred_binary_image)
                goal_tensor = goal_tensor_01 if goal_tensor_01 is not None else self._to_tensor_01(goal_binary_image)
                with torch.no_grad():
                    dists_val = float(dists_model(pred_tensor, goal_tensor).detach().cpu().item())
                results["dists_distance"] = dists_val
                results["dists_score"] = float(np.exp(-self.dists_norm_lambda * dists_val))

        return results

    def clear_cache(self):
        """Clear cached goal embeddings and tensors."""
        self._cached_goal_embedding = None
        self._cached_goal_tensor_01 = None
        self._cached_goal_mask = None
