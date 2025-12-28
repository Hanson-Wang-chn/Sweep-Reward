"""
Geometric metrics for mask comparison.
Includes Elastic IoU, F1-Score, and Sinkhorn Divergence calculations.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional, Set


class GeometricMetrics:
    """
    Geometric layer metrics: Elastic IoU and F1-Score.
    Handles rigid transformations to compensate for execution errors.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize geometric metrics with configuration.

        Args:
            config: Configuration dictionary.
        """
        self.config = config
        metrics_cfg = config.get("metrics", {}).get("geometric", {})

        # F1-Score parameters
        f1_cfg = metrics_cfg.get("f1_score", {})
        self.epsilon = f1_cfg.get("epsilon", 1e-6)

        # Elastic IoU parameters
        elastic_cfg = metrics_cfg.get("elastic_iou", {})
        self.elastic_enable = elastic_cfg.get("enable", True)
        self.method = elastic_cfg.get("method", "grid_search")
        self.downsample_factor = elastic_cfg.get("downsample_factor", 2)
        self.max_translation = elastic_cfg.get("max_translation", 10)
        self.max_rotation = elastic_cfg.get("max_rotation", 10)
        self.step_translation = elastic_cfg.get("step_translation", 2)
        self.step_rotation = elastic_cfg.get("step_rotation", 2)

        # Sinkhorn divergence parameters
        sinkhorn_cfg = metrics_cfg.get("sinkhorn", {})
        self.sinkhorn_enable = sinkhorn_cfg.get("enable", True)
        self.sinkhorn_downsample = sinkhorn_cfg.get("downsample", 4)
        self.sinkhorn_epsilon = sinkhorn_cfg.get("epsilon", 0.01)
        self.sinkhorn_max_iters = sinkhorn_cfg.get("max_iters", 50)
        self.sinkhorn_norm_lambda = sinkhorn_cfg.get("normalization_lambda", 0.05)
        self.sinkhorn_clip = sinkhorn_cfg.get("clip_divergence", 5.0)

        # Device for heavy torch computations
        system_cfg = config.get("system", {})
        device_str = system_cfg.get("device", "cuda")
        self.device = torch.device(device_str if torch.cuda.is_available() else "cpu")

        # Cache coordinate grids for Sinkhorn
        self._coord_cache: Dict[Tuple[int, int], torch.Tensor] = {}

    def compute_iou(
        self, pred_mask: np.ndarray, goal_mask: np.ndarray
    ) -> float:
        """
        Compute standard IoU between two binary masks.

        Args:
            pred_mask: Predicted binary mask (H, W), values 0/1.
            goal_mask: Goal binary mask (H, W), values 0/1.

        Returns:
            IoU score in [0, 1].
        """
        intersection = np.logical_and(pred_mask, goal_mask).sum()
        union = np.logical_or(pred_mask, goal_mask).sum()

        if union == 0:
            return 0.0

        return float(intersection / (union + self.epsilon))

    def compute_f1(
        self, pred_mask: np.ndarray, goal_mask: np.ndarray
    ) -> Tuple[float, float, float]:
        """
        Compute F1-Score, Precision, and Recall.

        Args:
            pred_mask: Predicted binary mask (H, W), values 0/1.
            goal_mask: Goal binary mask (H, W), values 0/1.

        Returns:
            Tuple of (F1, Precision, Recall).
        """
        intersection = np.logical_and(pred_mask, goal_mask).sum()
        pred_sum = pred_mask.sum()
        goal_sum = goal_mask.sum()

        precision = intersection / (pred_sum + self.epsilon)
        recall = intersection / (goal_sum + self.epsilon)

        f1 = 2 * precision * recall / (precision + recall + self.epsilon)

        return float(f1), float(precision), float(recall)

    def _apply_transform(
        self,
        mask: np.ndarray,
        theta: float,
        tx: float,
        ty: float,
    ) -> np.ndarray:
        """
        Apply affine transformation (rotation + translation) to mask.

        Args:
            mask: Binary mask (H, W).
            theta: Rotation angle in degrees.
            tx: Translation in x direction (pixels).
            ty: Translation in y direction (pixels).

        Returns:
            Transformed mask.
        """
        h, w = mask.shape
        center = (w / 2, h / 2)

        # Create rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, theta, 1.0)

        # Add translation
        rotation_matrix[0, 2] += tx
        rotation_matrix[1, 2] += ty

        # Apply transformation
        transformed = cv2.warpAffine(
            mask.astype(np.float32),
            rotation_matrix,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        return (transformed > 0.5).astype(np.uint8)

    def compute_elastic_iou(
        self, pred_mask: np.ndarray, goal_mask: np.ndarray
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute Elastic IoU with grid search over transformations.
        Finds the best rigid transformation to maximize IoU.

        Args:
            pred_mask: Predicted binary mask (H, W), values 0/1.
            goal_mask: Goal binary mask (H, W), values 0/1.

        Returns:
            Tuple of (best_iou, best_params) where best_params contains
            theta, tx, ty of the optimal transformation.
        """
        if not self.elastic_enable:
            iou = self.compute_iou(pred_mask, goal_mask)
            return iou, {"theta": 0.0, "tx": 0.0, "ty": 0.0}

        # Downsample for faster search
        if self.downsample_factor > 1:
            h, w = pred_mask.shape
            new_h, new_w = h // self.downsample_factor, w // self.downsample_factor
            pred_ds = cv2.resize(
                pred_mask.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST
            )
            goal_ds = cv2.resize(
                goal_mask.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST
            )
            scale = self.downsample_factor
        else:
            pred_ds = pred_mask.astype(np.uint8)
            goal_ds = goal_mask.astype(np.uint8)
            scale = 1

        best_iou = 0.0
        best_params = {"theta": 0.0, "tx": 0.0, "ty": 0.0}

        # Coarse search
        coarse_step_t = self.step_translation * 2
        coarse_step_r = self.step_rotation * 2

        for theta in np.arange(
            -self.max_rotation, self.max_rotation + 1, coarse_step_r
        ):
            for tx in np.arange(
                -self.max_translation / scale,
                self.max_translation / scale + 1,
                coarse_step_t / scale,
            ):
                for ty in np.arange(
                    -self.max_translation / scale,
                    self.max_translation / scale + 1,
                    coarse_step_t / scale,
                ):
                    transformed = self._apply_transform(pred_ds, theta, tx, ty)
                    iou = self.compute_iou(transformed, goal_ds)

                    if iou > best_iou:
                        best_iou = iou
                        best_params = {
                            "theta": theta,
                            "tx": tx * scale,
                            "ty": ty * scale,
                        }

        # Fine search around best coarse result
        fine_range_t = coarse_step_t
        fine_range_r = coarse_step_r

        for theta in np.arange(
            best_params["theta"] - fine_range_r,
            best_params["theta"] + fine_range_r + 0.1,
            self.step_rotation,
        ):
            for tx in np.arange(
                (best_params["tx"] - fine_range_t) / scale,
                (best_params["tx"] + fine_range_t) / scale + 0.1,
                self.step_translation / scale,
            ):
                for ty in np.arange(
                    (best_params["ty"] - fine_range_t) / scale,
                    (best_params["ty"] + fine_range_t) / scale + 0.1,
                    self.step_translation / scale,
                ):
                    transformed = self._apply_transform(pred_ds, theta, tx, ty)
                    iou = self.compute_iou(transformed, goal_ds)

                    if iou > best_iou:
                        best_iou = iou
                        best_params = {
                            "theta": theta,
                            "tx": tx * scale,
                            "ty": ty * scale,
                        }

        # Compute final IoU on full resolution
        if self.downsample_factor > 1:
            transformed_full = self._apply_transform(
                pred_mask.astype(np.uint8),
                best_params["theta"],
                best_params["tx"],
                best_params["ty"],
            )
            best_iou = self.compute_iou(transformed_full, goal_mask)

        return best_iou, best_params

    def compute(
        self,
        pred_mask: np.ndarray,
        goal_mask: np.ndarray,
        metrics_to_compute: Optional[Set[str]] = None,
    ) -> Dict[str, Any]:
        """
        Compute selected geometric metrics.

        Args:
            pred_mask: Predicted binary mask (H, W), values 0/1.
            goal_mask: Goal binary mask (H, W), values 0/1.
            metrics_to_compute: Subset of metrics to run. Defaults to all.

        Returns:
            Dictionary containing computed metric values (None if skipped).
        """
        requested = metrics_to_compute or {"iou", "elastic_iou", "f1", "sinkhorn"}
        results: Dict[str, Any] = {
            "iou": None,
            "elastic_iou": None,
            "f1": None,
            "precision": None,
            "recall": None,
            "transform_params": None,
            "sinkhorn_divergence": None,
            "sinkhorn_score": None,
        }

        if "iou" in requested:
            results["iou"] = self.compute_iou(pred_mask, goal_mask)

        if "elastic_iou" in requested:
            elastic_iou, transform_params = self.compute_elastic_iou(pred_mask, goal_mask)
            results["elastic_iou"] = elastic_iou
            results["transform_params"] = transform_params

        if "f1" in requested:
            f1, precision, recall = self.compute_f1(pred_mask, goal_mask)
            results["f1"] = f1
            results["precision"] = precision
            results["recall"] = recall

        if "sinkhorn" in requested and self.sinkhorn_enable:
            sinkhorn_div, sinkhorn_score = self.compute_sinkhorn(pred_mask, goal_mask)
            results["sinkhorn_divergence"] = sinkhorn_div
            results["sinkhorn_score"] = sinkhorn_score

        return results

    def compute_sinkhorn(self, pred_mask: np.ndarray, goal_mask: np.ndarray) -> Tuple[float, float]:
        """
        Compute Sinkhorn Divergence between two binary masks.
        Downsamples masks for efficiency, converts them to probability
        distributions, and applies entropic regularized OT.
        """
        pred_tensor, grid_shape = self._prepare_distribution(pred_mask)
        goal_tensor, _ = self._prepare_distribution(goal_mask)

        pred_mass = float(pred_tensor.sum().item())
        goal_mass = float(goal_tensor.sum().item())

        if pred_mass < 1e-6 and goal_mass < 1e-6:
            return 0.0, 1.0
        if pred_mass < 1e-6 or goal_mass < 1e-6:
            return float(self.sinkhorn_clip), 0.0

        coords = self._get_grid_coords(grid_shape)

        with torch.no_grad():
            ab = self._sinkhorn_cost(pred_tensor, goal_tensor, coords)
            aa = self._sinkhorn_cost(pred_tensor, pred_tensor, coords)
            bb = self._sinkhorn_cost(goal_tensor, goal_tensor, coords)

            divergence = torch.relu(ab - 0.5 * (aa + bb))
            divergence = torch.clamp(divergence, max=self.sinkhorn_clip)
            score = torch.exp(-self.sinkhorn_norm_lambda * divergence)

        return float(divergence.item()), float(score.item())

    def _prepare_distribution(self, mask: np.ndarray) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Downsample and normalize mask to a flat probability distribution.
        """
        tensor = torch.from_numpy(mask.astype(np.float32)).to(self.device)
        if self.sinkhorn_downsample > 1:
            kernel = self.sinkhorn_downsample
            tensor = F.avg_pool2d(tensor.unsqueeze(0).unsqueeze(0), kernel, stride=kernel)
            tensor = tensor.squeeze(0).squeeze(0)

        shape = tensor.shape
        total_mass = tensor.sum()
        if total_mass > 0:
            tensor = tensor / total_mass
        return tensor.flatten(), (shape[0], shape[1])

    def _get_grid_coords(self, shape: Tuple[int, int]) -> torch.Tensor:
        """
        Create (H*W, 2) coordinate grid for a given mask shape.
        Cached to avoid rebuilds.
        """
        if shape in self._coord_cache:
            return self._coord_cache[shape]

        h, w = shape
        ys, xs = torch.meshgrid(
            torch.arange(h, device=self.device, dtype=torch.float32),
            torch.arange(w, device=self.device, dtype=torch.float32),
            indexing="ij",
        )
        coords = torch.stack([ys.flatten(), xs.flatten()], dim=1)
        self._coord_cache[shape] = coords
        return coords

    def _sinkhorn_cost(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        coords: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute entropic OT cost using Sinkhorn-Knopp iterations.
        """
        C = torch.cdist(coords, coords, p=2)
        K = torch.exp(-C / self.sinkhorn_epsilon)

        # Initialize scaling vectors
        u = torch.ones_like(a) / max(a.numel(), 1)
        v = torch.ones_like(b) / max(b.numel(), 1)

        for _ in range(self.sinkhorn_max_iters):
            Kv = torch.matmul(K, v)
            u = a / (Kv + 1e-8)
            Ku = torch.matmul(K.transpose(0, 1), u)
            v = b / (Ku + 1e-8)

        transport = u.unsqueeze(1) * K * v.unsqueeze(0)
        cost = torch.sum(transport * C)
        return cost
