"""
Geometric metrics for mask comparison.
Includes Elastic IoU and F1-Score calculations.
"""

import cv2
import numpy as np
from typing import Dict, Any, Tuple, Optional


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
        self, pred_mask: np.ndarray, goal_mask: np.ndarray
    ) -> Dict[str, Any]:
        """
        Compute all geometric metrics.

        Args:
            pred_mask: Predicted binary mask (H, W), values 0/1.
            goal_mask: Goal binary mask (H, W), values 0/1.

        Returns:
            Dictionary containing:
                - iou: Standard IoU
                - elastic_iou: Elastic IoU (if enabled)
                - f1: F1-Score
                - precision: Precision
                - recall: Recall
                - transform_params: Best transformation parameters
        """
        # Standard IoU
        standard_iou = self.compute_iou(pred_mask, goal_mask)

        # Elastic IoU
        elastic_iou, transform_params = self.compute_elastic_iou(pred_mask, goal_mask)

        # F1-Score
        f1, precision, recall = self.compute_f1(pred_mask, goal_mask)

        return {
            "iou": standard_iou,
            "elastic_iou": elastic_iou,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "transform_params": transform_params,
        }
