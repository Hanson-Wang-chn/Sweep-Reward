"""
Contour metrics for edge-based comparison.
Implements bidirectional Chamfer Distance.
"""

import cv2
import numpy as np
from typing import Dict, Any, Tuple


class ContourMetrics:
    """
    Contour layer metrics: Chamfer Distance.
    Evaluates edge smoothness and topological consistency.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize contour metrics with configuration.

        Args:
            config: Configuration dictionary.
        """
        self.config = config
        contour_cfg = config.get("metrics", {}).get("contour", {}).get("chamfer_distance", {})

        # Canny edge detection parameters
        self.edge_detection = contour_cfg.get("edge_detection", "canny")
        self.canny_low = contour_cfg.get("canny_low_thresh", 50)
        self.canny_high = contour_cfg.get("canny_high_thresh", 150)

        # Normalization parameters
        self.norm_lambda = contour_cfg.get("normalization_lambda", 0.1)
        self.truncate_distance = contour_cfg.get("truncate_distance", 20.0)

    def extract_edges(self, mask: np.ndarray) -> np.ndarray:
        """
        Extract edges from binary mask using Canny edge detection.

        Args:
            mask: Binary mask (H, W), values 0/1 or 0/255.

        Returns:
            Edge map (H, W), binary.
        """
        # Ensure mask is uint8
        if mask.max() <= 1:
            mask_uint8 = (mask * 255).astype(np.uint8)
        else:
            mask_uint8 = mask.astype(np.uint8)

        # Apply Canny edge detection
        edges = cv2.Canny(mask_uint8, self.canny_low, self.canny_high)

        return edges

    def compute_distance_transform(self, edge_map: np.ndarray) -> np.ndarray:
        """
        Compute distance transform from edge map.
        Each pixel contains distance to nearest edge pixel.

        Args:
            edge_map: Binary edge map (H, W).

        Returns:
            Distance transform (H, W), float32.
        """
        # Invert edge map (distance transform expects 0 at edges)
        inverted = (edge_map == 0).astype(np.uint8)

        # Compute distance transform
        dist_transform = cv2.distanceTransform(inverted, cv2.DIST_L2, 5)

        return dist_transform.astype(np.float32)

    def compute_chamfer_distance(
        self,
        pred_edges: np.ndarray,
        goal_edges: np.ndarray,
        pred_dist: np.ndarray = None,
        goal_dist: np.ndarray = None,
    ) -> Tuple[float, float, float]:
        """
        Compute bidirectional Chamfer Distance.

        Args:
            pred_edges: Predicted edge map (H, W).
            goal_edges: Goal edge map (H, W).
            pred_dist: Pre-computed distance transform for pred (optional).
            goal_dist: Pre-computed distance transform for goal (optional).

        Returns:
            Tuple of (chamfer_distance, pred_to_goal, goal_to_pred).
        """
        # Compute distance transforms if not provided
        if pred_dist is None:
            pred_dist = self.compute_distance_transform(pred_edges)
        if goal_dist is None:
            goal_dist = self.compute_distance_transform(goal_edges)

        # Get edge points
        pred_points = np.where(pred_edges > 0)
        goal_points = np.where(goal_edges > 0)

        # Handle empty edge cases
        if len(pred_points[0]) == 0 or len(goal_points[0]) == 0:
            return self.truncate_distance * 2, self.truncate_distance, self.truncate_distance

        # Compute pred to goal distance (using goal distance transform)
        pred_to_goal_dists = goal_dist[pred_points]
        pred_to_goal_dists = np.clip(pred_to_goal_dists, 0, self.truncate_distance)
        pred_to_goal = np.mean(pred_to_goal_dists)

        # Compute goal to pred distance (using pred distance transform)
        goal_to_pred_dists = pred_dist[goal_points]
        goal_to_pred_dists = np.clip(goal_to_pred_dists, 0, self.truncate_distance)
        goal_to_pred = np.mean(goal_to_pred_dists)

        # Bidirectional Chamfer distance
        chamfer = pred_to_goal + goal_to_pred

        return float(chamfer), float(pred_to_goal), float(goal_to_pred)

    def normalize_to_score(self, chamfer_distance: float) -> float:
        """
        Normalize Chamfer distance to similarity score in [0, 1].
        Uses exponential normalization: score = exp(-lambda * distance).

        Args:
            chamfer_distance: Raw Chamfer distance.

        Returns:
            Similarity score in [0, 1].
        """
        score = np.exp(-self.norm_lambda * chamfer_distance)
        return float(score)

    def compute(
        self, pred_mask: np.ndarray, goal_mask: np.ndarray
    ) -> Dict[str, Any]:
        """
        Compute all contour metrics.

        Args:
            pred_mask: Predicted binary mask (H, W), values 0/1.
            goal_mask: Goal binary mask (H, W), values 0/1.

        Returns:
            Dictionary containing:
                - chamfer_distance: Raw bidirectional Chamfer distance
                - chamfer_score: Normalized similarity score [0, 1]
                - pred_to_goal: One-directional distance
                - goal_to_pred: One-directional distance
        """
        # Extract edges
        pred_edges = self.extract_edges(pred_mask)
        goal_edges = self.extract_edges(goal_mask)

        # Compute distance transforms
        pred_dist = self.compute_distance_transform(pred_edges)
        goal_dist = self.compute_distance_transform(goal_edges)

        # Compute Chamfer distance
        chamfer, pred_to_goal, goal_to_pred = self.compute_chamfer_distance(
            pred_edges, goal_edges, pred_dist, goal_dist
        )

        # Normalize to score
        score = self.normalize_to_score(chamfer)

        return {
            "chamfer_distance": chamfer,
            "chamfer_score": score,
            "pred_to_goal": pred_to_goal,
            "goal_to_pred": goal_to_pred,
            "pred_edges": pred_edges,
            "goal_edges": goal_edges,
        }
