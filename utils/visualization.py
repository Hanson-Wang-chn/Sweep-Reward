"""
Visualization utilities for debugging and analysis.
Provides tools to visualize masks, edges, and heatmaps.
"""

import os
from typing import Dict, Any, Optional, Tuple

import cv2
import numpy as np


class Visualizer:
    """
    Visualization utilities for evaluation debugging.
    """

    def __init__(self, output_dir: str = "./logs/vis_eval"):
        """
        Initialize visualizer.

        Args:
            output_dir: Directory to save visualization outputs.
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def visualize_mask_comparison(
        self,
        pred_mask: np.ndarray,
        goal_mask: np.ndarray,
        current_image: np.ndarray = None,
        save_path: str = None,
    ) -> np.ndarray:
        """
        Create visualization comparing prediction and goal masks.

        Args:
            pred_mask: Predicted binary mask (H, W).
            goal_mask: Goal binary mask (H, W).
            current_image: Optional current RGB image for overlay.
            save_path: Optional path to save visualization.

        Returns:
            Visualization image (H, W*3, 3) or (H, W*4, 3) if current_image provided.
        """
        h, w = pred_mask.shape

        # Create colored masks
        pred_colored = np.zeros((h, w, 3), dtype=np.uint8)
        pred_colored[pred_mask > 0] = [0, 255, 0]  # Green for prediction

        goal_colored = np.zeros((h, w, 3), dtype=np.uint8)
        goal_colored[goal_mask > 0] = [0, 0, 255]  # Red for goal

        # Create overlap visualization
        overlap = np.zeros((h, w, 3), dtype=np.uint8)
        intersection = np.logical_and(pred_mask > 0, goal_mask > 0)
        pred_only = np.logical_and(pred_mask > 0, goal_mask == 0)
        goal_only = np.logical_and(pred_mask == 0, goal_mask > 0)

        overlap[intersection] = [255, 255, 0]  # Yellow for intersection
        overlap[pred_only] = [0, 255, 0]  # Green for pred only
        overlap[goal_only] = [0, 0, 255]  # Red for goal only

        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(pred_colored, "Prediction", (5, 20), font, 0.5, (255, 255, 255), 1)
        cv2.putText(goal_colored, "Goal", (5, 20), font, 0.5, (255, 255, 255), 1)
        cv2.putText(overlap, "Overlap", (5, 20), font, 0.5, (255, 255, 255), 1)

        # Concatenate
        if current_image is not None:
            current_labeled = current_image.copy()
            if len(current_labeled.shape) == 2:
                current_labeled = cv2.cvtColor(current_labeled, cv2.COLOR_GRAY2RGB)
            cv2.putText(current_labeled, "Current", (5, 20), font, 0.5, (255, 255, 255), 1)
            vis = np.concatenate([current_labeled, pred_colored, goal_colored, overlap], axis=1)
        else:
            vis = np.concatenate([pred_colored, goal_colored, overlap], axis=1)

        if save_path:
            cv2.imwrite(save_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

        return vis

    def visualize_edges(
        self,
        pred_edges: np.ndarray,
        goal_edges: np.ndarray,
        save_path: str = None,
    ) -> np.ndarray:
        """
        Create visualization comparing edge maps.

        Args:
            pred_edges: Predicted edge map (H, W).
            goal_edges: Goal edge map (H, W).
            save_path: Optional path to save visualization.

        Returns:
            Visualization image (H, W*3, 3).
        """
        h, w = pred_edges.shape

        # Create colored edge visualizations
        pred_colored = np.zeros((h, w, 3), dtype=np.uint8)
        pred_colored[pred_edges > 0] = [0, 255, 0]

        goal_colored = np.zeros((h, w, 3), dtype=np.uint8)
        goal_colored[goal_edges > 0] = [0, 0, 255]

        # Overlay both edges
        combined = np.zeros((h, w, 3), dtype=np.uint8)
        combined[pred_edges > 0] = [0, 255, 0]
        combined[goal_edges > 0] = [0, 0, 255]
        # Where both edges exist
        both = np.logical_and(pred_edges > 0, goal_edges > 0)
        combined[both] = [255, 255, 0]

        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(pred_colored, "Pred Edges", (5, 20), font, 0.5, (255, 255, 255), 1)
        cv2.putText(goal_colored, "Goal Edges", (5, 20), font, 0.5, (255, 255, 255), 1)
        cv2.putText(combined, "Combined", (5, 20), font, 0.5, (255, 255, 255), 1)

        vis = np.concatenate([pred_colored, goal_colored, combined], axis=1)

        if save_path:
            cv2.imwrite(save_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

        return vis

    def visualize_distance_heatmap(
        self,
        distance_map: np.ndarray,
        edges: np.ndarray = None,
        save_path: str = None,
    ) -> np.ndarray:
        """
        Create heatmap visualization of distance transform.

        Args:
            distance_map: Distance transform (H, W), float.
            edges: Optional edge map to overlay.
            save_path: Optional path to save visualization.

        Returns:
            Heatmap visualization (H, W, 3).
        """
        # Normalize distance map
        dist_norm = distance_map.copy()
        if dist_norm.max() > 0:
            dist_norm = (dist_norm / dist_norm.max() * 255).astype(np.uint8)
        else:
            dist_norm = dist_norm.astype(np.uint8)

        # Apply colormap
        heatmap = cv2.applyColorMap(dist_norm, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # Overlay edges if provided
        if edges is not None:
            heatmap[edges > 0] = [255, 255, 255]

        if save_path:
            cv2.imwrite(save_path, cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR))

        return heatmap

    def visualize_evaluation_result(
        self,
        current_image: np.ndarray,
        pred_mask: np.ndarray,
        goal_mask: np.ndarray,
        result: Dict[str, Any],
        save_path: str = None,
    ) -> np.ndarray:
        """
        Create comprehensive evaluation result visualization.

        Args:
            current_image: Current RGB image (H, W, 3).
            pred_mask: Predicted binary mask (H, W).
            goal_mask: Goal binary mask (H, W).
            result: Evaluation result dictionary.
            save_path: Optional path to save visualization.

        Returns:
            Visualization image.
        """
        h, w = pred_mask.shape

        # Create mask comparison
        mask_vis = self.visualize_mask_comparison(pred_mask, goal_mask, current_image)

        # Create score panel
        score_panel = np.zeros((100, mask_vis.shape[1], 3), dtype=np.uint8)
        score_panel[:] = [40, 40, 40]  # Dark gray background

        font = cv2.FONT_HERSHEY_SIMPLEX
        details = result.get("details", {})

        # Format scores
        total = result.get("total_score", 0)
        iou = details.get("iou", 0)
        f1 = details.get("f1", 0)
        chamfer = details.get("chamfer", 0)
        dino = details.get("dino", "N/A")
        vlm = details.get("vlm", "N/A")

        # Draw scores
        y_offset = 25
        cv2.putText(
            score_panel,
            f"Total: {total:.4f}" if isinstance(total, float) else f"Total: {total}",
            (10, y_offset),
            font,
            0.6,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            score_panel,
            f"IoU: {iou:.4f}" if isinstance(iou, float) else f"IoU: {iou}",
            (200, y_offset),
            font,
            0.5,
            (200, 200, 200),
            1,
        )
        cv2.putText(
            score_panel,
            f"F1: {f1:.4f}" if isinstance(f1, float) else f"F1: {f1}",
            (350, y_offset),
            font,
            0.5,
            (200, 200, 200),
            1,
        )

        y_offset = 55
        cv2.putText(
            score_panel,
            f"Chamfer: {chamfer:.4f}" if isinstance(chamfer, float) else f"Chamfer: {chamfer}",
            (10, y_offset),
            font,
            0.5,
            (200, 200, 200),
            1,
        )
        cv2.putText(
            score_panel,
            f"DINO: {dino:.4f}" if isinstance(dino, float) else f"DINO: {dino}",
            (200, y_offset),
            font,
            0.5,
            (200, 200, 200),
            1,
        )
        cv2.putText(
            score_panel,
            f"VLM: {vlm:.4f}" if isinstance(vlm, float) else f"VLM: {vlm}",
            (350, y_offset),
            font,
            0.5,
            (200, 200, 200),
            1,
        )

        # Gating status
        gating = result.get("gating_passed", False)
        gating_text = "Gating: PASSED" if gating else "Gating: FAILED"
        gating_color = (0, 255, 0) if gating else (0, 0, 255)
        cv2.putText(score_panel, gating_text, (10, 85), font, 0.5, gating_color, 1)

        # Combine
        vis = np.concatenate([mask_vis, score_panel], axis=0)

        if save_path:
            cv2.imwrite(save_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

        return vis

    def create_animation_frame(
        self,
        frame_idx: int,
        current_image: np.ndarray,
        pred_mask: np.ndarray,
        goal_mask: np.ndarray,
        score: float,
    ) -> np.ndarray:
        """
        Create a single frame for animation/video export.

        Args:
            frame_idx: Frame index.
            current_image: Current RGB image.
            pred_mask: Predicted mask.
            goal_mask: Goal mask.
            score: Current score.

        Returns:
            Frame image.
        """
        h, w = pred_mask.shape

        # Create overlay on current image
        overlay = current_image.copy()

        # Add semi-transparent goal overlay
        goal_overlay = np.zeros_like(overlay)
        goal_overlay[goal_mask > 0] = [255, 0, 0]  # Red
        overlay = cv2.addWeighted(overlay, 0.7, goal_overlay, 0.3, 0)

        # Add prediction outline
        pred_edges = cv2.Canny((pred_mask * 255).astype(np.uint8), 50, 150)
        overlay[pred_edges > 0] = [0, 255, 0]  # Green outline

        # Add frame info
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(overlay, f"Frame: {frame_idx}", (10, 20), font, 0.5, (255, 255, 255), 1)
        cv2.putText(overlay, f"Score: {score:.4f}", (10, 40), font, 0.5, (255, 255, 255), 1)

        return overlay
