"""
Main Evaluator module implementing the ensemble evaluation logic.
Combines geometric, contour, semantic, and perceptual metrics.
"""

import logging
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional

import numpy as np

from .preprocessor import Preprocessor
from .metrics.geometric import GeometricMetrics
from .metrics.contour import ContourMetrics
from .metrics.semantic import SemanticMetrics
from .vlm_client import VLMClient


logger = logging.getLogger(__name__)


class Evaluator:
    """
    Multi-modal Ensemble Evaluator for Sweep-to-Shapes task.
    Implements weighted gating mechanism for efficient evaluation.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize evaluator with all sub-modules.
        DINOv2 model is loaded once and kept in memory.

        Args:
            config: Configuration dictionary.
        """
        self.config = config

        # Initialize sub-modules
        self.preprocessor = Preprocessor(config)
        self.geometric_metrics = GeometricMetrics(config)
        self.contour_metrics = ContourMetrics(config)
        self.semantic_metrics = SemanticMetrics(config)
        self.vlm_client = VLMClient(config)

        # Ensemble configuration
        ensemble_cfg = config.get("ensemble", {})

        # Gating configuration
        gating_cfg = ensemble_cfg.get("gating", {})
        self.gating_enable = gating_cfg.get("enable", True)
        self.gating_threshold = gating_cfg.get("threshold", 0.4)

        # Weights
        weights_cfg = ensemble_cfg.get("weights", {})
        self.weight_geometric = weights_cfg.get("geometric", 0.35)
        self.weight_contour = weights_cfg.get("contour", 0.25)
        self.weight_semantic = weights_cfg.get("semantic", 0.20)
        self.weight_perceptual = weights_cfg.get("perceptual", 0.20)

        # Debug configuration
        debug_cfg = config.get("debug", {})
        self.enable_visualization = debug_cfg.get("enable_visualization", False)
        self.vis_output_dir = debug_cfg.get("vis_output_dir", "./logs/vis_eval")
        self.save_metrics_to_json = debug_cfg.get("save_metrics_to_json", True)

        # System configuration
        system_cfg = config.get("system", {})
        self.image_size = tuple(system_cfg.get("image_size", [224, 224]))

        # Cached goal mask
        self._goal_mask: Optional[np.ndarray] = None

        # Ensure output directory exists
        if self.enable_visualization or self.save_metrics_to_json:
            os.makedirs(self.vis_output_dir, exist_ok=True)

    def set_goal(self, goal_mask: np.ndarray) -> None:
        """
        Set and cache goal mask. Computes goal embeddings.
        Should be called once at task start.

        Args:
            goal_mask: Goal mask image (can be RGB or grayscale).
        """
        # Process goal mask
        self._goal_mask = self.preprocessor.process_goal_mask(goal_mask)

        # Cache DINOv2 embedding for goal
        self.semantic_metrics.set_goal(self._goal_mask)

        logger.info("Goal mask set and embeddings cached.")

    def evaluate(
        self,
        current_image: np.ndarray,
        goal_mask: np.ndarray = None,
        save_debug: bool = None,
    ) -> Dict[str, Any]:
        """
        Evaluate current state against goal.
        Implements weighted gating mechanism.

        Args:
            current_image: Current RGB image (H, W, 3), uint8.
            goal_mask: Goal mask (optional if already set via set_goal).
            save_debug: Whether to save debug outputs (overrides config).

        Returns:
            Dictionary containing:
                - total_score: Final ensemble score [0, 1]
                - details: Dict with individual metric scores
                - gating_passed: Whether gating threshold was passed
        """
        # Handle goal mask
        if goal_mask is not None:
            self.set_goal(goal_mask)
        elif self._goal_mask is None:
            raise ValueError("Goal mask must be set before evaluation.")

        goal = self._goal_mask

        # Resize current image if needed
        current_image = self.preprocessor.resize_image(current_image)

        # Preprocess current image to get prediction mask
        pred_mask, pred_soft_mask = self.preprocessor.process(current_image)

        # === Stage 1: Compute basic metrics (always computed) ===
        geometric_results = self.geometric_metrics.compute(pred_mask, goal)
        contour_results = self.contour_metrics.compute(pred_mask, goal)

        # Use F1 as the geometric score (could also use elastic_iou)
        s_geo = geometric_results["f1"]
        s_contour = contour_results["chamfer_score"]

        # Initialize results
        result = {
            "total_score": 0.0,
            "details": {
                "iou": geometric_results["elastic_iou"],
                "f1": s_geo,
                "chamfer": s_contour,
                "dino": None,
                "vlm": None,
            },
            "gating_passed": False,
            "raw_metrics": {
                "geometric": geometric_results,
                "contour": contour_results,
            },
        }

        # === Stage 2: Gating check ===
        if self.gating_enable and s_geo < self.gating_threshold:
            # Failed gating - return early with geometric score only
            logger.info(f"Gating failed: F1={s_geo:.4f} < threshold={self.gating_threshold}")
            result["total_score"] = s_geo
            result["gating_passed"] = False

            # Save debug if requested
            if save_debug or (save_debug is None and self.save_metrics_to_json):
                self._save_debug_output(result, current_image, pred_mask)

            return result

        # === Stage 3: Full evaluation (passed gating) ===
        result["gating_passed"] = True

        # Compute semantic score (DINOv2)
        semantic_results = self.semantic_metrics.compute(
            current_image, use_cached_goal=True
        )
        s_semantic = semantic_results["dino_score"]
        result["details"]["dino"] = s_semantic
        result["raw_metrics"]["semantic"] = semantic_results

        # Compute perceptual score (VLM)
        try:
            vlm_results = self.vlm_client.compute(current_image, goal)
            s_perceptual = vlm_results["vlm_score"]
            result["details"]["vlm"] = s_perceptual
            result["raw_metrics"]["vlm"] = vlm_results
        except Exception as e:
            logger.warning(f"VLM scoring failed, using fallback: {e}")
            s_perceptual = (s_geo + s_contour + s_semantic) / 3
            result["details"]["vlm"] = s_perceptual
            result["raw_metrics"]["vlm"] = {"vlm_score": s_perceptual, "reasoning": "fallback"}

        # === Stage 4: Weighted ensemble ===
        total_score = (
            self.weight_geometric * s_geo
            + self.weight_contour * s_contour
            + self.weight_semantic * s_semantic
            + self.weight_perceptual * s_perceptual
        )

        result["total_score"] = float(total_score)

        logger.info(
            f"Evaluation complete: total={total_score:.4f}, "
            f"geo={s_geo:.4f}, contour={s_contour:.4f}, "
            f"dino={s_semantic:.4f}, vlm={s_perceptual:.4f}"
        )

        # Save debug if requested
        if save_debug or (save_debug is None and self.save_metrics_to_json):
            self._save_debug_output(result, current_image, pred_mask)

        return result

    def _save_debug_output(
        self,
        result: Dict[str, Any],
        current_image: np.ndarray,
        pred_mask: np.ndarray,
    ) -> None:
        """Save debug outputs to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        if self.save_metrics_to_json:
            # Save metrics JSON
            json_path = os.path.join(self.vis_output_dir, f"metrics_{timestamp}.json")
            # Convert numpy types for JSON serialization
            result_serializable = self._make_serializable(result)
            with open(json_path, "w") as f:
                json.dump(result_serializable, f, indent=2)

        if self.enable_visualization:
            import cv2

            # Save current image
            img_path = os.path.join(self.vis_output_dir, f"current_{timestamp}.png")
            cv2.imwrite(img_path, cv2.cvtColor(current_image, cv2.COLOR_RGB2BGR))

            # Save prediction mask
            mask_path = os.path.join(self.vis_output_dir, f"pred_mask_{timestamp}.png")
            cv2.imwrite(mask_path, (pred_mask * 255).astype(np.uint8))

            # Save goal mask
            if self._goal_mask is not None:
                goal_path = os.path.join(self.vis_output_dir, f"goal_mask_{timestamp}.png")
                cv2.imwrite(goal_path, (self._goal_mask * 255).astype(np.uint8))
                
                # Save rendered goal image (DINO input)
                goal_rendered = self.semantic_metrics.render_goal_image(self._goal_mask)
                goal_rendered_path = os.path.join(self.vis_output_dir, f"goal_rendered_{timestamp}.png")
                cv2.imwrite(goal_rendered_path, cv2.cvtColor(goal_rendered, cv2.COLOR_RGB2BGR))
                logger.info(f"Saved rendered goal image to: {goal_rendered_path}")

    def _make_serializable(self, obj: Any) -> Any:
        """Convert numpy types to Python native types for JSON serialization."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj

    def evaluate_batch(
        self,
        images: list,
        goal_mask: np.ndarray = None,
    ) -> list:
        """
        Evaluate a batch of images.

        Args:
            images: List of current RGB images.
            goal_mask: Goal mask (optional if already set).

        Returns:
            List of evaluation results.
        """
        if goal_mask is not None:
            self.set_goal(goal_mask)

        results = []
        for i, image in enumerate(images):
            logger.info(f"Evaluating image {i + 1}/{len(images)}")
            result = self.evaluate(image, save_debug=False)
            results.append(result)

        return results
