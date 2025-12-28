"""
Main Evaluator module implementing the ensemble evaluation logic.
Combines geometric, contour, semantic, and perceptual metrics.
"""

import logging
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional, Set

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
        self.gating_metric = gating_cfg.get("metric", "f1")

        # Weights
        weights_cfg = ensemble_cfg.get("weights", {})
        self.metric_weights = self._load_metric_weights(weights_cfg)
        self._refresh_weight_cache()

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

    def _load_metric_weights(self, weights_cfg: Dict[str, Any]) -> Dict[str, float]:
        """
        Flatten nested weight configuration into a metric-level mapping.
        """
        defaults = {
            "iou": 0.05,
            "elastic_iou": 0.10,
            "f1": 0.15,
            "sinkhorn": 0.05,
            "chamfer": 0.25,
            "dino": 0.12,
            "lpips": 0.05,
            "dists": 0.03,
            "vlm": 0.20,
        }

        flat = defaults.copy()
        for key, value in weights_cfg.items():
            if isinstance(value, dict):
                for sub_key, sub_val in value.items():
                    flat[sub_key] = float(sub_val)
            else:
                flat[key] = float(value)
        return flat

    def _refresh_weight_cache(self) -> None:
        """Recompute active metric sets and weight sums."""
        self.active_geometric_metrics = {
            m for m in ["iou", "elastic_iou", "f1", "sinkhorn"]
            if self.metric_weights.get(m, 0.0) > 0
        }
        self.active_contour_metrics = (
            {"chamfer"} if self.metric_weights.get("chamfer", 0.0) > 0 else set()
        )
        self.semantic_active_metrics = {
            m for m in ["dino", "lpips", "dists"]
            if self.metric_weights.get(m, 0.0) > 0
        }
        self.perceptual_active_metrics = (
            {"vlm"} if self.metric_weights.get("vlm", 0.0) > 0 else set()
        )

    def _metric_enabled(self, metric: str, allowed_metrics: Optional[set]) -> bool:
        """Check if a metric is both weighted and allowed by runtime flags."""
        if allowed_metrics is not None and metric not in allowed_metrics:
            return False
        return self.metric_weights.get(metric, 0.0) > 0

    def _semantic_metrics_requested(
        self,
        allowed_metrics: Optional[set],
        skip_semantic: bool,
    ) -> Set[str]:
        """Determine which semantic metrics should be computed."""
        if skip_semantic:
            return set()
        metrics = self.semantic_active_metrics.copy()
        if allowed_metrics is not None:
            metrics &= allowed_metrics
        return metrics

    def aggregate_scores(
        self,
        score_map: Dict[str, Optional[float]],
        allowed_metrics: Optional[Set[str]] = None,
    ) -> float:
        """
        Compute weighted score using available metrics, renormalizing if some are skipped.
        """
        active_items = []
        for name, score in score_map.items():
            if score is None:
                continue
            if allowed_metrics is not None and name not in allowed_metrics:
                continue
            weight = self.metric_weights.get(name, 0.0)
            if weight <= 0:
                continue
            active_items.append((weight, score))

        if not active_items:
            return 0.0

        weight_sum = sum(w for w, _ in active_items)
        total = sum(w * s for w, s in active_items) / weight_sum
        return float(total)

    def _build_details(self, score_map: Dict[str, Optional[float]]) -> Dict[str, Optional[float]]:
        """Return ordered details mapping for downstream logging/visualization."""
        keys = ["iou", "elastic_iou", "f1", "sinkhorn", "chamfer", "dino", "lpips", "dists", "vlm"]
        return {k: score_map.get(k) for k in keys}

    def set_goal(self, goal_mask: np.ndarray, semantic_metrics_to_cache: Optional[Set[str]] = None) -> None:
        """
        Set and cache goal mask. Computes goal embeddings.
        Should be called once at task start.

        Args:
            goal_mask: Goal mask image (can be RGB or grayscale).
        """
        # Process goal mask
        self._goal_mask = self.preprocessor.process_goal_mask(goal_mask)

        metrics = semantic_metrics_to_cache or self.semantic_active_metrics
        if metrics:
            self.semantic_metrics.set_goal(self._goal_mask, metrics_to_cache=metrics)

        logger.info("Goal mask set and embeddings cached.")

    def evaluate(
        self,
        current_image: np.ndarray,
        goal_mask: np.ndarray = None,
        save_debug: bool = None,
        image_index: int = 0,
        allowed_metrics: Optional[Set[str]] = None,
        skip_vlm: bool = False,
        skip_semantic: bool = False,
    ) -> Dict[str, Any]:
        """
        Evaluate current state against goal.
        Implements weighted gating mechanism.

        Args:
            current_image: Current RGB image (H, W, 3), uint8.
            goal_mask: Goal mask (optional if already set via set_goal).
            save_debug: Whether to save debug outputs (overrides config).
            image_index: Index for distinguishing multiple evaluations when saving VLM images.
            allowed_metrics: Subset of metrics to evaluate (None means all weighted metrics).
            skip_vlm: Force-skip VLM evaluation regardless of weight.
            skip_semantic: Force-skip semantic metrics (DINO/LPIPS/DISTS).

        Returns:
            Dictionary containing:
                - total_score: Final ensemble score [0, 1]
                - details: Dict with individual metric scores
                - gating_passed: Whether gating threshold was passed
        """
        allowed_metrics = set(allowed_metrics) if allowed_metrics is not None else None

        semantic_metrics_to_cache = self._semantic_metrics_requested(allowed_metrics, skip_semantic)

        # Handle goal mask
        if goal_mask is not None:
            self.set_goal(goal_mask, semantic_metrics_to_cache=semantic_metrics_to_cache)
        elif self._goal_mask is None:
            raise ValueError("Goal mask must be set before evaluation.")
        elif semantic_metrics_to_cache:
            needs_cache = False
            if "dino" in semantic_metrics_to_cache and self.semantic_metrics._cached_goal_embedding is None:
                needs_cache = True
            if ({"lpips", "dists"} & semantic_metrics_to_cache) and self.semantic_metrics._cached_goal_tensor_01 is None:
                needs_cache = True
            if needs_cache:
                self.semantic_metrics.set_goal(self._goal_mask, metrics_to_cache=semantic_metrics_to_cache)

        goal = self._goal_mask

        # Resize current image if needed
        current_image = self.preprocessor.resize_image(current_image)

        # Preprocess current image to get prediction mask
        pred_mask, pred_soft_mask = self.preprocessor.process(current_image)

        # === Stage 1: Compute basic metrics ===
        geometric_requests = {
            m for m in ["iou", "elastic_iou", "f1"] if self._metric_enabled(m, allowed_metrics)
        }
        geometric_requests.add(self.gating_metric)
        geometric_results = self.geometric_metrics.compute(pred_mask, goal, metrics_to_compute=geometric_requests)

        contour_results = {}
        if self._metric_enabled("chamfer", allowed_metrics):
            contour_results = self.contour_metrics.compute(pred_mask, goal)

        score_map: Dict[str, Optional[float]] = {
            "iou": geometric_results.get("iou"),
            "elastic_iou": geometric_results.get("elastic_iou"),
            "f1": geometric_results.get("f1"),
            "sinkhorn": geometric_results.get("sinkhorn_score"),
            "chamfer": contour_results.get("chamfer_score") if contour_results else None,
            "dino": None,
            "lpips": None,
            "dists": None,
            "vlm": None,
        }

        gating_value = geometric_results.get(self.gating_metric) or 0.0
        gating_passed = not self.gating_enable or gating_value >= self.gating_threshold

        raw_metrics: Dict[str, Any] = {
            "geometric": geometric_results,
        }
        if contour_results:
            raw_metrics["contour"] = contour_results

        # Early exit on gating failure to save computation
        if not gating_passed:
            result = {
                "total_score": float(gating_value),
                "details": self._build_details(score_map),
                "gating_passed": False,
                "raw_metrics": raw_metrics,
                "score_map": score_map,
            }
            if save_debug or (save_debug is None and self.save_metrics_to_json):
                self._save_debug_output(result, current_image, pred_mask)
            return result

        # === Stage 2: Expensive geometric metric (Sinkhorn) ===
        if self._metric_enabled("sinkhorn", allowed_metrics):
            sinkhorn_only = self.geometric_metrics.compute(
                pred_mask, goal, metrics_to_compute={"sinkhorn"}
            )
            score_map["sinkhorn"] = sinkhorn_only.get("sinkhorn_score")
            geometric_results["sinkhorn_score"] = sinkhorn_only.get("sinkhorn_score")
            geometric_results["sinkhorn_divergence"] = sinkhorn_only.get("sinkhorn_divergence")

        # === Stage 3: Semantic metrics ===
        if semantic_metrics_to_cache:
            semantic_results = self.semantic_metrics.compute(
                pred_mask=pred_mask,
                goal_mask=goal,
                metrics_to_compute=semantic_metrics_to_cache,
                use_cached_goal=True,
            )
            score_map["dino"] = semantic_results.get("dino_score")
            score_map["lpips"] = semantic_results.get("lpips_score")
            score_map["dists"] = semantic_results.get("dists_score")
            raw_metrics["semantic"] = semantic_results

        # === Stage 4: Perceptual metric (VLM) ===
        vlm_requested = (not skip_vlm) and self._metric_enabled("vlm", allowed_metrics)
        if vlm_requested:
            try:
                vlm_results = self.vlm_client.compute(pred_mask, goal, image_index=image_index)
                score_map["vlm"] = vlm_results["vlm_score"]
                raw_metrics["vlm"] = vlm_results
            except Exception as e:
                logger.warning(f"VLM scoring failed, using fallback: {e}")
                non_vlm_scores = [v for k, v in score_map.items() if k != "vlm" and v is not None]
                fallback = float(np.mean(non_vlm_scores)) if non_vlm_scores else 0.0
                score_map["vlm"] = fallback
                raw_metrics["vlm"] = {"vlm_score": fallback, "reasoning": f"fallback: {e}"}

        # === Stage 5: Weighted ensemble ===
        total_score = self.aggregate_scores(score_map, allowed_metrics=allowed_metrics)

        result = {
            "total_score": float(total_score),
            "details": self._build_details(score_map),
            "gating_passed": True,
            "raw_metrics": raw_metrics,
            "score_map": score_map,
        }

        logger.info(
            "Evaluation complete: total=%.4f, geom=%.4f, contour=%s, semantic=%s, vlm=%s",
            total_score,
            gating_value,
            f"{score_map['chamfer']:.4f}" if score_map["chamfer"] is not None else "N/A",
            f"{score_map['dino']:.4f}" if score_map["dino"] is not None else "N/A",
            f"{score_map['vlm']:.4f}" if score_map["vlm"] is not None else "N/A",
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
