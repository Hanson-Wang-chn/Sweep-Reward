"""
Main test script for Sweep-Reward evaluation module.
Demonstrates usage of the multi-modal ensemble evaluator.
"""

import argparse
import logging
import os
import sys

import cv2
import numpy as np
import yaml

from utils.visualization import Visualizer


def run_basic_evaluation(config, current_image, goal_mask, output_dir):
    """
    Run basic evaluation using only geometric and contour metrics.
    Does not require DINOv2 or VLM.
    """
    from src.preprocessor import Preprocessor
    from src.metrics.geometric import GeometricMetrics
    from src.metrics.contour import ContourMetrics
    import json
    from datetime import datetime

    # Initialize modules
    preprocessor = Preprocessor(config)
    geometric_metrics = GeometricMetrics(config)
    contour_metrics = ContourMetrics(config)

    # Process images
    current_image = preprocessor.resize_image(current_image)
    pred_mask, _ = preprocessor.process(current_image)
    goal_processed = preprocessor.process_goal_mask(goal_mask)

    # Compute metrics
    geo_results = geometric_metrics.compute(pred_mask, goal_processed)
    contour_results = contour_metrics.compute(pred_mask, goal_processed)

    # Get weights from config
    weights = config.get("ensemble", {}).get("weights", {})
    w_geo = weights.get("geometric", 0.35)
    w_contour = weights.get("contour", 0.25)

    # Normalize weights for basic-only mode
    total_w = w_geo + w_contour
    w_geo_norm = w_geo / total_w
    w_contour_norm = w_contour / total_w

    # Compute total score
    s_geo = geo_results["f1"]
    s_contour = contour_results["chamfer_score"]
    total_score = w_geo_norm * s_geo + w_contour_norm * s_contour

    # Build result
    result = {
        "total_score": total_score,
        "details": {
            "iou": geo_results["elastic_iou"],
            "f1": s_geo,
            "chamfer": s_contour,
            "dino": None,
            "vlm": None,
        },
        "gating_passed": s_geo >= config.get("ensemble", {}).get("gating", {}).get("threshold", 0.4),
        "raw_metrics": {
            "geometric": geo_results,
            "contour": contour_results,
        },
        "mode": "basic_only",
    }

    # Save metrics to JSON
    if config.get("debug", {}).get("save_metrics_to_json", True):
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = os.path.join(output_dir, f"metrics_{timestamp}.json")

        # Make serializable
        def make_serializable(obj):
            if isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [make_serializable(v) for v in obj]
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            return obj

        with open(json_path, "w") as f:
            json.dump(make_serializable(result), f, indent=2)

    # Save visualization outputs if enabled
    if config.get("debug", {}).get("enable_visualization", False):
        # Save goal binary image
        from src.metrics.semantic import SemanticMetrics
        semantic_metrics = SemanticMetrics(config)
        goal_binary = semantic_metrics.mask_to_binary_image(goal_processed)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        goal_binary_path = os.path.join(output_dir, f"goal_binary_{timestamp}.png")
        cv2.imwrite(goal_binary_path, cv2.cvtColor(goal_binary, cv2.COLOR_RGB2BGR))
    
    return result


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def load_image(image_path: str) -> np.ndarray:
    """
    Load image from file path.
    Returns RGB image.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def main():
    parser = argparse.ArgumentParser(
        description="Sweep-Reward: Multi-modal Ensemble Evaluation Module"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--current",
        type=str,
        default="example/example_current.png",
        help="Path to current state image",
    )
    parser.add_argument(
        "--goal",
        type=str,
        default="example/example_goal.png",
        help="Path to goal mask image",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./logs/vis_eval",
        help="Output directory for results",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualization outputs",
    )
    parser.add_argument(
        "--skip-vlm",
        action="store_true",
        help="Skip VLM evaluation (useful when API key not available)",
    )
    parser.add_argument(
        "--skip-dino",
        action="store_true",
        help="Skip DINOv2 evaluation (useful when model not available)",
    )
    parser.add_argument(
        "--basic-only",
        action="store_true",
        help="Only run basic geometric and contour metrics (no DINOv2 or VLM)",
    )

    args = parser.parse_args()

    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)

    # Setup logging
    log_level = config.get("debug", {}).get("log_level", "INFO")
    setup_logging(log_level)
    logger = logging.getLogger(__name__)

    # Override output directory if specified
    if args.output:
        config["debug"]["vis_output_dir"] = args.output

    # Override visualization setting
    if args.visualize:
        config["debug"]["enable_visualization"] = True

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Load images
    logger.info(f"Loading current image: {args.current}")
    current_image = load_image(args.current)
    logger.info(f"Current image shape: {current_image.shape}")

    logger.info(f"Loading goal mask: {args.goal}")
    goal_mask = load_image(args.goal)
    logger.info(f"Goal mask shape: {goal_mask.shape}")

    # Check if running basic-only mode
    if args.basic_only or (args.skip_vlm and args.skip_dino):
        logger.info("Running basic metrics only (no DINOv2 or VLM)...")
        result = run_basic_evaluation(config, current_image, goal_mask, args.output)
    else:
        # Initialize full evaluator
        logger.info("Initializing evaluator...")
        from src.evaluator import Evaluator
        evaluator = Evaluator(config)

        # Set goal mask (caches DINOv2 embedding)
        logger.info("Setting goal mask...")
        evaluator.set_goal(goal_mask)

        # Run evaluation
        logger.info("Running evaluation...")

        if args.skip_vlm:
            # Temporarily disable VLM by setting very low weight
            logger.info("Skipping VLM evaluation (--skip-vlm flag set)")
            evaluator.weight_perceptual = 0.0
            # Redistribute weight
            total_other = (
                evaluator.weight_geometric
                + evaluator.weight_contour
                + evaluator.weight_semantic
            )
            if total_other > 0:
                scale = 1.0 / total_other
                evaluator.weight_geometric *= scale
                evaluator.weight_contour *= scale
                evaluator.weight_semantic *= scale

        result = evaluator.evaluate(current_image, save_debug=True)

    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Total Score: {result['total_score']:.4f}")
    print(f"Gating Passed: {result['gating_passed']}")
    print("\nDetail Scores:")
    for key, value in result["details"].items():
        if value is not None:
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: N/A")
    print("=" * 60)

    # Generate visualization if requested
    if args.visualize or config.get("debug", {}).get("enable_visualization", False):
        logger.info("Generating visualization...")
        visualizer = Visualizer(args.output)

        # Get prediction mask for visualization
        from src.preprocessor import Preprocessor
        preprocessor = Preprocessor(config)
        current_resized = preprocessor.resize_image(current_image)
        pred_mask, _ = preprocessor.process(current_resized)
        goal_processed = preprocessor.process_goal_mask(goal_mask)

        # Save binary goal image for DINO
        if not args.basic_only:
            from src.metrics.semantic import SemanticMetrics
            semantic_metrics = SemanticMetrics(config)
            goal_binary = semantic_metrics.mask_to_binary_image(goal_processed)
            goal_binary_path = os.path.join(args.output, "goal_binary_final.png")
            cv2.imwrite(goal_binary_path, cv2.cvtColor(goal_binary, cv2.COLOR_RGB2BGR))
            logger.info(f"Binary goal image saved to: {goal_binary_path}")

        # Create evaluation visualization
        vis_path = os.path.join(args.output, "evaluation_result.png")
        visualizer.visualize_evaluation_result(
            current_resized, pred_mask, goal_processed, result, save_path=vis_path
        )
        logger.info(f"Visualization saved to: {vis_path}")

        # Create mask comparison
        mask_vis_path = os.path.join(args.output, "mask_comparison.png")
        visualizer.visualize_mask_comparison(
            pred_mask, goal_processed, current_resized, save_path=mask_vis_path
        )
        logger.info(f"Mask comparison saved to: {mask_vis_path}")

    logger.info("Evaluation complete!")

    return result


if __name__ == "__main__":
    main()
