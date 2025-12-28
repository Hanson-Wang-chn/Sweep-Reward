"""
Main test script for Sweep-Reward evaluation module.
Demonstrates usage of the multi-modal ensemble evaluator.
Supports both single image and multiple images evaluation.
"""

import argparse
import logging
import os
import sys
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
import yaml

from utils.visualization import Visualizer


def run_basic_evaluation(config, current_image, goal_mask, output_dir):
    """
    Run basic evaluation using only geometric and contour metrics.
    Does not require DINOv2 or VLM.
    """
    from src.evaluator import Evaluator
    from src.metrics.semantic import SemanticMetrics

    # Ensure debug output directory matches CLI argument
    debug_cfg = config.get("debug", {})
    debug_cfg["vis_output_dir"] = output_dir
    config["debug"] = debug_cfg

    evaluator = Evaluator(config)
    allowed_metrics = {"iou", "elastic_iou", "f1", "sinkhorn", "chamfer"}

    result = evaluator.evaluate(
        current_image,
        goal_mask=goal_mask,
        save_debug=True,
        allowed_metrics=allowed_metrics,
        skip_vlm=True,
        skip_semantic=True,
    )
    result["mode"] = "basic_only"

    # Save goal binary image for reference if visualization is enabled
    if config.get("debug", {}).get("enable_visualization", False):
        goal_processed = evaluator.preprocessor.process_goal_mask(goal_mask)
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


def get_image_paths_from_directory(directory: str, extensions: tuple = ('.png', '.jpg', '.jpeg', '.bmp')) -> List[str]:
    """
    Get all image paths from a directory.

    Args:
        directory: Directory path to scan.
        extensions: Tuple of valid image extensions.

    Returns:
        List of image file paths.
    """
    image_paths = []
    for filename in sorted(os.listdir(directory)):
        if filename.lower().endswith(extensions):
            image_paths.append(os.path.join(directory, filename))
    return image_paths


def apply_debug_config(config: dict, is_multi_image: bool) -> dict:
    """
    Apply appropriate debug settings based on single/multi image mode.

    Args:
        config: Configuration dictionary.
        is_multi_image: Whether processing multiple images.

    Returns:
        Updated configuration dictionary.
    """
    debug_cfg = config.get("debug", {})

    if is_multi_image:
        mode_cfg = debug_cfg.get("multi_image", {})
    else:
        mode_cfg = debug_cfg.get("single_image", {})

    # Apply mode-specific settings if they exist
    if mode_cfg:
        if "enable_visualization" in mode_cfg:
            debug_cfg["enable_visualization"] = mode_cfg["enable_visualization"]
        if "save_metrics_to_json" in mode_cfg:
            debug_cfg["save_metrics_to_json"] = mode_cfg["save_metrics_to_json"]

    config["debug"] = debug_cfg
    return config


def make_serializable(obj: Any) -> Any:
    """Convert numpy types to Python native types for JSON serialization."""
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


def evaluate_single_image_with_vlm(
    evaluator,
    current_image: np.ndarray,
    image_path: str,
    logger,
    skip_vlm: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate a single image. Called from thread pool for concurrent VLM calls.

    Args:
        evaluator: The Evaluator instance.
        current_image: The current RGB image.
        image_path: Path to the image (for logging).
        logger: Logger instance.
        skip_vlm: Whether to skip VLM evaluation.

    Returns:
        Evaluation result dictionary.
    """
    try:
        result = evaluator.evaluate(current_image, save_debug=False)
        result["image_path"] = image_path
        return result
    except Exception as e:
        logger.error(f"Evaluation failed for {image_path}: {e}")
        return {
            "image_path": image_path,
            "error": str(e),
            "total_score": 0.0,
            "details": {"iou": None, "f1": None, "chamfer": None, "dino": None, "vlm": None},
            "gating_passed": False,
        }


def run_multi_image_evaluation(
    config: dict,
    current_images: List[np.ndarray],
    image_paths: List[str],
    goal_mask: np.ndarray,
    output_dir: str,
    logger,
    skip_vlm: bool = False,
    skip_dino: bool = False,
    basic_only: bool = False,
    visualize: bool = False,
    max_workers: int = 4,
) -> List[Dict[str, Any]]:
    """
    Evaluate multiple current images against a single goal.
    Uses concurrent VLM calls for efficiency.

    Args:
        config: Configuration dictionary.
        current_images: List of current RGB images.
        image_paths: List of image paths for identification.
        goal_mask: Goal mask image.
        output_dir: Output directory for results.
        logger: Logger instance.
        skip_vlm: Whether to skip VLM evaluation.
        skip_dino: Whether to skip DINO evaluation.
        basic_only: Whether to run basic metrics only.
        visualize: Whether to generate visualizations.
        max_workers: Maximum number of concurrent workers for VLM calls.

    Returns:
        List of evaluation results.
    """
    from src.evaluator import Evaluator

    results = []

    # Build allowed metric set
    full_metric_set = {"iou", "elastic_iou", "f1", "sinkhorn", "chamfer", "dino", "lpips", "dists", "vlm"}
    allowed_metrics = set(full_metric_set)
    if basic_only:
        allowed_metrics = {"iou", "elastic_iou", "f1", "sinkhorn", "chamfer"}
        skip_vlm = True
        skip_dino = True
    if skip_vlm:
        allowed_metrics.discard("vlm")
    if skip_dino:
        allowed_metrics -= {"dino", "lpips", "dists"}

    logger.info("Initializing evaluator...")
    evaluator = Evaluator(config)

    semantic_to_cache = evaluator._semantic_metrics_requested(allowed_metrics, skip_semantic=skip_dino)
    evaluator.set_goal(goal_mask, semantic_metrics_to_cache=semantic_to_cache)

    # Stage 1: compute all non-VLM metrics
    logger.info("Computing geometric/contour/semantic metrics...")
    base_results = []
    for i, (image, path) in enumerate(zip(current_images, image_paths)):
        logger.info(f"Processing image {i + 1}/{len(current_images)}: {os.path.basename(path)}")
        base = evaluator.evaluate(
            image,
            save_debug=False,
            allowed_metrics=allowed_metrics,
            skip_vlm=True,
            skip_semantic=skip_dino,
        )
        base["image_path"] = path
        base_results.append(base)

    # Stage 2: VLM calls if requested
    vlm_requested = (not skip_vlm) and evaluator.metric_weights.get("vlm", 0.0) > 0 and "vlm" in allowed_metrics
    if vlm_requested:
        goal_processed = evaluator._goal_mask
        preprocessor = evaluator.preprocessor
        vlm_client = evaluator.vlm_client

        tasks = [
            (idx, base)
            for idx, base in enumerate(base_results)
            if base.get("gating_passed", False)
        ]

        if tasks:
            logger.info(f"Running concurrent VLM calls for {len(tasks)} images...")

            def call_vlm(idx_and_result):
                idx, base = idx_and_result
                current_image = current_images[idx]
                current_resized = preprocessor.resize_image(current_image)
                pred_mask, _ = preprocessor.process(current_resized)
                try:
                    vlm_result = vlm_client.compute(pred_mask, goal_processed, image_index=idx)
                    score = vlm_result["vlm_score"]
                except Exception as e:
                    logger.warning(f"VLM call failed for {base['image_path']}: {e}")
                    non_vlm_scores = [
                        v for k, v in base.get("score_map", {}).items()
                        if k != "vlm" and v is not None
                    ]
                    score = float(np.mean(non_vlm_scores)) if non_vlm_scores else 0.0
                    vlm_result = {"vlm_score": score, "reasoning": f"VLM failed: {e}"}
                return idx, vlm_result, score

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(call_vlm, task): task for task in tasks}
                for future in as_completed(futures):
                    idx, vlm_result, vlm_score = future.result()
                    base_results[idx].setdefault("raw_metrics", {})["vlm"] = vlm_result
                    base_results[idx].setdefault("score_map", {})["vlm"] = vlm_score

    # Stage 3: Finalize totals (recompute if VLM was added)
    for base in base_results:
        if "score_map" in base:
            base["details"] = evaluator._build_details(base["score_map"])
            if base.get("gating_passed", False):
                base["total_score"] = evaluator.aggregate_scores(base["score_map"], allowed_metrics=allowed_metrics)
        results.append(base)

    return results


def print_results(results: List[Dict[str, Any]], is_multi: bool = False):
    """Print evaluation results to terminal."""
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    if is_multi:
        # Summary statistics
        scores = [r["total_score"] for r in results if "error" not in r]
        if scores:
            print(f"Total Images: {len(results)}")
            print(f"Average Score: {np.mean(scores):.4f}")
            print(f"Min Score: {np.min(scores):.4f}")
            print(f"Max Score: {np.max(scores):.4f}")
            print(f"Std Dev: {np.std(scores):.4f}")
        print("-" * 70)

        # Individual results
        for i, result in enumerate(results):
            image_name = os.path.basename(result.get("image_path", f"Image {i + 1}"))
            if "error" in result:
                print(f"[{i + 1:3d}] {image_name}: ERROR - {result['error']}")
            else:
                gating = "✓" if result["gating_passed"] else "✗"
                print(f"[{i + 1:3d}] {image_name}: {result['total_score']:.4f} (Gating: {gating})")
                details = result["details"]
                detail_str = ", ".join(
                    f"{k}={v:.3f}" if v is not None else f"{k}=N/A"
                    for k, v in details.items()
                )
                print(f"      {detail_str}")
    else:
        result = results[0] if results else {}
        print(f"Total Score: {result.get('total_score', 0):.4f}")
        print(f"Gating Passed: {result.get('gating_passed', False)}")
        print("\nDetail Scores:")
        for key, value in result.get("details", {}).items():
            if value is not None:
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: N/A")

    print("=" * 70)


def save_results_json(results: List[Dict[str, Any]], output_dir: str, timestamp: str):
    """Save all results to a single JSON file."""
    os.makedirs(output_dir, exist_ok=True)

    # Build summary
    scores = [r["total_score"] for r in results if "error" not in r]
    summary = {
        "timestamp": timestamp,
        "total_images": len(results),
        "successful_evaluations": len(scores),
        "failed_evaluations": len(results) - len(scores),
    }

    if scores:
        summary["statistics"] = {
            "mean_score": float(np.mean(scores)),
            "min_score": float(np.min(scores)),
            "max_score": float(np.max(scores)),
            "std_score": float(np.std(scores)),
        }

    output = {
        "summary": summary,
        "results": make_serializable(results),
    }

    json_path = os.path.join(output_dir, f"batch_results_{timestamp}.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)

    return json_path


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
        nargs="+",
        default=["example/example_current.png"],
        help="Path(s) to current state image(s). Can specify multiple images or a directory.",
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
        help="Skip semantic metrics (DINOv2/LPIPS/DISTS) when model weights are unavailable",
    )
    parser.add_argument(
        "--basic-only",
        action="store_true",
        help="Only run basic geometric and contour metrics (no DINOv2 or VLM)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum number of concurrent workers for VLM calls (default: 4)",
    )

    args = parser.parse_args()

    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)

    # Setup logging
    log_level = config.get("debug", {}).get("log_level", "INFO")
    setup_logging(log_level)
    logger = logging.getLogger(__name__)

    # Determine if single or multiple images
    current_paths = []
    for path in args.current:
        if os.path.isdir(path):
            # Directory provided - get all images
            dir_images = get_image_paths_from_directory(path)
            if not dir_images:
                logger.warning(f"No images found in directory: {path}")
            current_paths.extend(dir_images)
        else:
            current_paths.append(path)

    if not current_paths:
        logger.error("No valid image paths provided.")
        sys.exit(1)

    is_multi_image = len(current_paths) > 1

    # Apply appropriate debug config based on mode
    config = apply_debug_config(config, is_multi_image)

    # Override output directory if specified
    if args.output:
        config["debug"]["vis_output_dir"] = args.output

    # Override visualization setting (command line takes precedence)
    if args.visualize:
        config["debug"]["enable_visualization"] = True

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Load goal mask
    logger.info(f"Loading goal mask: {args.goal}")
    goal_mask = load_image(args.goal)
    logger.info(f"Goal mask shape: {goal_mask.shape}")

    # Load current images
    logger.info(f"Loading {len(current_paths)} current image(s)...")
    current_images = []
    valid_paths = []
    for path in current_paths:
        try:
            image = load_image(path)
            current_images.append(image)
            valid_paths.append(path)
            logger.info(f"  Loaded: {path} (shape: {image.shape})")
        except Exception as e:
            logger.error(f"  Failed to load {path}: {e}")

    if not current_images:
        logger.error("No valid images loaded.")
        sys.exit(1)

    # Determine which metrics are allowed based on CLI flags
    full_metric_set = {"iou", "elastic_iou", "f1", "sinkhorn", "chamfer", "dino", "lpips", "dists", "vlm"}
    allowed_metrics = set(full_metric_set)
    if args.basic_only:
        allowed_metrics = {"iou", "elastic_iou", "f1", "sinkhorn", "chamfer"}
        args.skip_vlm = True
        args.skip_dino = True
    if args.skip_vlm:
        allowed_metrics.discard("vlm")
    if args.skip_dino:
        allowed_metrics -= {"dino", "lpips", "dists"}

    # Generate timestamp for output files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Run evaluation
    if is_multi_image:
        logger.info(f"Running multi-image evaluation ({len(current_images)} images)...")
        results = run_multi_image_evaluation(
            config=config,
            current_images=current_images,
            image_paths=valid_paths,
            goal_mask=goal_mask,
            output_dir=args.output,
            logger=logger,
            skip_vlm=args.skip_vlm,
            skip_dino=args.skip_dino,
            basic_only=args.basic_only,
            visualize=args.visualize,
            max_workers=args.max_workers,
        )
    else:
        # Single image evaluation
        current_image = current_images[0]
        logger.info(f"Running single image evaluation...")

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
            semantic_to_cache = evaluator._semantic_metrics_requested(allowed_metrics, skip_semantic=args.skip_dino)
            evaluator.set_goal(goal_mask, semantic_metrics_to_cache=semantic_to_cache)

            # Run evaluation
            logger.info("Running evaluation...")
            result = evaluator.evaluate(
                current_image,
                save_debug=True,
                allowed_metrics=allowed_metrics,
                skip_vlm=args.skip_vlm,
                skip_semantic=args.skip_dino,
            )

        result["image_path"] = valid_paths[0]
        results = [result]

    # Print results
    print_results(results, is_multi=is_multi_image)

    # Save results to JSON
    if config.get("debug", {}).get("save_metrics_to_json", True):
        json_path = save_results_json(results, args.output, timestamp)
        logger.info(f"Results saved to: {json_path}")

    # Generate visualization if requested (single image only for detailed visualization)
    if not is_multi_image and (args.visualize or config.get("debug", {}).get("enable_visualization", False)):
        logger.info("Generating visualization...")
        visualizer = Visualizer(args.output)

        # Get prediction mask for visualization
        from src.preprocessor import Preprocessor
        preprocessor = Preprocessor(config)
        current_resized = preprocessor.resize_image(current_images[0])
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
            current_resized, pred_mask, goal_processed, results[0], save_path=vis_path
        )
        logger.info(f"Visualization saved to: {vis_path}")

        # Create mask comparison
        mask_vis_path = os.path.join(args.output, "mask_comparison.png")
        visualizer.visualize_mask_comparison(
            pred_mask, goal_processed, current_resized, save_path=mask_vis_path
        )
        logger.info(f"Mask comparison saved to: {mask_vis_path}")

    logger.info("Evaluation complete!")

    return results


if __name__ == "__main__":
    main()
