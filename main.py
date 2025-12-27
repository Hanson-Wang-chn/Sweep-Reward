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
    from src.preprocessor import Preprocessor
    from src.metrics.geometric import GeometricMetrics
    from src.metrics.contour import ContourMetrics

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
    results = []

    if basic_only or (skip_vlm and skip_dino):
        # Basic-only mode: sequential processing
        logger.info("Running basic metrics only for multiple images...")
        for i, (image, path) in enumerate(zip(current_images, image_paths)):
            logger.info(f"Processing image {i + 1}/{len(current_images)}: {os.path.basename(path)}")
            result = run_basic_evaluation(config, image, goal_mask, output_dir)
            result["image_path"] = path
            results.append(result)
    else:
        # Full evaluation with concurrent VLM calls
        from src.evaluator import Evaluator
        from src.preprocessor import Preprocessor
        from src.metrics.geometric import GeometricMetrics
        from src.metrics.contour import ContourMetrics
        from src.metrics.semantic import SemanticMetrics
        from src.vlm_client import VLMClient

        logger.info("Initializing evaluator...")
        evaluator = Evaluator(config)

        # Adjust weights if skipping modules
        if skip_vlm:
            logger.info("Skipping VLM evaluation (--skip-vlm flag set)")
            evaluator.weight_perceptual = 0.0
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

        # Set goal mask ONCE (caches DINOv2 embedding)
        logger.info("Setting goal mask (computing DINOv2 embedding once)...")
        evaluator.set_goal(goal_mask)

        # Pre-compute geometric/contour/DINO metrics sequentially
        # (DINO uses cached goal embedding)
        preprocessor = evaluator.preprocessor
        geometric_metrics = evaluator.geometric_metrics
        contour_metrics = evaluator.contour_metrics
        semantic_metrics = evaluator.semantic_metrics
        vlm_client = evaluator.vlm_client

        goal_processed = evaluator._goal_mask

        # Stage 1: Compute non-VLM metrics (can be parallelized on GPU for DINO)
        logger.info("Computing geometric, contour, and semantic metrics...")
        intermediate_results = []

        for i, (image, path) in enumerate(zip(current_images, image_paths)):
            logger.info(f"Processing image {i + 1}/{len(current_images)}: {os.path.basename(path)}")

            # Resize and process
            current_resized = preprocessor.resize_image(image)
            pred_mask, _ = preprocessor.process(current_resized)

            # Compute base metrics
            geo_results = geometric_metrics.compute(pred_mask, goal_processed)
            contour_results = contour_metrics.compute(pred_mask, goal_processed)

            s_geo = geo_results["f1"]
            s_contour = contour_results["chamfer_score"]

            # Check gating
            gating_threshold = config.get("ensemble", {}).get("gating", {}).get("threshold", 0.4)
            gating_passed = s_geo >= gating_threshold

            intermediate = {
                "image_path": path,
                "current_image": current_resized,
                "pred_mask": pred_mask,
                "geo_results": geo_results,
                "contour_results": contour_results,
                "s_geo": s_geo,
                "s_contour": s_contour,
                "gating_passed": gating_passed,
                "s_semantic": None,
                "s_perceptual": None,
            }

            if gating_passed and not skip_dino:
                # Compute semantic (DINO) score
                semantic_results = semantic_metrics.compute(current_resized, use_cached_goal=True)
                intermediate["s_semantic"] = semantic_results["dino_score"]
                intermediate["semantic_results"] = semantic_results

            intermediate_results.append(intermediate)

        # Stage 2: Concurrent VLM calls (only for images that passed gating)
        if not skip_vlm:
            images_for_vlm = [
                (i, r) for i, r in enumerate(intermediate_results)
                if r["gating_passed"]
            ]

            if images_for_vlm:
                logger.info(f"Running concurrent VLM calls for {len(images_for_vlm)} images...")

                def call_vlm(idx_and_result):
                    idx, result = idx_and_result
                    try:
                        vlm_result = vlm_client.compute(result["current_image"], goal_processed, image_index=idx)
                        return idx, vlm_result
                    except Exception as e:
                        logger.warning(f"VLM call failed for {result['image_path']}: {e}")
                        # Fallback score
                        s_fallback = (result["s_geo"] + result["s_contour"] +
                                     (result["s_semantic"] or 0.5)) / 3
                        return idx, {"vlm_score": s_fallback, "reasoning": f"VLM failed: {e}"}

                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {executor.submit(call_vlm, item): item for item in images_for_vlm}
                    for future in as_completed(futures):
                        idx, vlm_result = future.result()
                        intermediate_results[idx]["s_perceptual"] = vlm_result["vlm_score"]
                        intermediate_results[idx]["vlm_results"] = vlm_result

        # Stage 3: Compute final scores
        logger.info("Computing final scores...")
        for r in intermediate_results:
            if not r["gating_passed"]:
                # Failed gating
                total_score = r["s_geo"]
            else:
                s_geo = r["s_geo"]
                s_contour = r["s_contour"]
                s_semantic = r["s_semantic"] if r["s_semantic"] is not None else 0.5
                s_perceptual = r["s_perceptual"] if r["s_perceptual"] is not None else (
                    (s_geo + s_contour + s_semantic) / 3
                )

                total_score = (
                    evaluator.weight_geometric * s_geo
                    + evaluator.weight_contour * s_contour
                    + evaluator.weight_semantic * s_semantic
                    + evaluator.weight_perceptual * s_perceptual
                )

            result = {
                "image_path": r["image_path"],
                "total_score": float(total_score),
                "details": {
                    "iou": r["geo_results"]["elastic_iou"],
                    "f1": r["s_geo"],
                    "chamfer": r["s_contour"],
                    "dino": r.get("s_semantic"),
                    "vlm": r.get("s_perceptual"),
                },
                "gating_passed": r["gating_passed"],
                "raw_metrics": {
                    "geometric": r["geo_results"],
                    "contour": r["contour_results"],
                },
            }

            if "semantic_results" in r:
                result["raw_metrics"]["semantic"] = r["semantic_results"]
            if "vlm_results" in r:
                result["raw_metrics"]["vlm"] = r["vlm_results"]

            results.append(result)

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
        help="Skip DINOv2 evaluation (useful when model not available)",
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
            evaluator.set_goal(goal_mask)

            # Run evaluation
            logger.info("Running evaluation...")

            if args.skip_vlm:
                logger.info("Skipping VLM evaluation (--skip-vlm flag set)")
                evaluator.weight_perceptual = 0.0
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
