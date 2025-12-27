#!/usr/bin/env python
"""
Pseudo-label generator for Sweep-Reward project.
Generates goal images (binary masks) from real block photos.

Functionality:
  - Takes a real photo of blocks and extracts a binary mask using the project's Preprocessor
  - Uses HSV color segmentation for red blocks
  - Optionally applies morphological operations (closing to fill gaps, opening to remove noise)
  - Saves the result as a binary image (0/255)

Command-line arguments:
  - --input: Path to single input image
  - --input_dir: Path to directory containing images (batch mode)
  - --output: Output path for single image mode
  - --output_dir: Output directory for batch mode
  - --config: Path to config file (defaults to config/config.yaml)
  - --use-morphology: Enable morphological operations (default: False)

Example usage:
  # Single image without morphology (default)
  python generate_pseudo_label.py --input data/end-0.png --output data/goal-0.png

  # Single image with morphology
  python generate_pseudo_label.py --input data/end-0.png --output data/goal-0.png --use-morphology

  # Batch processing with morphology
  python generate_pseudo_label.py --input_dir data/raw --output_dir data/goals --use-morphology
"""

import argparse
import os
import sys

import cv2
import numpy as np
import yaml

from src.preprocessor import Preprocessor


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


def generate_pseudo_label(
    image: np.ndarray,
    preprocessor: Preprocessor,
    use_morphology: bool = False,
) -> np.ndarray:
    """
    Generate pseudo-label (binary mask) from input image.

    Args:
        image: RGB image (H, W, 3), uint8.
        preprocessor: Preprocessor instance.
        use_morphology: Whether to apply morphological operations.

    Returns:
        Binary mask (H, W), uint8, values 0 or 255.
    """
    # Process image to get binary mask
    # force_morphology parameter controls whether to apply morphological operations
    binary_mask, _ = preprocessor.process(image, force_morphology=use_morphology)

    # Convert 0/1 to 0/255 for saving
    binary_mask_255 = (binary_mask * 255).astype(np.uint8)

    return binary_mask_255


def save_binary_image(mask: np.ndarray, output_path: str) -> None:
    """
    Save binary mask as image file.

    Args:
        mask: Binary mask (H, W), uint8.
        output_path: Output file path.
    """
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Save as grayscale image
    cv2.imwrite(output_path, mask)
    print(f"Saved: {output_path}")


def process_single_image(
    input_path: str,
    output_path: str,
    preprocessor: Preprocessor,
    use_morphology: bool = False,
) -> None:
    """
    Process a single image and save the pseudo-label.

    Args:
        input_path: Path to input image.
        output_path: Path to save output binary mask.
        preprocessor: Preprocessor instance.
        use_morphology: Whether to apply morphological operations.
    """
    # Load image
    image = load_image(input_path)
    print(f"Loaded image: {input_path}, shape: {image.shape}")

    # Generate pseudo-label
    mask = generate_pseudo_label(image, preprocessor, use_morphology)

    # Save binary image
    save_binary_image(mask, output_path)


def process_directory(
    input_dir: str,
    output_dir: str,
    preprocessor: Preprocessor,
    use_morphology: bool = False,
    extensions: tuple = (".png", ".jpg", ".jpeg"),
) -> None:
    """
    Process all images in a directory.

    Args:
        input_dir: Input directory path.
        output_dir: Output directory path.
        preprocessor: Preprocessor instance.
        use_morphology: Whether to apply morphological operations.
        extensions: Tuple of valid image file extensions.
    """
    # Get all image files
    image_files = []
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(extensions):
            image_files.append(filename)

    if not image_files:
        print(f"No images found in {input_dir}")
        return

    print(f"Found {len(image_files)} images to process")

    # Process each image
    for filename in image_files:
        input_path = os.path.join(input_dir, filename)

        # Generate output filename
        name, _ = os.path.splitext(filename)
        output_filename = f"{name}_goal.png"
        output_path = os.path.join(output_dir, output_filename)

        try:
            process_single_image(input_path, output_path, preprocessor, use_morphology)
        except Exception as e:
            print(f"Error processing {input_path}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate pseudo-labels (binary masks) from real block photos."
    )

    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input",
        type=str,
        help="Path to single input image",
    )
    input_group.add_argument(
        "--input_dir",
        type=str,
        help="Path to input directory containing images",
    )

    # Output options
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save output binary mask (for single image mode)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Path to output directory (for directory mode)",
    )

    # Config file
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file",
    )

    # Morphology option
    parser.add_argument(
        "--use-morphology",
        action="store_true",
        help="Apply morphological operations (closing and opening) to the mask (default: False)",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.input and not args.output:
        # Default output path
        name, _ = os.path.splitext(args.input)
        args.output = f"{name}_goal.png"
        print(f"Output path not specified, using: {args.output}")

    if args.input_dir and not args.output_dir:
        # Default output directory
        args.output_dir = os.path.join(args.input_dir, "goals")
        print(f"Output directory not specified, using: {args.output_dir}")

    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)

    # Initialize preprocessor
    preprocessor = Preprocessor(config)
    print("Preprocessor initialized")

    # Display morphology setting
    if args.use_morphology:
        print("Morphological operations: ENABLED")
    else:
        print("Morphological operations: DISABLED (default)")

    # Process images
    if args.input:
        # Single image mode
        process_single_image(args.input, args.output, preprocessor, args.use_morphology)
    else:
        # Directory mode
        process_directory(args.input_dir, args.output_dir, preprocessor, args.use_morphology)

    print("Done!")


if __name__ == "__main__":
    main()
