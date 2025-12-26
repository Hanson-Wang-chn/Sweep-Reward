"""
Preprocessor module for color segmentation and morphological operations.
Extracts clean Lego block masks from RGB images.
"""

import cv2
import numpy as np
from typing import Dict, Any, Tuple


class Preprocessor:
    """
    Preprocessor for extracting Lego block masks from RGB images.
    Performs color segmentation in HSV space and morphological operations.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize preprocessor with configuration.

        Args:
            config: Configuration dictionary containing preprocess settings.
        """
        self.config = config
        preprocess_cfg = config.get("preprocess", {})

        # Color segmentation parameters (HSV ranges for red)
        color_seg = preprocess_cfg.get("color_segmentation", {})
        hsv1 = color_seg.get("hsv_range_1", {})
        hsv2 = color_seg.get("hsv_range_2", {})

        self.hsv_lower_1 = np.array(hsv1.get("lower", [0, 100, 70]))
        self.hsv_upper_1 = np.array(hsv1.get("upper", [10, 255, 255]))
        self.hsv_lower_2 = np.array(hsv2.get("lower", [170, 100, 70]))
        self.hsv_upper_2 = np.array(hsv2.get("upper", [180, 255, 255]))

        # Morphology parameters
        morph_cfg = preprocess_cfg.get("morphology", {})

        # Closing operation parameters
        closing_cfg = morph_cfg.get("closing", {})
        self.closing_kernel_size = closing_cfg.get("kernel_size", 5)
        self.closing_iterations = closing_cfg.get("iterations", 2)
        self.closing_kernel_shape = closing_cfg.get("kernel_shape", "ellipse")

        # Opening operation parameters
        opening_cfg = morph_cfg.get("opening", {})
        self.opening_kernel_size = opening_cfg.get("kernel_size", 3)
        self.opening_iterations = opening_cfg.get("iterations", 1)
        self.opening_kernel_shape = opening_cfg.get("kernel_shape", "ellipse")

        # Smoothing parameters
        smoothing_cfg = preprocess_cfg.get("smoothing", {})
        self.smoothing_enable = smoothing_cfg.get("enable", True)
        self.gaussian_kernel = smoothing_cfg.get("gaussian_kernel", 5)
        self.gaussian_sigma = smoothing_cfg.get("sigma", 1.5)

        # Image size from system config
        system_cfg = config.get("system", {})
        self.image_size = tuple(system_cfg.get("image_size", [224, 224]))

    def _get_kernel_shape(self, shape_name: str) -> int:
        """Convert kernel shape name to OpenCV constant."""
        shape_map = {
            "rect": cv2.MORPH_RECT,
            "cross": cv2.MORPH_CROSS,
            "ellipse": cv2.MORPH_ELLIPSE,
        }
        return shape_map.get(shape_name, cv2.MORPH_ELLIPSE)

    def _create_kernel(self, size: int, shape: str) -> np.ndarray:
        """Create morphological kernel."""
        shape_cv = self._get_kernel_shape(shape)
        return cv2.getStructuringElement(shape_cv, (size, size))

    def resize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Resize image to standard size if needed.

        Args:
            image: Input image (H, W, C) or (H, W).

        Returns:
            Resized image.
        """
        h, w = image.shape[:2]
        target_h, target_w = self.image_size

        if h != target_h or w != target_w:
            if len(image.shape) == 3:
                return cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            else:
                return cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        return image

    def color_segmentation(self, image: np.ndarray) -> np.ndarray:
        """
        Perform color segmentation to extract red Lego blocks.
        Uses dual HSV ranges to handle red color crossing 0 degree.

        Args:
            image: RGB image (H, W, 3), uint8.

        Returns:
            Binary mask (H, W), uint8, 0 or 255.
        """
        # Convert RGB to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        # Create masks for both red ranges
        mask1 = cv2.inRange(hsv, self.hsv_lower_1, self.hsv_upper_1)
        mask2 = cv2.inRange(hsv, self.hsv_lower_2, self.hsv_upper_2)

        # Combine masks
        mask = cv2.bitwise_or(mask1, mask2)

        return mask

    def morphological_operations(self, mask: np.ndarray) -> np.ndarray:
        """
        Apply morphological operations to clean up the mask.
        Closing fills gaps between blocks, opening removes noise.

        Args:
            mask: Binary mask (H, W), uint8.

        Returns:
            Cleaned binary mask (H, W), uint8.
        """
        # Closing operation: fill gaps between Lego blocks
        closing_kernel = self._create_kernel(
            self.closing_kernel_size, self.closing_kernel_shape
        )
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, closing_kernel, iterations=self.closing_iterations
        )

        # Opening operation: remove isolated noise
        opening_kernel = self._create_kernel(
            self.opening_kernel_size, self.opening_kernel_shape
        )
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN, opening_kernel, iterations=self.opening_iterations
        )

        return mask

    def smooth_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Apply Gaussian blur to smooth mask edges.

        Args:
            mask: Binary mask (H, W), uint8.

        Returns:
            Smoothed mask (H, W), float32, values in [0, 1].
        """
        if not self.smoothing_enable:
            return (mask / 255.0).astype(np.float32)

        # Apply Gaussian blur
        smoothed = cv2.GaussianBlur(
            mask.astype(np.float32),
            (self.gaussian_kernel, self.gaussian_kernel),
            self.gaussian_sigma,
        )

        # Normalize to [0, 1]
        smoothed = smoothed / 255.0

        return smoothed.astype(np.float32)

    def process(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Full preprocessing pipeline.

        Args:
            image: RGB image (H, W, 3), uint8.

        Returns:
            Tuple of:
                - Binary mask (H, W), uint8, 0/255
                - Soft mask (H, W), float32, [0, 1]
        """
        # Resize if needed
        image = self.resize_image(image)

        # Color segmentation
        mask = self.color_segmentation(image)

        # Morphological operations
        mask = self.morphological_operations(mask)

        # Generate soft mask
        soft_mask = self.smooth_mask(mask)

        # Convert binary mask to 0/1
        binary_mask = (mask > 127).astype(np.uint8)

        return binary_mask, soft_mask

    def process_goal_mask(self, goal_mask: np.ndarray) -> np.ndarray:
        """
        Process goal mask to ensure it's in correct format.

        Args:
            goal_mask: Goal mask image, can be RGB or grayscale.

        Returns:
            Binary mask (H, W), uint8, 0/1.
        """
        # Resize if needed
        goal_mask = self.resize_image(goal_mask)

        # Convert to grayscale if needed
        if len(goal_mask.shape) == 3:
            goal_mask = cv2.cvtColor(goal_mask, cv2.COLOR_RGB2GRAY)

        # Use Otsu's thresholding for robust binarization
        # This automatically finds the optimal threshold
        _, binary = cv2.threshold(goal_mask, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Check which side has more pixels - assume target shape has fewer pixels
        # (letters typically occupy less than 50% of image)
        if binary.sum() > (binary.size / 2):
            # Invert if more than half is white (shape is the dark part)
            binary = 1 - binary

        return binary.astype(np.uint8)
