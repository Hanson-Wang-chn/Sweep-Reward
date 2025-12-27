"""
VLM Client for OpenRouter API integration.
Handles GPT-4o scoring with Chain-of-Thought prompting.
"""

import os
import json
import base64
import time
import logging
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

import numpy as np
import cv2
import requests


logger = logging.getLogger(__name__)


class VLMClient:
    """
    VLM Client for perceptual scoring using OpenRouter API.
    Uses VLM with CoT prompting to evaluate shape matching quality.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize VLM client with configuration.

        Args:
            config: Configuration dictionary.
        """
        self.config = config
        vlm_cfg = config.get("metrics", {}).get("perceptual", {}).get("vlm", {})

        # API configuration
        self.model_name = vlm_cfg.get("model_name", "openai/gpt-4o")
        self.api_key_env_var = vlm_cfg.get("api_key_env_var", "OPENROUTER_API_KEY")
        self.base_url = vlm_cfg.get("base_url", "https://openrouter.ai/api/v1")

        # Request parameters
        self.temperature = vlm_cfg.get("temperature", 0.0)
        self.max_tokens = vlm_cfg.get("max_tokens", 1024)
        self.timeout = vlm_cfg.get("timeout", 30)
        self.max_retries = vlm_cfg.get("max_retries", 5)

        # System prompt
        self.system_prompt = vlm_cfg.get("system_prompt", self._default_system_prompt())

        # Debug configuration
        debug_cfg = config.get("debug", {})
        self.save_vlm_imgs = debug_cfg.get("save_vlm_imgs", False)
        self.vis_output_dir = debug_cfg.get("vis_output_dir", "./logs")

    def _default_system_prompt(self) -> str:
        """Return default system prompt."""
        return """# Role
        You are a pragmatic robot evaluator assessing a "Lego Sweeping" task. You understand that sweeping granular objects (small Lego bricks) results in naturally rough edges and noise. **Perfection is not required.** Your goal is to judge if the robot has successfully formed the *semantic shape*.
        # Input
        - **Image 1**: The Goal Shape (Ideal binary mask).
        - **Image 2**: The Current Observation (Actual state of bricks).
        All images are black-and-white masks. White pixels represent LEGO bricks filled material. Black pixels represent empty space.
        # Context & Lenience Guidelines
        - **Granular Material**: The objects are small bricks. Straight lines will inevitably look jagged or "pixelated." **Do not penalize for jagged edges.**
        - **Stray Bricks**: A moderate amount of scattered bricks (noise) outside the main shape is acceptable and expected. **Ignore isolated background noise.**
        - **Focus**: Prioritize **Topological Correctness** (e.g., "Does it look like the letter?") over **Geometric Precision** (e.g., "Are the lines perfectly straight?").
        # Evaluation Criteria (0.1 - 0.9)
        Please assign a score based on the following relaxed standards:
        - **0.1 (Unrecognizable)**: The bricks are piled randomly. No coherent shape is visible.
        - **0.3 (Attempted but Failed)**: You can guess what the robot tried to do, but major parts are missing (e.g., an 'E' missing the middle bar) or the shape is broken into disconnected islands.
        - **0.5 (Passable)**: The shape is clearly recognizable as the target letter/symbol. It may be significantly thicker/thinner than the goal, or have 1-2 moderate gaps, but the identity is unambiguous.
        - **0.7 (Good Success)**: The shape matches the goal's topology perfectly. The strokes are connected. There might be some fuzzy edges or a few scattered bricks nearby, but the main structure is solid.
        - **0.9 (Excellent)**: The shape is distinct, correctly oriented, and topologically complete. Even if the borders are wavy or not perfectly aligned with the goal mask pixels, visually, it is a great result for a robot.
        # Reasoning Steps
        1. **Identify**: Can you instantly recognize the shape in Image 2 as the shape in Image 1 without guessing? If yes, start from score 0.5.
        2. **Topology Check**: Are all necessary strokes present and connected? (e.g., "Z" has top, diagonal, bottom). If yes, boost score to 0.7+.
        3. **Noise Tolerance**: Is the noise distracting? If the main shape is prominent enough to ignore the noise, maintain the high score.
        # Output Format
        {
          "reasoning": "Brief justification focusing on recognizability and topology...",
          "score": <float between 0.1 and 0.9>
        }"""

    def _get_api_key(self) -> str:
        """Get API key from environment variable."""
        api_key = os.environ.get(self.api_key_env_var)
        if not api_key:
            raise ValueError(
                f"API key not found. Please set the {self.api_key_env_var} environment variable."
            )
        return api_key

    def mask_to_binary_image(self, mask: np.ndarray) -> np.ndarray:
        """
        Convert binary mask to 3-channel black-white image.

        Args:
            mask: Binary mask (H, W), values 0/1.

        Returns:
            RGB image (H, W, 3), uint8, black (0) and white (255).
        """
        # Convert to 0-255 range
        binary_255 = (mask * 255).astype(np.uint8)

        # Create 3-channel image
        binary_rgb = np.stack([binary_255, binary_255, binary_255], axis=-1)

        return binary_rgb

    def create_comparison_image(
        self,
        current_mask: np.ndarray,
        goal_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Create side-by-side comparison image for VLM input.
        Both images are binary (black-white).

        Args:
            current_mask: Binary mask for current image (H, W), values 0/1.
            goal_mask: Binary goal mask (H, W), values 0/1.

        Returns:
            Concatenated comparison image (H, 2W, 3), uint8.
        """
        # Convert masks to binary RGB images
        current_binary = self.mask_to_binary_image(current_mask)

        # Convert goal mask to binary image
        goal_image = self.mask_to_binary_image(goal_mask)

        # Ensure same size
        h1, w1 = current_binary.shape[:2]
        h2, w2 = goal_image.shape[:2]

        if h1 != h2 or w1 != w2:
            goal_image = cv2.resize(goal_image, (w1, h1))

        # Add labels
        current_labeled = current_binary.copy()
        goal_labeled = goal_image.copy()

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(current_labeled, "Current", (10, 25), font, 0.7, (128, 128, 128), 2)
        cv2.putText(goal_labeled, "Goal", (10, 25), font, 0.7, (128, 128, 128), 2)

        # Concatenate horizontally
        comparison = np.concatenate([current_labeled, goal_labeled], axis=1)

        return comparison

    def image_to_base64(self, image: np.ndarray) -> str:
        """
        Convert numpy image to base64 string.

        Args:
            image: RGB image (H, W, 3), uint8.

        Returns:
            Base64 encoded string.
        """
        # Convert RGB to BGR for cv2
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Encode to PNG
        _, buffer = cv2.imencode(".png", image_bgr)
        base64_str = base64.b64encode(buffer).decode("utf-8")

        return base64_str

    def _parse_response(self, response_text: str) -> Tuple[float, str]:
        """
        Parse VLM response to extract score and reasoning.

        Args:
            response_text: Raw response text from VLM.

        Returns:
            Tuple of (score, reasoning).
        """
        # Try to find JSON in response
        try:
            # Look for JSON pattern
            import re
            json_match = re.search(r'\{[^}]*"score"[^}]*\}', response_text)

            if json_match:
                json_str = json_match.group()
                data = json.loads(json_str)
                score = float(data.get("score", 0.5))
                reasoning = data.get("reasoning", "")
                return score, reasoning
        except (json.JSONDecodeError, ValueError, KeyError):
            pass

        # Fallback: try to extract score from text
        try:
            import re
            score_match = re.search(r'score["\s:]+(\d+\.?\d*)', response_text.lower())
            if score_match:
                score = float(score_match.group(1))
                score = max(0.0, min(1.0, score))
                return score, response_text
        except (ValueError, AttributeError):
            pass

        # Default fallback
        logger.warning(f"Could not parse VLM response: {response_text}")
        return 0.5, response_text

    def call_api(
        self,
        comparison_image: np.ndarray,
    ) -> Tuple[float, str]:
        """
        Call OpenRouter API with comparison image.

        Args:
            comparison_image: Side-by-side comparison image.

        Returns:
            Tuple of (score, reasoning).
        """
        api_key = self._get_api_key()
        base64_image = self.image_to_base64(comparison_image)

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": self.system_prompt,
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Please evaluate how well the current Lego pile matches the goal shape. The left image shows the current state, and the right image shows the target shape.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}",
                            },
                        },
                    ],
                },
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        url = f"{self.base_url}/chat/completions"

        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()

                result = response.json()
                response_text = result["choices"][0]["message"]["content"]

                return self._parse_response(response_text)

            except requests.exceptions.RequestException as e:
                logger.warning(f"API request failed (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise

        return 0.5, "API call failed after all retries"

    def _save_vlm_input_image(self, image: np.ndarray, image_index: int = 0) -> None:
        """
        Save VLM input image for debugging.

        Args:
            image: The comparison image sent to VLM (H, W, 3), uint8.
            image_index: Index for distinguishing multiple VLM calls.
        """
        if not self.save_vlm_imgs:
            return

        os.makedirs(self.vis_output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"vlm_input_{timestamp}_{image_index}.png"
        filepath = os.path.join(self.vis_output_dir, filename)

        # Convert RGB to BGR for cv2
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filepath, image_bgr)
        logger.debug(f"Saved VLM input image to {filepath}")

    def compute(
        self,
        current_mask: np.ndarray,
        goal_mask: np.ndarray,
        image_index: int = 0,
    ) -> Dict[str, Any]:
        """
        Compute VLM perceptual score.

        Args:
            current_mask: Binary current mask (H, W), values 0/1.
            goal_mask: Binary goal mask (H, W), values 0/1.
            image_index: Index for distinguishing multiple VLM calls when saving images.

        Returns:
            Dictionary containing:
                - vlm_score: Perceptual score [0, 1]
                - reasoning: VLM's reasoning text
        """
        # Create comparison image
        comparison = self.create_comparison_image(current_mask, goal_mask)

        # Save VLM input image if debugging is enabled
        self._save_vlm_input_image(comparison, image_index)

        # Call API
        try:
            score, reasoning = self.call_api(comparison)
        except Exception as e:
            logger.error(f"VLM scoring failed: {e}")
            score = 0.5
            reasoning = f"VLM scoring failed: {str(e)}"

        return {
            "vlm_score": score,
            "reasoning": reasoning,
        }
