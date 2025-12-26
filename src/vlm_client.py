"""
VLM Client for OpenRouter API integration.
Handles GPT-4o scoring with Chain-of-Thought prompting.
"""

import os
import json
import base64
import time
import logging
from typing import Dict, Any, Optional, Tuple

import numpy as np
import cv2
import requests


logger = logging.getLogger(__name__)


class VLMClient:
    """
    VLM Client for perceptual scoring using OpenRouter API.
    Uses GPT-4o with CoT prompting to evaluate shape matching quality.
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

        # Goal rendering config (for creating comparison image)
        semantic_cfg = config.get("metrics", {}).get("semantic", {}).get("dinov2", {})
        render_cfg = semantic_cfg.get("goal_render", {})
        self.fg_color = np.array(render_cfg.get("foreground_color", [200, 0, 0]))
        self.bg_color = np.array(render_cfg.get("background_color", [200, 200, 200]))

    def _default_system_prompt(self) -> str:
        """Return default system prompt."""
        return """You are a strict robot manipulation judge evaluating a Lego sweeping task.
1. Compare the current pile of Legos with the goal shape.
2. Analyze geometric alignment, disconnected parts, and scattered debris. i.e. Identify specific defects: Are there disconnected parts? Is the shape too thin? Is there extra debris outside?
3. Output a final score from 0.0 to 1.0 in JSON format: {"reasoning": "...", "score": 0.0}."""

    def _get_api_key(self) -> str:
        """Get API key from environment variable."""
        api_key = os.environ.get(self.api_key_env_var)
        if not api_key:
            raise ValueError(
                f"API key not found. Please set the {self.api_key_env_var} environment variable."
            )
        return api_key

    def render_goal_image(self, goal_mask: np.ndarray) -> np.ndarray:
        """
        Render binary goal mask as RGB image.

        Args:
            goal_mask: Binary goal mask (H, W), values 0/1.

        Returns:
            RGB image (H, W, 3), uint8.
        """
        h, w = goal_mask.shape
        rendered = np.zeros((h, w, 3), dtype=np.uint8)
        rendered[:, :] = self.bg_color
        rendered[goal_mask > 0] = self.fg_color
        return rendered

    def create_comparison_image(
        self,
        current_image: np.ndarray,
        goal_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Create side-by-side comparison image for VLM input.

        Args:
            current_image: Current RGB image (H, W, 3), uint8.
            goal_mask: Binary goal mask (H, W), values 0/1.

        Returns:
            Concatenated comparison image (H, 2W, 3), uint8.
        """
        # Render goal mask
        goal_image = self.render_goal_image(goal_mask)

        # Ensure same size
        h1, w1 = current_image.shape[:2]
        h2, w2 = goal_image.shape[:2]

        if h1 != h2 or w1 != w2:
            goal_image = cv2.resize(goal_image, (w1, h1))

        # Add labels
        current_labeled = current_image.copy()
        goal_labeled = goal_image.copy()

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(current_labeled, "Current", (10, 25), font, 0.7, (255, 255, 255), 2)
        cv2.putText(goal_labeled, "Goal", (10, 25), font, 0.7, (255, 255, 255), 2)

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

    def compute(
        self,
        current_image: np.ndarray,
        goal_mask: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Compute VLM perceptual score.

        Args:
            current_image: Current RGB image (H, W, 3), uint8.
            goal_mask: Binary goal mask (H, W), values 0/1.

        Returns:
            Dictionary containing:
                - vlm_score: Perceptual score [0, 1]
                - reasoning: VLM's reasoning text
        """
        # Create comparison image
        comparison = self.create_comparison_image(current_image, goal_mask)

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
