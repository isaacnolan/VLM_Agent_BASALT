"""Parse VLM responses into actions."""
import json
import re
import logging
from typing import Dict, Any, Tuple
from common.actions import create_minerl_action, get_default_minerl_action, get_exploration_action

logger = logging.getLogger(__name__)


class VLMResponseParser:
    """Handles parsing of VLM responses."""
    
    @staticmethod
    def parse_single_step(response_text: str) -> Tuple[Dict[str, Any], str]:
        """
        Parse single-step VLM response.
        
        Args:
            response_text: Text response from the VLM
            
        Returns:
            Tuple of (action_dict, reasoning)
        """
        try:
            # Try to extract JSON from the response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if not json_match:
                raise ValueError("No JSON found in response")

            raw_json = json_match.group()
            logger.info(f"Raw JSON extracted from VLM:\n{raw_json}")

            # Clean up common non-JSON tokens
            clean = VLMResponseParser._clean_json(raw_json)
            logger.info(f"Cleaned JSON to parse:\n{clean}")

            response_json = json.loads(clean)

            # Extract action and reasoning
            action_dict = response_json.get('action', {})
            reasoning = response_json.get('reasoning', 'No reasoning provided')

            # Create MineRL action format
            minerl_action = create_minerl_action(action_dict)

            return minerl_action, reasoning

        except Exception as e:
            logger.error(f"Error parsing VLM response: {e}")
            logger.error(f"Response text: {response_text}")
            
            # Return a safe default action
            return get_default_minerl_action(), f"Error parsing response: {str(e)}"
    
    @staticmethod
    def _clean_json(raw_json: str) -> str:
        """Clean up common non-JSON tokens."""
        clean = raw_json
        # Replace constructs like: "forward": 0 or 1  -> choose 1
        clean = re.sub(r'(?P<key>"[a-zA-Z0-9_.]+"\s*:\s*)(?:0\s*or\s*1|1\s*or\s*0)', r"\g<key>1", clean)
        # Replace boolean-like words (True/False) with JSON booleans
        clean = re.sub(r"\bTrue\b", "true", clean)
        clean = re.sub(r"\bFalse\b", "false", clean)
        # Replace single quotes with double quotes
        clean = clean.replace("\'", '"')
        # Remove trailing commas before closing braces/brackets
        clean = re.sub(r",\s*(\}|\])", r"\1", clean)
        # Remove duplicate commas
        clean = re.sub(r",\s*,", ",", clean)
        # Fix missing commas between properties
        clean = re.sub(r'([0-9\]\}])\s*\n\s*"', r'\1,\n"', clean)
        # Remove comments
        clean = re.sub(r'//.*$', '', clean, flags=re.MULTILINE)
        return clean
