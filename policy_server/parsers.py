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
                logger.warning(f"No JSON found in VLM response: {response_text[:200]}")
                raise ValueError("No JSON found in response")

            raw_json = json_match.group()
            logger.info(f"Raw JSON extracted from VLM:\n{raw_json}")

            # Clean up common non-JSON tokens
            clean = VLMResponseParser._clean_json(raw_json)
            logger.info(f"Cleaned JSON to parse:\n{clean}")

            response_json = json.loads(clean)
            logger.info(f"Parsed JSON object: {response_json}")

            # Extract action and reasoning
            action_dict = response_json.get('action', {})
            reasoning = response_json.get('reasoning', 'No reasoning provided')
            
            logger.info(f"Extracted action_dict: {action_dict}")
            logger.info(f"Extracted reasoning: {reasoning}")

            # Create MineRL action format
            minerl_action = create_minerl_action(action_dict)
            
            logger.info(f"Final MineRL action: {minerl_action}")

            return minerl_action, reasoning

        except Exception as e:
            logger.error(f"Error parsing VLM response: {e}")
            logger.error(f"Full response text:\n{response_text}")
            
            # Return a safe default action
            default_action = get_default_minerl_action()
            logger.warning(f"Returning default action due to parse error: {default_action}")
            return default_action, f"Error parsing response: {str(e)}"
    
    @staticmethod
    def _clean_json(raw_json: str) -> str:
        """Clean up common non-JSON tokens."""
        clean = raw_json
        
        # ONLY replace "0 or 1" if it appears AFTER a colon (in a value position)
        # AND is followed by a comma or closing brace (not in documentation)
        # This prevents converting template text like '"forward": 0 or 1' when the VLM copies the template
        # Only match when it's clearly a malformed value, not when all fields have it
        
        # Count how many "0 or 1" or "1 or 0" appear - if too many, VLM probably copied the template
        or_count = len(re.findall(r':\s*(?:0\s*or\s*1|1\s*or\s*0)', clean))
        total_fields = len(re.findall(r'"[^"]+"\s*:', clean))
        
        # If more than 50% of fields have "or" syntax, the VLM likely copied the template literally
        # In this case, DON'T auto-convert - return an error instead
        if total_fields > 0 and or_count / total_fields > 0.5:
            logger.error(f"VLM appears to have copied the template literally ({or_count}/{total_fields} fields have 'or' syntax)")
            raise ValueError("VLM returned template format instead of concrete values")
        
        # Only clean up occasional "or" syntax (when VLM made a mistake on a few fields)
        clean = re.sub(r'(?P<key>"[a-zA-Z0-9_.]+"\s*:\s*)(?:0\s*or\s*1|1\s*or\s*0)(?=\s*[,}])', r"\g<key>0", clean)
        
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
