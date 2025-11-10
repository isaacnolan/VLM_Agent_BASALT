"""Image processing utilities."""
import base64
import io
import logging
from PIL import Image
from typing import List
from fastapi import HTTPException
from common.models import StateActionPair, ImageData
from common.actions import summarize_action

logger = logging.getLogger(__name__)


class ImageProcessor:
    """Handles image decoding and processing."""
    
    @staticmethod
    def decode_image(image_data: ImageData) -> Image.Image:
        """
        Decode a base64-encoded image.
        
        Args:
            image_data: ImageData with base64 encoded string
            
        Returns:
            PIL Image
            
        Raises:
            HTTPException: If decoding fails
        """
        try:
            image_bytes = base64.b64decode(image_data.data)
            image = Image.open(io.BytesIO(image_bytes))
            return image
        except Exception as e:
            logger.error(f"Failed to decode image: {e}")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid image data: {str(e)}"
            )
    
    @staticmethod
    def process_history(history: List[StateActionPair], max_len: int, 
                       current_step: int) -> tuple:
        """
        Process history of state-action pairs.
        
        Args:
            history: List of state-action pairs
            max_len: Maximum number of frames to process
            current_step: Current step number
            
        Returns:
            Tuple of (images_list, history_text)
        """
        images = []
        history_text = ""
        
        # Limit history length
        history_subset = history[-max_len:]
        
        # Decode all historical images
        for idx, state_action in enumerate(history_subset):
            try:
                img = ImageProcessor.decode_image(state_action.image)
                images.append(img)
                
                # Build history description
                if state_action.action:
                    # Summarize the action taken
                    action_summary = summarize_action(state_action.action)
                    history_text += f"\nStep {current_step - len(history_subset) + idx}: {action_summary}"
                else:
                    history_text += f"\nStep {current_step - len(history_subset) + idx}: (initial state)"
                    
            except HTTPException as e:
                # Re-raise HTTP exceptions
                raise
            except Exception as e:
                logger.error(f"Failed to decode history image {idx}: {e}")
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid image data in history index {idx}: {str(e)}"
                )
        
        logger.info(f"Decoded {len(images)} images from history")
        return images, history_text
