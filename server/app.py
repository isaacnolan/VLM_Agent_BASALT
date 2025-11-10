"""
FastAPI Server for QWEN VLM Policy Function

This server provides an API endpoint for getting Minecraft actions from the QWEN VLM model.
It processes visual observations and returns actions in MineRL format.
"""

from fastapi import FastAPI, HTTPException
import logging
from common.models import PolicyRequest, ActionResponse
from server.model_loader import ModelLoader
from server.prompt_builder import PromptBuilder
from server.parsers import VLMResponseParser
from server.image_processor import ImageProcessor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="QWEN VLM Policy Server", version="1.0.0")

# Initialize components
model_loader = ModelLoader(model_name="Qwen/Qwen2-VL-7B-Instruct")
prompt_builder = PromptBuilder()
parser = VLMResponseParser()
image_processor = ImageProcessor()


@app.on_event("startup")
async def startup():
    """Load the QWEN VLM model on server startup."""
    await model_loader.load()


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "online",
        "service": "QWEN VLM Policy Server",
        "model_loaded": model_loader.is_loaded(),
        "device": model_loader.device
    }


@app.get("/health")
async def health_check():
    """Detailed health check."""
    return {
        "status": "healthy" if model_loader.is_loaded() else "unhealthy",
        "model": model_loader.model_name,
        "device": model_loader.device,
        "ready": model_loader.is_loaded()
    }


@app.post("/get_action", response_model=ActionResponse)
async def get_action(request: PolicyRequest):
    """
    Get an action from the QWEN VLM policy given an observation and optional history.
    
    Args:
        request: PolicyRequest containing task name, current image or history of state-action pairs
        
    Returns:
        ActionResponse with the predicted action and reasoning
    """
    if not model_loader.is_loaded():
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please wait for server startup to complete."
        )
    
    logger.info(f"Received policy request for task: {request.task_name}, step: {request.step}")
    
    try:
        # Process history or single image
        images = []
        history_text = ""
        
        if request.history:
            # Use history mode
            logger.info(f"Processing history with {len(request.history)} state-action pairs")
            max_len = request.max_history_length or 5
            images, history_text = image_processor.process_history(
                request.history, max_len, request.step
            )
            
        elif request.image:
            # Backward compatibility: single image mode
            logger.info("Processing single image (no history)")
            image = image_processor.decode_image(request.image)
            images = [image]
            logger.info(f"Decoded image: {image.size}, mode: {image.mode}")
        else:
            raise HTTPException(
                status_code=400,
                detail="Either 'image' or 'history' must be provided"
            )
        
        # Build prompt
        text_prompt = prompt_builder.build_text_prompt(
            request.task_name, 
            request.step, 
            history_text if history_text else None
        )
        
        # Build messages for VLM
        messages = prompt_builder.build_messages(
            request.task_name,
            images,
            text_prompt
        )
        
        # Generate response from VLM
        logger.info(f"Generating action from VLM (with {len(images)} image(s))...")
        response_text = await model_loader.generate(
            messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        
        logger.info(f"VLM Response: {response_text[:200]}...")
        
        # Parse the response
        action_dict, reasoning = parser.parse_single_step(response_text)
        
        return ActionResponse(
            action=action_dict,
            reasoning=reasoning,
            step=request.step
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during action generation: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error during action generation: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
