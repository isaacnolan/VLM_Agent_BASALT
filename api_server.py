from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Union
import uvicorn
import anthropic
import json
import os
import logging
from dotenv import load_dotenv
from base64 import b64decode

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = FastAPI()

# Initialize Anthropic client
api_key = os.getenv('ANTHROPIC_API_KEY')
if not api_key:
    logger.error("ANTHROPIC_API_KEY environment variable not set")
    raise ValueError("ANTHROPIC_API_KEY environment variable not set")

client = anthropic.Anthropic(api_key=api_key)

# Log available models
try:
    logger.info("Checking available models...")
    models = client.models.list()
    for model in models.data:
        logger.info(f"Available model: {model.id}")
except Exception as e:
    logger.warning(f"Could not fetch models list: {e}")

class ImageData(BaseModel):
    media_type: str
    data: str  # base64 encoded image

class EvaluationRequest(BaseModel):
    task_name: str
    image: ImageData
    task_description: Optional[str] = None

@app.post("/evaluate")
async def evaluate_task(request: EvaluationRequest):
    logger.info(f"Received evaluation request for task: {request.task_name}")
    try:
        # Log the start of evaluation
        logger.info("Starting evaluation process")
        
        # Construct the evaluation prompt
        prompt = f"""Please evaluate this Minecraft {request.task_name} task result.
        Task Description: {request.task_description or 'Complete the specified Minecraft task'}
        
        Examine the final state image and rate it on a scale of 0-10 based on:
        1. Task Completion (0-4 points)
        2. Quality of Execution (0-3 points)
        3. Aesthetics and Integration (0-3 points)
        
        Provide:
        1. Numerical score (0-10)
        2. Brief explanation of the score
        Format the response as a JSON with 'score' and 'feedback' fields.
        """
        
        logger.info("Preparing message for Anthropic API")
        
        try:
            # Make the API call using Anthropic's client
            logger.info("Sending request to Anthropic API")
            message = client.messages.create(
                model="claude-3-5-haiku-latest",  # Using Claude 3 Opus
                max_tokens=1000,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": request.image.media_type,
                                "data": request.image.data
                            }
                        }
                    ]
                }],
                system="You are an expert Minecraft task evaluator. You will receive images of completed Minecraft tasks and provide numerical scores and feedback."
            )
        

        # Parse and return the response
            logger.info("Received response from Anthropic API")
            try:
                if hasattr(message, 'content'):
                    # New Anthropic API format
                    content = message.content[0].text
                    logger.info("Using new API response format")
                else:
                    # Older Anthropic API format
                    content = message.completion
                    logger.info("Using old API response format")
                
                logger.info(f"Raw response content: {content}")
                evaluation = json.loads(content)
                logger.info("Successfully parsed response as JSON")
                return evaluation
                
            except Exception as e:
                logger.error(f"Error parsing API response: {str(e)}")
                logger.error(f"Raw message object: {str(message)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to parse API response: {str(e)}\nFull message: {str(message)}"
                )
                
        except anthropic.APIError as e:
            logger.error(f"Anthropic API error: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Anthropic API error: {str(e)}"
            )
        except Exception as e:
            logger.error(f"Unexpected error during API call: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error during API call: {str(e)}"
            )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error during evaluation: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)