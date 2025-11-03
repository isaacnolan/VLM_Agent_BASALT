import requests
from PIL import Image
import base64
import io
import json
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def encode_image_base64(image_path):
    """
    Encode image to base64 string and determine its media type.
    """
    # Get file extension
    file_ext = image_path.lower().split('.')[-1]
    
    # Map file extension to media type
    media_types = {
        'jpg': 'image/jpeg',
        'jpeg': 'image/jpeg',
        'png': 'image/png'
    }
    media_type = media_types.get(file_ext, 'image/jpeg')  # default to jpeg if unknown
    
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8'), media_type

def evaluate_model(task_name, final_state_image_path):
    """
    Evaluates the final state of a Minecraft task using Claude API.
    
    Args:
        task_name: Name of the Minecraft task (e.g., 'MakeWaterfall', 'FindCave', etc.)
        final_state_image_path: Path to the screenshot of final Minecraft state
        
    Returns:
        Dictionary containing the evaluation score (0-10) and feedback
    """
    # Get API key from environment
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")
    
    # Encode image to base64 and get media type
    image_base64, media_type = encode_image_base64(final_state_image_path)
    
    # Construct prompt for Claude
    task_descriptions = {
        'MakeWaterfall': 'Create an aesthetically pleasing waterfall in Minecraft',
        'FindCave': 'Locate and identify a cave entrance in Minecraft',
        'CreateVillageAnimalPen': 'Build a functional and safe animal pen in a village',
        'BuildVillageHouse': 'Construct a house that fits the village aesthetic'
    }
    
    prompt = f"""Please evaluate this Minecraft {task_name} task result.
    Task Description: {task_descriptions.get(task_name, 'Complete the specified Minecraft task')}
    
    Examine the final state image and rate it on a scale of 0-10 based on:
    1. Task Completion (0-4 points)
    2. Quality of Execution (0-3 points)
    3. Aesthetics and Integration (0-3 points)
    
    Provide:
    1. Numerical score (0-10)
    2. Brief explanation of the score
    Format the response as a JSON with 'score' and 'feedback' fields.
    """
    
    # Prepare the API request
    headers = {
        'anthropic-version': '2023-06-01',
        'content-type': 'application/json',
        'x-api-key': api_key,
        'anthropic-beta': 'messages-2023-12-15',
    }
    
    data = {
        'model': 'claude-2.1',
        'max_tokens': 1000,
        'messages': [{
            'role': 'user',
            'content': [
                {
                    'type': 'text',
                    'text': prompt
                },
                {
                    'type': 'image',
                    'source': {
                        'type': 'base64',
                        'media_type': media_type,
                        'data': image_base64
                    }
                }
            ]
        }]
    }
    
    # Call Claude API directly using requests
    try:
        response = requests.post(
            'https://api.anthropic.com/v1/messages',
            headers=headers,
            json={
                'model': 'claude-3-opus-20240229',
                'max_tokens': 1000,
                'messages': [{
                    'role': 'user',
                    'content': [
                        {
                            'type': 'text',
                            'text': prompt
                        },
                        {
                            'type': 'image',
                            'source': {
                                'type': 'base64',
                                'media_type': media_type,
                                'data': image_base64
                            }
                        }
                    ]
                }]
            }
        )
        
        if response.status_code == 200:
            response_data = response.json()
            evaluation = json.loads(response_data['completion'])
            return evaluation
        else:
            return {
                "score": 0,
                "feedback": f"API Error: {response.status_code} - {response.text}"
            }
    except Exception as e:
        return {
            "score": 0,
            "feedback": f"Error during evaluation: {str(e)}"
        }
if __name__ == "__main__":
    # Example usage
    task_name = "MakeWaterfall"
    image_path = "Final_Output_Images/Good_waterfall.png"  # Example path
    
    evaluation_results = evaluate_model(task_name, image_path)
    print("Evaluation Results:", json.dumps(evaluation_results, indent=2))