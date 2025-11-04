import requests
import base64
import json

def evaluate_minecraft_task(image_path, task_name):
    # Read and encode the image
    with open(image_path, 'rb') as image_file:
        image_data = base64.b64encode(image_file.read()).decode('utf-8')
    
    # Determine media type from file extension
    media_type = 'image/png' if image_path.endswith('.png') else 'image/jpeg'
    
    # Prepare the request payload
    payload = {
        "task_name": task_name,
        "image": {
            "media_type": media_type,
            "data": image_data
        }
    }
    
    # Make request to the local API server
    try:
        response = requests.post(
            "http://localhost:8000/evaluate",
            json=payload
        )
        response.raise_for_status()  # Raise exception for bad status codes
        
        # Parse and return the results
        result = response.json()
        print(f"Score: {result['score']}")
        print(f"Feedback: {result['feedback']}")
        return result
        
    except requests.exceptions.RequestException as e:
        print(f"Error calling API: {e}")
        return None

if __name__ == "__main__":
    # Example usage
    task_name = "MakeWaterfall"
    image_path = "Final_Output_Images/Good_waterfall.png"
    
    print(f"Evaluating {task_name} task...")
    result = evaluate_minecraft_task(image_path, task_name)
    if result:
        print("\nFull response:")
        print(json.dumps(result, indent=2))