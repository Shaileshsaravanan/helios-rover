import google.generativeai as genai
import os
import base64
import json

# Set up API key
api_key = os.getenv("API_KEY")
genai.configure(api_key=api_key)

# Initialize the model
model = genai.GenerativeModel('gemini-1.5-flash')

def process_image(base64_image):
    """
    Process a base64-encoded image to determine the plant state and calculate percentage yield.
    """
    # Decode the base64 image
    image_data = base64.b64decode(base64_image)
    
    # Convert binary data to a format suitable for the API
    image_base64_str = base64.b64encode(image_data).decode('utf-8')
    
    # Create the prompt for the Gemini model
    prompt = {
        "image": {
            "base64": image_base64_str
        },
        "query": "Determine the state of the plant and calculate its percentage yield."
    }

    # Call the model with the prompt
    response = model.generate_content(prompt)
    result = response.text

    # Parse the result (assuming result is JSON-formatted text)
    try:
        result_data = json.loads(result)
        plant_state = result_data.get('state', 'Unknown')
        percentage_yield = result_data.get('percentage_yield', 0.0)
    except json.JSONDecodeError:
        plant_state = "Error in response"
        percentage_yield = 0.0

    return plant_state, percentage_yield

def process_images(base64_images):
    """
    Process a list of base64-encoded images.
    """
    results = []
    for base64_image in base64_images:
        state, yield_percentage = process_image(base64_image)
        results.append({
            "state": state,
            "percentage_yield": yield_percentage
        })
    return results

# Example usage
if __name__ == "__main__":
    # Example base64 images list
    base64_images = [
        "<base64_image_1>",
        "<base64_image_2>",
        "<base64_image_3>"
    ]

    results = process_images(base64_images)
    for idx, result in enumerate(results):
        print(f"Image {idx+1}: State - {result['state']}, Percentage Yield - {result['percentage_yield']:.2f}%")
