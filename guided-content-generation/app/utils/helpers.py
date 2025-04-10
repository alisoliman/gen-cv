import os
import base64
import json
from typing import Dict, List, Any, Optional
from PIL import Image
import io
import httpx
from openai import AzureOpenAI

from app.core.config import settings
from app.services.constants import GPT4O_SYSTEM_MESSAGE


def dict_to_markdown_table(data: Dict[str, Any]) -> str:
    """
    Convert a dictionary to a markdown table

    Args:
        data: Dictionary to convert

    Returns:
        Markdown table as string
    """
    if not data:
        return ""

    # Create header
    markdown = "| Key | Value |\n| --- | --- |\n"

    # Add rows
    for key, value in data.items():
        # Handle nested dictionaries
        if isinstance(value, dict):
            nested_value = "<br>".join([f"{k}: {v}" for k, v in value.items()])
            markdown += f"| {key} | {nested_value} |\n"
        # Handle lists
        elif isinstance(value, list):
            list_value = "<br>".join([str(item) for item in value])
            markdown += f"| {key} | {list_value} |\n"
        # Handle other types
        else:
            markdown += f"| {key} | {value} |\n"

    return markdown


def display_moderation_results(results: Dict[str, Dict]) -> str:
    """
    Format content moderation results for display, matching the format in the Streamlit app

    Args:
        results: Dictionary of moderation results

    Returns:
        Formatted string with HTML styling for highlighting filtered content
    """
    display_text = ""
    for category, info in results.items():
        if category == "jailbreak":
            if info.get('filtered', False):
                display_text += f"<span style='color:red;'>jailbreak attempt</span>, "
            else:
                display_text += f"jailbreak: not detected, "
        else:
            if info.get('filtered', False):
                display_text += f"{category}: <span style='color:red;'>{info.get('severity', 'unknown')}</span>, "
            else:
                display_text += f"{category}: {info.get('severity', 'unknown')}, "

    # Remove trailing comma and space
    display_text = display_text.rstrip(", ")
    return display_text


async def analyze_image_gpt4o(image_path: str, prompt: str) -> Dict:
    """
    Analyze an image using GPT-4o Vision

    Args:
        image_path: Path to the image file
        prompt: Prompt to send to GPT-4o

    Returns:
        Dictionary containing analysis results
    """
    # Read and encode image
    try:
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")
    except Exception as e:
        print(f"Error reading image file: {str(e)}")
        return {
            "error": f"Error reading image file: {str(e)}",
            "brands": []
        }

    # Use the system message from our constants
    system_message = GPT4O_SYSTEM_MESSAGE

    # Use the standard user prompt if none provided
    if not prompt or prompt.strip() == "":
        prompt = "List all brands and product names found in this image:"

    try:
        # Try using the OpenAI client (Azure) instead of raw HTTP
        client = AzureOpenAI(
            api_key=settings.AOAI_KEY,
            api_version=settings.AOAI_API_VERSION,
            azure_endpoint=settings.AOAI_ENDPOINT
        )

        print(f"Using GPT deployment: {settings.GPT_DEPLOYMENT}")
        print(f"API Version: {settings.AOAI_API_VERSION}")
        print(f"Endpoint: {settings.AOAI_ENDPOINT}")

        response = client.chat.completions.create(
            model=settings.GPT_DEPLOYMENT,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": "auto"
                    }}
                ]}
            ],
            max_tokens=150,
            temperature=0,
            seed=0
        )

        # Extract the response content
        analysis = response.choices[0].message.content
        print(f"GPT-4o response: {analysis}")

        # Process the analysis to extract brand names
        if "no brands found" in analysis.lower():
            brands = []
        else:
            # Extract brands from the response
            if "found" in analysis.lower():
                # Extract the part after "Found "
                brands_text = analysis.split("Found ", 1)[1].strip()
                # Split by commas and clean up
                brands = [brand.strip()
                          for brand in brands_text.split(",") if brand.strip()]
            else:
                brands = []

        return {
            "analysis": analysis,
            "brands": brands,
            # Convert to string as response object may not be serializable
            "raw_response": str(response)
        }

    except Exception as e:
        # Enhanced error handling with more detailed logs
        error_msg = f"Error analyzing image with GPT-4o: {str(e)}"
        print(error_msg)
        print(f"Debug info - GPT deployment: {settings.GPT_DEPLOYMENT}")
        print(f"Debug info - API Version: {settings.AOAI_API_VERSION}")

        # Try direct REST API as fallback
        try:
            print("Attempting direct REST API call as fallback...")

            headers = {
                "Content-Type": "application/json",
                "api-key": settings.AOAI_KEY
            }

            # Ensure the endpoint URL has a trailing slash for proper concatenation
            endpoint = settings.AOAI_ENDPOINT
            if not endpoint.endswith('/'):
                endpoint += '/'

            url = f"{endpoint}openai/deployments/{settings.GPT_DEPLOYMENT}/chat/completions?api-version={settings.AOAI_API_VERSION}"
            print(f"Using URL: {url}")

            payload = {
                "model": settings.GPT_DEPLOYMENT,
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "auto"
                            }
                        }
                    ]}
                ],
                "temperature": 0,
                "max_tokens": 150
            }

            # Make the request with httpx directly
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(url, headers=headers, json=payload)

                if response.status_code != 200:
                    error_msg = f"Error code: {response.status_code} - {response.text}"
                    print(error_msg)
                    return {
                        "error": error_msg,
                        "brands": []
                    }

                result = response.json()

                analysis = result["choices"][0]["message"]["content"]
                print(f"GPT-4o response (via REST API): {analysis}")

                # Process the analysis to extract brand names
                if "no brands found" in analysis.lower():
                    brands = []
                else:
                    # Extract brands from the response
                    if "found" in analysis.lower():
                        # Extract the part after "Found "
                        brands_text = analysis.split("Found ", 1)[1].strip()
                        # Split by commas and clean up
                        brands = [brand.strip() for brand in brands_text.split(
                            ",") if brand.strip()]
                    else:
                        brands = []

                return {
                    "analysis": analysis,
                    "brands": brands,
                    "raw_response": result
                }
        except Exception as nested_e:
            err_msg = f"Both attempts at image analysis failed. Original error: {str(e)}, Fallback error: {str(nested_e)}"
            print(err_msg)
            return {
                "error": err_msg,
                "brands": []
            }


def check_and_reduce_image_size(image_path: str, max_size_mb: float = 4.0) -> str:
    """
    Check if an image is too large and reduce its size if necessary

    Args:
        image_path: Path to the image file
        max_size_mb: Maximum size in MB

    Returns:
        Path to the resized image (same as input if no resize was needed)
    """
    # Get file size in MB
    file_size_mb = os.path.getsize(image_path) / (1024 * 1024)

    # If file is smaller than max size, return original path
    if file_size_mb <= max_size_mb:
        return image_path

    # Open the image
    img = Image.open(image_path)

    # Calculate new dimensions (reduce by 25% at a time)
    width, height = img.size
    quality = 95

    # Create a new filename for the resized image
    filename, ext = os.path.splitext(image_path)
    resized_path = f"{filename}_resized{ext}"

    # Reduce size until it's small enough
    while file_size_mb > max_size_mb and (width > 100 or height > 100):
        # Reduce dimensions by 25%
        width = int(width * 0.75)
        height = int(height * 0.75)

        # Resize the image
        resized_img = img.resize((width, height), Image.LANCZOS)

        # Save with reduced quality
        resized_img.save(resized_path, quality=quality)

        # Check new file size
        file_size_mb = os.path.getsize(resized_path) / (1024 * 1024)

        # If still too large, reduce quality
        if file_size_mb > max_size_mb:
            quality -= 5

    return resized_path


async def create_sdxl_image(prompt: str, params: Dict = None) -> Dict:
    """
    Create an image using Stable Diffusion XL via Azure ML

    Args:
        prompt: Prompt for image generation
        params: Additional parameters for generation

    Returns:
        Dictionary with generation results
    """
    if not settings.AML_IMGEN_API_KEY or not settings.AML_IMGEN_ONLINE_ENDPOINT_URL:
        return {"error": "Azure ML credentials not configured"}

    # Default parameters
    default_params = {
        "prompt": prompt,
        "negative_prompt": "",
        "num_inference_steps": 50,
        "guidance_scale": 7.5,
        "width": 1024,
        "height": 1024,
        "seed": None
    }

    # Update with provided parameters
    if params:
        default_params.update(params)

    # Prepare headers
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {settings.AML_IMGEN_API_KEY}",
        "azureml-model-deployment": settings.AML_DEPLOYMENT_NAME
    }

    # Make API request
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            settings.AML_IMGEN_ONLINE_ENDPOINT_URL,
            headers=headers,
            json=default_params
        )

        if response.status_code != 200:
            return {
                "error": f"API request failed with status {response.status_code}",
                "details": response.text
            }

        result = response.json()

        # Decode base64 image
        image_data = base64.b64decode(result["image"])

        # Generate a unique filename
        import uuid
        filename = f"{uuid.uuid4()}.png"
        file_path = os.path.join(settings.IMAGE_DIR, filename)

        # Save the image
        with open(file_path, "wb") as f:
            f.write(image_data)

        return {
            "image_path": file_path,
            "image_filename": filename,
            "image_id": filename.split(".")[0]
        }


async def azure_image_analysis_predict(image_path: str, vision_threshold: float = 0.6) -> Dict:
    """
    Analyze an image using Azure AI Vision for brand detection

    Args:
        image_path: Path to the image file
        vision_threshold: Confidence threshold for detections

    Returns:
        Dictionary containing analysis results with detected brands
    """
    # Check if Azure Vision is configured
    if not settings.AZURE_AI_VISION_ENDPOINT or not settings.AZURE_AI_VISION_KEY:
        return {
            "error": "Azure AI Vision not configured",
            "brands": []
        }

    try:
        # Ensure the endpoint URL has a trailing slash for proper concatenation
        endpoint = settings.AZURE_AI_VISION_ENDPOINT
        if not endpoint.endswith('/'):
            endpoint += '/'

        # Prepare API URL - use the model name from settings
        url = f"{endpoint}computervision/imageanalysis:analyze?model-name={settings.AZURE_AI_VISION_MODEL}&api-version=2023-02-01-preview"

        # Read image data
        with open(image_path, 'rb') as file:
            data = file.read()

        # Set up headers
        headers = {
            'Ocp-Apim-Subscription-Key': settings.AZURE_AI_VISION_KEY,
            'Content-Type': 'application/octet-stream'
        }

        # Make API request
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                url,
                headers=headers,
                content=data
            )

        if response.status_code == 200:
            result = response.json()

            # Extract brands from the response
            brands = []
            if 'customModelResult' in result and 'objectsResult' in result['customModelResult']:
                objects = result['customModelResult']['objectsResult']['values']
                brands = [obj['tags'][0]['name'] for obj in objects
                          if 'tags' in obj and len(obj['tags']) > 0
                          and obj['tags'][0]['confidence'] > vision_threshold]

            return {
                "raw_response": result,
                "brands": brands
            }
        else:
            return {
                "error": f"API request failed with status {response.status_code}",
                "details": response.text,
                "brands": []
            }
    except Exception as e:
        print(f"Error analyzing image with Azure AI Vision: {str(e)}")
        return {
            "error": f"Error analyzing image with Azure AI Vision: {str(e)}",
            "brands": []
        }
