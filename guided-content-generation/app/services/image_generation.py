import os
import base64
import uuid
import time
import json
import requests
import replicate
from typing import Dict, List, Optional, Any

from app.core.config import settings
from app.services.content_safety import ContentSafetyService
from app.services.constants import (
    BASIC_SYSTEM_MESSAGE,
    NEUTRALIZE_COMPETITORS_SYSTEM_MESSAGE,
    REPLACE_COMPETITORS_SYSTEM_MESSAGE,
    NEGATIVE_PROMPT,
    GPT4O_SYSTEM_MESSAGE,
    GPT4O_USER_PROMPT
)
from app.utils.helpers import (
    analyze_image_gpt4o,
    create_sdxl_image,
    check_and_reduce_image_size,
    dict_to_markdown_table,
    azure_image_analysis_predict,
    display_moderation_results
)


class ImageGenerationService:
    """Service for image generation and analysis"""

    def __init__(self):
        """Initialize the image generation service"""
        self.content_safety = ContentSafetyService()
        self.severity_to_id = {"safe": 0, "low": 2, "med": 4, "high": 6}
        self.id_to_severity = {0: "safe", 2: "low", 4: "med", 6: "high"}

        # Ensure directories exist
        os.makedirs(settings.IMAGE_DIR, exist_ok=True)
        os.makedirs(os.path.join(settings.VIDEO_DIR,
                    "generated"), exist_ok=True)

    async def get_system_message(self, brand_protection: str, style: str, model: str, brands: Optional[str] = None) -> str:
        """
        Get the appropriate system message based on brand protection settings

        Args:
            brand_protection: Brand protection strategy ('off', 'neutralize', 'replace')
            style: Visual style for the image
            model: Image generation model to use
            brands: Comma-separated brands to protect

        Returns:
            System message for prompt refinement
        """
        if brand_protection == "off":
            return BASIC_SYSTEM_MESSAGE.format(style=style, model=model)
        elif brand_protection == "neutralize":
            return NEUTRALIZE_COMPETITORS_SYSTEM_MESSAGE.format(style=style, model=model, brands=brands)
        elif brand_protection == "replace":
            return REPLACE_COMPETITORS_SYSTEM_MESSAGE.format(style=style, model=model, brands=brands)
        else:
            return BASIC_SYSTEM_MESSAGE.format(style=style, model=model)

    async def refine_prompt(self, prompt: str, system_message: str) -> Dict:
        """
        Refine the user prompt using the specified system message

        Args:
            prompt: Original user prompt
            system_message: System message for prompt refinement

        Returns:
            Dictionary with refined prompt and content filter results
        """
        messages = [
            {'role': 'system', 'content': system_message},
            {'role': 'user', 'content': prompt}
        ]

        # Configure OpenAI client for Azure OpenAI
        import openai
        client = openai.AzureOpenAI(
            api_key=settings.AOAI_KEY,
            api_version=settings.AOAI_API_VERSION,
            azure_endpoint=settings.AOAI_ENDPOINT
        )

        try:
            print(f"Using model: {settings.PROMPT_MODERATION_DEPLOYMENT}")
            print(f"API Version: {settings.AOAI_API_VERSION}")

            response = client.chat.completions.create(
                model=settings.PROMPT_MODERATION_DEPLOYMENT,
                messages=messages,
                temperature=0,
                max_tokens=200
            )

            # Print the full response structure for debugging
            print(f"Response type: {type(response)}")
            print(f"Response attributes: {dir(response)}")

            # Print the response JSON to see where content filtering might be located
            try:
                response_json = json.loads(response.model_dump_json())
                print(f"Response JSON: {json.dumps(response_json, indent=2)}")
            except Exception as dump_error:
                print(f"Error dumping response to JSON: {dump_error}")

            # Initialize empty prompt filter results (newer API versions might not include this)
            prompt_filter_results = {}

            # Check for different ways content filtering might be reported
            if hasattr(response, 'prompt_filter_results'):
                # Original way (older API versions)
                prompt_filter = response.prompt_filter_results[0]['content_filter_results']
                # Convert to the same format as the original implementation
                for category, data in prompt_filter.items():
                    prompt_filter_results[category] = {
                        'filtered': data['filtered'],
                        'severity': self.id_to_severity.get(data.get('severity', 0), 'safe')
                    }
            elif hasattr(response, 'choices') and len(response.choices) > 0 and hasattr(response.choices[0], 'content_filter_results'):
                # Newer structure (might be in choices)
                prompt_filter = response.choices[0].content_filter_results
                # Convert to our standard format
                for category, data in prompt_filter.items():
                    prompt_filter_results[category] = {
                        'filtered': data.get('filtered', False),
                        'severity': self.id_to_severity.get(data.get('severity', 0), 'safe')
                    }
            elif 'response_json' in locals():
                # Try to extract from the JSON dump (newer API versions might have different structure)
                print("Attempting to extract filter info from response_json")
                # Try checking in different locations where the filter might be found

                # Check in top-level attributes
                filter_locations = [
                    'content_filter_results',
                    'filter_results',
                    'safety_ratings',
                    'moderation_results'
                ]

                for location in filter_locations:
                    if location in response_json:
                        print(f"Found filter info in response_json.{location}")
                        try:
                            filter_data = response_json[location]
                            for category, data in filter_data.items():
                                if isinstance(data, dict) and ('filtered' in data or 'severity' in data):
                                    prompt_filter_results[category] = {
                                        'filtered': data.get('filtered', False),
                                        'severity': self.id_to_severity.get(data.get('severity', 0), 'safe')
                                    }
                        except Exception as e:
                            print(f"Error extracting from {location}: {e}")

                # Check in choices
                if 'choices' in response_json and len(response_json['choices']) > 0:
                    choice = response_json['choices'][0]
                    for location in filter_locations:
                        if location in choice:
                            print(
                                f"Found filter info in response_json.choices[0].{location}")
                            try:
                                filter_data = choice[location]
                                for category, data in filter_data.items():
                                    if isinstance(data, dict) and ('filtered' in data or 'severity' in data):
                                        prompt_filter_results[category] = {
                                            'filtered': data.get('filtered', False),
                                            'severity': self.id_to_severity.get(data.get('severity', 0), 'safe')
                                        }
                            except Exception as e:
                                print(
                                    f"Error extracting from choices.{location}: {e}")

            # If still empty, provide default structure
            if not prompt_filter_results:
                print(
                    "Warning: No prompt filtering information available in API response")
                for category in ['hate', 'sexual', 'violence', 'self_harm']:
                    prompt_filter_results[category] = {
                        'filtered': False,
                        'severity': 'safe'
                    }

            # Return both the refined prompt and content filter results
            return {
                "refined_prompt": response.choices[0].message.content,
                "prompt_filter_results": prompt_filter_results,
                "filtered": False
            }

        except Exception as e:
            # Handle content filter exceptions
            error_string = str(e)
            print(f"Exception in refine_prompt: {error_string}")

            if hasattr(e, 'code') and e.code == 'content_filter':
                # Extract content filter results from the error
                prompt_filter_results = {}
                if hasattr(e, 'body') and 'innererror' in e.body:
                    prompt_filter = e.body['innererror']['content_filter_result']

                    # Convert to the same format as the original implementation
                    for category, data in prompt_filter.items():
                        prompt_filter_results[category] = {
                            'filtered': data['filtered'],
                            'severity': self.id_to_severity.get(data.get('severity', 0), 'safe')
                        }
                elif 'content_filter_result' in error_string:
                    # Try to parse the error string if it contains filter info
                    print("Attempting to parse content filter info from error string")
                    try:
                        # This is a basic attempt and may need refinement
                        import re
                        import json
                        match = re.search(
                            r'content_filter_result["\']:\s*(\{[^}]+\})', error_string)
                        if match:
                            filter_json = match.group(1).replace("'", '"')
                            filter_data = json.loads(filter_json)
                            for category, data in filter_data.items():
                                prompt_filter_results[category] = {
                                    'filtered': data.get('filtered', True),
                                    'severity': self.id_to_severity.get(data.get('severity', 6), 'high')
                                }
                    except Exception as parse_error:
                        print(
                            f"Error parsing filter info from string: {parse_error}")

                return {
                    "refined_prompt": None,
                    "prompt_filter_results": prompt_filter_results,
                    "filtered": True,
                    "error": "Prompt filtered due to content policy"
                }
            else:
                raise ValueError(f"Error refining prompt: {str(e)}")

    async def generate_image(self, prompt: str, model_name: str, params: Dict = {}) -> Dict:
        """
        Generate an image based on the refined prompt

        Args:
            prompt: Refined prompt for image generation
            model_name: Image generation model to use
            params: Additional parameters for generation

        Returns:
            Dictionary with generation results
        """
        generated_image = None
        caption = prompt  # Default caption is the prompt itself

        # Generate a unique ID for the image
        image_id = str(uuid.uuid4())
        filename = f"{image_id}.png"
        image_path = os.path.join(settings.IMAGE_DIR, filename)

        # FLUX.1 [pro] generation
        if model_name == "FLUX.1 [pro]":
            if not settings.REPLICATE_API_KEY:
                raise ValueError(
                    "REPLICATE_API_KEY is required for FLUX.1 [pro]")

            input_data = {
                "prompt": prompt,
                "aspect_ratio": params.get("aspect_ratio", "1:1"),
                "safety_tolerance": 5,
                "steps": params.get("steps", 25),
                "Guidance": params.get("guidance", 3),
                "Interval": params.get("interval", 2),
            }

            api = replicate.Client(api_token=settings.REPLICATE_API_KEY)
            output = api.run("black-forest-labs/flux-pro", input=input_data)
            image_url = output

            # Download the image
            response = requests.get(image_url)
            if response.status_code == 200:
                generated_image = response.content
            else:
                raise ValueError(
                    f"Failed to download image: {response.status_code}")

        # DALL-E 3 generation
        elif model_name == "DALL E-3":
            # Configure OpenAI client for Azure OpenAI
            import openai
            client = openai.AzureOpenAI(
                api_key=settings.AOAI_KEY,
                api_version=settings.AOAI_API_VERSION,
                azure_endpoint=settings.AOAI_ENDPOINT
            )

            quality = params.get("quality", True)
            style = params.get("style", True)
            size = params.get("size", "1024x1024")

            quality_param = 'hd' if quality else 'standard'
            style_param = 'vivid' if style else 'natural'

            try:
                result = client.images.generate(
                    model=settings.DALLE_DEPLOYMENT,
                    prompt=prompt,
                    n=1,
                    quality=quality_param,
                    size=size,
                    style=style_param
                )

                json_response = json.loads(result.model_dump_json())
                dalle_revised_prompt = json_response.get(
                    'data', [{}])[0].get('revised_prompt', None)
                caption = dalle_revised_prompt if dalle_revised_prompt else prompt

                image_url = json_response["data"][0]["url"]
                response = requests.get(image_url)
                if response.status_code == 200:
                    generated_image = response.content
                else:
                    raise ValueError(
                        f"Failed to download image: {response.status_code}")

            except openai.BadRequestError as e:
                # Handle content filter errors specifically
                error_msg = str(e)
                if "content_policy_violation" in error_msg:
                    raise ValueError(f"Content policy violation: {error_msg}")
                raise ValueError(
                    f"Error generating image with DALL-E 3: {error_msg}")
            except Exception as e:
                raise ValueError(
                    f"Error generating image with DALL-E 3: {str(e)}")

        # Stable Diffusion XL generation
        elif model_name == "Stable Diffusion XL":
            if not settings.AML_IMGEN_API_KEY or not settings.AML_IMGEN_ONLINE_ENDPOINT_URL:
                raise ValueError(
                    "Azure ML credentials not configured for SDXL")

            input_data = {
                'prompt': prompt,
                'negative_prompt': NEGATIVE_PROMPT if params.get("use_negative_prompt", True) else None,
                'width': params.get("width", 1024),
                'height': params.get("height", 1024),
                'n_steps': params.get("steps", 50),
                'high_noise_frac': 0.7,
                'seed': params.get("seed", None)
            }

            try:
                response = await create_sdxl_image(
                    prompt=prompt,
                    params=input_data
                )

                if "error" in response:
                    raise ValueError(
                        f"Error generating image with SDXL: {response['error']}")

                generated_image = base64.b64decode(response["image"])
                image_path = response["image_path"]
                filename = response["image_filename"]
                image_id = response["image_id"]

            except Exception as e:
                raise ValueError(f"Error generating image with SDXL: {str(e)}")

        # Stable Diffusion 3 generation
        elif model_name == "Stable Diffusion 3":
            if not settings.STABILITY_API_KEY:
                raise ValueError(
                    "STABILITY_API_KEY is required for Stable Diffusion 3")

            sd3_model = params.get("model", "sd3-large")
            sd3_aspect_ratio = params.get("aspect_ratio", "1:1")

            try:
                response = requests.post(
                    f"https://api.stability.ai/v2beta/stable-image/generate/sd3",
                    headers={
                        "authorization": f"Bearer {settings.STABILITY_API_KEY}",
                        "accept": "image/*"
                    },
                    files={
                        "prompt": (None, prompt),
                        "model": (None, sd3_model),
                        "aspect_ratio": (None, sd3_aspect_ratio),
                        "output_format": (None, "png")
                    }
                )

                if response.status_code == 200:
                    generated_image = response.content
                else:
                    raise ValueError(
                        f"Error generating image with SD3: {str(response.json())}")

            except Exception as e:
                raise ValueError(f"Error generating image with SD3: {str(e)}")

        else:
            raise ValueError(f"Unsupported model: {model_name}")

        # Save the generated image
        if generated_image:
            with open(image_path, "wb") as image_file:
                image_file.write(generated_image)
        else:
            raise ValueError("Failed to generate image")

        return {
            "image_id": image_id,
            "image_path": image_path,
            "image_filename": filename,
            "caption": caption
        }

    async def moderate_image(self, image_path: str, thresholds: Dict[str, str], use_llm_detection: bool = True, use_vision_detection: bool = False, vision_threshold: float = 0.6) -> Dict:
        """
        Moderate the generated image for content safety

        Args:
            image_path: Path to the image file
            thresholds: Dictionary of category -> severity threshold
            use_llm_detection: Whether to use GPT-4o for brand detection
            use_vision_detection: Whether to use Azure Vision for brand detection
            vision_threshold: Vision model detection threshold

        Returns:
            Dictionary with moderation results
        """
        # Convert thresholds to numeric values
        numeric_thresholds = {
            category: self.severity_to_id.get(severity.lower(), 2)
            for category, severity in thresholds.items()
        }

        # Check image for harmful content
        content_safety_image_path = check_and_reduce_image_size(image_path)

        # Analyze image with Content Safety API
        content_safety_results = await self.content_safety.analyze_image(
            content_safety_image_path,
            thresholds
        )

        # Detect brands using GPT-4o
        detected_brands = []
        brand_analysis_gpt = {}
        if use_llm_detection:
            try:
                # Use the same prompt as in the original implementation
                response = await analyze_image_gpt4o(
                    image_path=image_path,
                    prompt=GPT4O_USER_PROMPT
                )

                # Extract brands from the response
                if "brands" in response and isinstance(response["brands"], list):
                    detected_brands.extend(response["brands"])
                brand_analysis_gpt = response
            except Exception as e:
                print(f"Error detecting brands with GPT-4o: {str(e)}")
                brand_analysis_gpt = {"error": str(e)}

        # Detect brands using Azure Vision (if configured)
        brand_analysis_vision = {}
        if use_vision_detection:
            try:
                # Call the Azure Vision analysis function
                vision_results = await azure_image_analysis_predict(
                    image_path=image_path,
                    vision_threshold=vision_threshold
                )

                # Add detected brands to the list
                if "brands" in vision_results and isinstance(vision_results["brands"], list):
                    # Add brands that aren't already in the list
                    for brand in vision_results["brands"]:
                        if brand not in detected_brands:
                            detected_brands.append(brand)
                brand_analysis_vision = vision_results
            except Exception as e:
                print(f"Error detecting brands with Azure Vision: {str(e)}")
                brand_analysis_vision = {"error": str(e)}

        return {
            "content_safety_results": content_safety_results.get("content_safety_results", {}),
            "detected_brands": detected_brands,
            "brand_analysis_gpt": brand_analysis_gpt,
            "brand_analysis_vision": brand_analysis_vision
        }

    async def image_to_video(self, image_id: str) -> Dict:
        """
        Convert an image to a video clip using Stable Diffusion 3

        Args:
            image_id: ID of the image to convert

        Returns:
            Dictionary with video generation results
        """
        if not settings.STABILITY_API_KEY:
            raise ValueError(
                "STABILITY_API_KEY is required for image to video conversion")

        # Get the image path
        image_dir = settings.IMAGE_DIR
        image_filename = f"{image_id}.png"
        source_img_path = os.path.join(image_dir, image_filename)

        if not os.path.exists(source_img_path):
            # Try to find the file by ID prefix
            for file in os.listdir(image_dir):
                if file.startswith(image_id):
                    image_filename = file
                    source_img_path = os.path.join(image_dir, file)
                    break
            else:
                raise FileNotFoundError(f"Image with ID {image_id} not found")

        # Create video target folder
        video_target_folder = os.path.join(settings.VIDEO_DIR, "generated")
        os.makedirs(video_target_folder, exist_ok=True)

        # Generate a unique ID for the video
        video_id = str(uuid.uuid4())
        video_filename = f"{video_id}.mp4"
        video_path = os.path.join(video_target_folder, video_filename)

        # Resize image (placeholder for actual implementation)
        resized_filename = f"resized-{image_filename}"
        temp_image_path = os.path.join(video_target_folder, resized_filename)

        # Copy the image to the video folder
        import shutil
        shutil.copy(source_img_path, temp_image_path)

        # Call Stability API for image to video conversion
        # This is a placeholder for the actual implementation
        # In a real implementation, we would call the Stability API

        # For now, we'll just create a dummy video file
        with open(video_path, "wb") as video_file:
            video_file.write(b"dummy video content")

        return {
            "video_id": video_id,
            "video_path": video_path,
            "video_filename": video_filename
        }
