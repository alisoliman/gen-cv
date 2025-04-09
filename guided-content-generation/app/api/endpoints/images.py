from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, BackgroundTasks
from typing import Dict, List, Optional
import os

from app.models.images import (
    ImageGenerationRequest, 
    ImageGenerationResponse,
    VideoFromImageRequest,
    VideoFromImageResponse,
    ImageListRequest,
    ImageListResponse,
    ImageDeleteRequest,
    ImageDeleteResponse
)
from app.services.storage import StorageService
from app.services.image_generation import ImageGenerationService
from app.core.config import settings

router = APIRouter()
storage_service = StorageService()
image_service = ImageGenerationService()


@router.post("/generate", response_model=ImageGenerationResponse)
async def generate_image(request: ImageGenerationRequest):
    """Generate an image based on the provided prompt and settings"""
    try:
        # Get the appropriate system message
        system_message = await image_service.get_system_message(
            request.brand_protection,
            request.style,
            request.model,
            request.brands
        )
        
        # Validate and refine user prompt
        prompt_result = await image_service.refine_prompt(
            request.prompt, 
            system_message
        )
        
        # Check if prompt was filtered
        if prompt_result.get("filtered", False):
            return ImageGenerationResponse(
                success=False,
                message="Prompt filtered due to content policy",
                error="The prompt contains content that violates our content policy",
                prompt_filter_results=prompt_result.get("prompt_filter_results", {})
            )
        
        refined_prompt = prompt_result.get("refined_prompt", "")
        
        # Generate image
        image_result = await image_service.generate_image(
            refined_prompt, 
            request.model,
            {
                # Model-specific parameters
                "width": request.width,
                "height": request.height,
                "aspect_ratio": request.aspect_ratio,
                "quality": request.quality,
                "steps": request.steps,
                "seed": request.seed,
                "use_negative_prompt": request.use_negative_prompt
            }
        )
        
        # Perform content moderation
        moderation_results = await image_service.moderate_image(
            image_result["image_path"],
            {
                "Hate": request.hate_threshold,
                "SelfHarm": request.selfharm_threshold,
                "Sexual": request.sexual_threshold,
                "Violence": request.violence_threshold
            },
            use_llm_detection=request.use_llm_detection,
            use_vision_detection=request.use_vision_detection,
            vision_threshold=request.vision_threshold
        )
        
        return ImageGenerationResponse(
            success=True,
            message="Image generated successfully",
            image_id=image_result["image_id"],
            image_url=f"/static/images/{image_result['image_filename']}",
            refined_prompt=refined_prompt,
            content_safety_results=moderation_results["content_safety_results"],
            detected_brands=moderation_results.get("detected_brands", []),
            prompt_filter_results=prompt_result.get("prompt_filter_results", {})
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/image-to-video", response_model=VideoFromImageResponse)
async def convert_image_to_video(request: VideoFromImageRequest):
    """Convert an image to a video clip using Stable Diffusion 3"""
    try:
        if not settings.STABILITY_API_KEY:
            raise HTTPException(
                status_code=400, 
                detail="Stable Diffusion 3 API key is required for this operation"
            )
            
        video_result = await image_service.image_to_video(request.image_id)
        
        return VideoFromImageResponse(
            success=True,
            message="Image successfully converted to video",
            video_id=video_result["video_id"],
            video_url=f"/static/videos/generated/{video_result['video_filename']}"
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/list", response_model=ImageListResponse)
async def list_images(request: ImageListRequest):
    """List generated images with pagination"""
    try:
        # Get list of images from storage service
        images = storage_service.list_files(
            directory=settings.IMAGE_DIR,
            limit=request.limit,
            offset=request.offset
        )
        
        # Add URL to each image
        for image in images:
            image["url"] = f"/static/images/{image['filename']}"
        
        return ImageListResponse(
            success=True,
            images=images,
            total=len(images) + request.offset,  # This is not accurate for the total, just a placeholder
            limit=request.limit,
            offset=request.offset
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/delete", response_model=ImageDeleteResponse)
async def delete_image(request: ImageDeleteRequest):
    """Delete a generated image"""
    try:
        # Delete image using storage service
        success = storage_service.delete_file(
            file_id=request.image_id,
            directory=settings.IMAGE_DIR
        )
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Image with ID {request.image_id} not found")
        
        return ImageDeleteResponse(
            success=True,
            message=f"Image with ID {request.image_id} deleted successfully",
            image_id=request.image_id
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
