from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict
from app.models.common import BaseResponse, ContentModerationThresholds, ContentSafetyResult


class ImageGenerationRequest(BaseModel):
    """Request model for image generation"""
    prompt: str = Field(..., description="User prompt for image generation")
    model: str = Field(..., description="Image generation model to use")
    brand_protection: Literal["off", "neutralize", "replace"] = Field(
        "off", description="Brand protection strategy"
    )
    brands: Optional[str] = Field(None, description="Comma-separated brands to protect")
    style: str = Field("Photorealistic", description="Visual style for the image")
    
    # Model-specific parameters (will vary based on selected model)
    width: Optional[int] = None
    height: Optional[int] = None
    aspect_ratio: Optional[str] = None
    quality: Optional[bool] = None
    steps: Optional[int] = None
    seed: Optional[int] = None
    use_negative_prompt: Optional[bool] = None
    
    # Moderation thresholds
    hate_threshold: str = Field("low", description="Hate threshold severity")
    selfharm_threshold: str = Field("low", description="Self-harm threshold severity")
    sexual_threshold: str = Field("low", description="Sexual threshold severity")
    violence_threshold: str = Field("low", description="Violence threshold severity")
    
    # Brand detection options
    use_llm_detection: bool = Field(True, description="Use GPT-4o for brand detection")
    use_vision_detection: bool = Field(False, description="Use Azure Vision for brand detection")
    vision_threshold: Optional[float] = Field(0.6, description="Vision model detection threshold")


class ImageGenerationResponse(BaseResponse, ContentSafetyResult):
    """Response model for image generation"""
    image_id: Optional[str] = Field(None, description="Generated image ID")
    image_url: Optional[str] = Field(None, description="URL to the generated image")
    refined_prompt: Optional[str] = Field(None, description="Refined prompt used for generation")
    prompt_filter_results: Optional[dict] = Field(None, description="Content filter results for the prompt")


class VideoFromImageRequest(BaseModel):
    """Request model for converting an image to video"""
    image_id: str = Field(..., description="ID of the image to convert to video")


class VideoFromImageResponse(BaseResponse):
    """Response model for image to video conversion"""
    video_id: str = Field(..., description="Generated video ID")
    video_url: str = Field(..., description="URL to the generated video")


class ImageListRequest(BaseModel):
    """Request model for listing images"""
    limit: int = Field(20, description="Number of images to return")
    offset: int = Field(0, description="Offset for pagination")


class ImageListResponse(BaseResponse):
    """Response model for listing images"""
    images: List[dict] = Field(..., description="List of images")
    total: int = Field(..., description="Total number of images")
    limit: int = Field(..., description="Number of images per page")
    offset: int = Field(..., description="Offset for pagination")


class ImageDeleteRequest(BaseModel):
    """Request model for deleting an image"""
    image_id: str = Field(..., description="ID of the image to delete")


class ImageDeleteResponse(BaseResponse):
    """Response model for image deletion"""
    image_id: str = Field(..., description="ID of the deleted image")
