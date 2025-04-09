from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime


class BaseResponse(BaseModel):
    """Base response model with common fields"""
    success: bool = Field(True, description="Whether the request was successful")
    message: Optional[str] = Field(None, description="Response message")
    error: Optional[str] = Field(None, description="Error message if any")


class PaginatedResponse(BaseResponse):
    """Base model for paginated responses"""
    total: int = Field(..., description="Total number of items")
    limit: int = Field(..., description="Number of items per page")
    offset: int = Field(..., description="Offset for pagination")
    items: List[Any] = Field(..., description="List of items")


class ContentModerationThresholds(BaseModel):
    """Content moderation thresholds"""
    hate_threshold: str = Field("low", description="Hate threshold severity")
    selfharm_threshold: str = Field("low", description="Self-harm threshold severity")
    sexual_threshold: str = Field("low", description="Sexual threshold severity")
    violence_threshold: str = Field("low", description="Violence threshold severity")


class ContentSafetyResult(BaseModel):
    """Content safety analysis result"""
    content_safety_results: Dict = Field({}, description="Content moderation results")
    detected_brands: Optional[List[str]] = Field(None, description="Detected brands in the content")


class FileInfo(BaseModel):
    """File information"""
    file_id: str = Field(..., description="Unique file ID")
    filename: str = Field(..., description="Original filename")
    file_path: str = Field(..., description="Path to the file")
    size: int = Field(..., description="File size in bytes")
    created_at: float = Field(..., description="Creation timestamp")
    modified_at: float = Field(..., description="Last modification timestamp")
    url: Optional[str] = Field(None, description="URL to access the file")
