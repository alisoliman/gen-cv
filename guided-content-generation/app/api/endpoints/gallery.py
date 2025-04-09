from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, List, Optional

from app.models.common import BaseResponse, PaginatedResponse
from app.services.storage import StorageService
from app.core.config import settings

router = APIRouter()
storage_service = StorageService()


@router.get("/", response_model=PaginatedResponse)
async def get_gallery_items(limit: int = 20, offset: int = 0):
    """Get all gallery items (images and videos) with pagination"""
    try:
        # Get images
        images = storage_service.list_files(
            directory=settings.IMAGE_DIR,
            limit=limit,
            offset=offset
        )
        
        # Add URL and type to each image
        for image in images:
            image["url"] = f"/static/images/{image['filename']}"
            image["type"] = "image"
        
        # In a real implementation, we would also get videos and combine them
        # This is a placeholder implementation
        
        return PaginatedResponse(
            success=True,
            message="Gallery items retrieved successfully",
            total=len(images),
            limit=limit,
            offset=offset,
            items=images
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/images", response_model=PaginatedResponse)
async def get_gallery_images(limit: int = 20, offset: int = 0):
    """Get all gallery images with pagination"""
    try:
        # Get images
        images = storage_service.list_files(
            directory=settings.IMAGE_DIR,
            limit=limit,
            offset=offset
        )
        
        # Add URL and type to each image
        for image in images:
            image["url"] = f"/static/images/{image['filename']}"
            image["type"] = "image"
        
        return PaginatedResponse(
            success=True,
            message="Gallery images retrieved successfully",
            total=len(images),
            limit=limit,
            offset=offset,
            items=images
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/videos", response_model=PaginatedResponse)
async def get_gallery_videos(limit: int = 20, offset: int = 0):
    """Get all gallery videos with pagination"""
    try:
        # Get videos
        videos = storage_service.list_files(
            directory=settings.VIDEO_DIR,
            limit=limit,
            offset=offset
        )
        
        # Add URL and type to each video
        for video in videos:
            video["url"] = f"/static/videos/{video['filename']}"
            video["type"] = "video"
        
        return PaginatedResponse(
            success=True,
            message="Gallery videos retrieved successfully",
            total=len(videos),
            limit=limit,
            offset=offset,
            items=videos
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
