from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from app.models.common import BaseResponse, ContentModerationThresholds, ContentSafetyResult


class VideoAnalysisRequest(BaseModel):
    """Request model for video analysis"""
    video_id: str = Field(..., description="ID of an existing video")
    frames_per_scene: int = Field(2, description="Number of frames to extract per scene")
    drop_similar_frames: bool = Field(True, description="Whether to drop similar frames")
    frame_similarity_threshold: int = Field(20, description="Threshold for frame similarity")
    transcribe_audio: bool = Field(True, description="Whether to transcribe audio")
    
    # Moderation thresholds
    hate_threshold: str = Field("low", description="Hate threshold severity")
    selfharm_threshold: str = Field("low", description="Self-harm threshold severity")
    sexual_threshold: str = Field("low", description="Sexual threshold severity")
    violence_threshold: str = Field("low", description="Violence threshold severity")


class VideoAnalysisResponse(BaseResponse, ContentSafetyResult):
    """Response model for video analysis"""
    video_id: str = Field(..., description="Video ID")
    scenes_count: int = Field(..., description="Number of detected scenes")
    frames_count: int = Field(..., description="Number of extracted frames")
    unique_frames_count: Optional[int] = Field(None, description="Number of unique frames after filtering")
    transcription: Optional[str] = Field(None, description="Video transcription")
    insights: Dict = Field(..., description="AI-generated insights about the video")


class VideoUploadResponse(BaseResponse):
    """Response model for video upload"""
    video_id: str = Field(..., description="Uploaded video ID")
    filename: str = Field(..., description="Original filename")


class VideoListRequest(BaseModel):
    """Request model for listing videos"""
    limit: int = Field(20, description="Number of videos to return")
    offset: int = Field(0, description="Offset for pagination")


class VideoListResponse(BaseResponse):
    """Response model for listing videos"""
    videos: List[dict] = Field(..., description="List of videos")
    total: int = Field(..., description="Total number of videos")
    limit: int = Field(..., description="Number of videos per page")
    offset: int = Field(..., description="Offset for pagination")


class VideoDeleteRequest(BaseModel):
    """Request model for deleting a video"""
    video_id: str = Field(..., description="ID of the video to delete")


class VideoDeleteResponse(BaseResponse):
    """Response model for video deletion"""
    video_id: str = Field(..., description="ID of the deleted video")
