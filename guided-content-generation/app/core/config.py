from pydantic_settings import BaseSettings
from typing import List, Optional


class Settings(BaseSettings):
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Guided Content Generation API"
    
    # Azure OpenAI
    AOAI_ENDPOINT: str
    AOAI_KEY: str
    GPT_DEPLOYMENT: str
    PROMPT_MODERATION_DEPLOYMENT: str
    DALLE_DEPLOYMENT: str
    
    # Azure OpenAI API Version
    AOAI_API_VERSION: str = "2024-05-01-preview"
    
    # Azure OpenAI Whisper (Sweden Central)
    AOAI_ENDPOINT_SWECE: str
    AOAI_KEY_SWECE: str 
    WHISPER_DEPLOYMENT: str
    
    # Content Safety
    CONTENT_SAFETY_ENDPOINT: str
    CONTENT_SAFETY_KEY: str
    
    # Optional: Image Generation APIs
    STABILITY_API_KEY: Optional[str] = None
    REPLICATE_API_KEY: Optional[str] = None
    
    # Optional: Azure ML for SDXL
    AML_IMGEN_API_KEY: Optional[str] = None
    AML_IMGEN_ONLINE_ENDPOINT_URL: Optional[str] = None
    AML_DEPLOYMENT_NAME: Optional[str] = None
    
    # Optional: Azure AI Vision
    AZURE_AI_VISION_ENDPOINT: Optional[str] = None
    AZURE_AI_VISION_KEY: Optional[str] = None
    AZURE_AI_VISION_DEPLOYMENT: Optional[str] = None
    
    # File storage
    UPLOAD_DIR: str = "./static/uploads"
    IMAGE_DIR: str = "./static/images"
    VIDEO_DIR: str = "./static/videos"
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
