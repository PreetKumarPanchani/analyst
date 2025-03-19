# app/core/config.py
import os
from pathlib import Path
from pydantic_settings import BaseSettings
from typing import Dict, Any, Optional

class Settings(BaseSettings):
    """Application settings"""
    
    # API settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Sales Forecast"
    
    # CORS settings
    BACKEND_CORS_ORIGINS: list[str] = ["http://localhost:3000", "https://localhost:8001", 
                                      "https://mm2xymkp2i.eu-west-2.awsapprunner.com",
                                      "https://*.amplifyapp.com"]  # Added wildcard for Amplify domains
    
    # Data directories
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    RAW_DATA_DIR: str = os.path.join(BASE_DIR, "data", "raw")
    PROCESSED_DATA_DIR: str = os.path.join(BASE_DIR, "data", "processed")
    CACHE_DIR: str = os.path.join(BASE_DIR, "data", "cache")
    MODEL_DIR: str = os.path.join(BASE_DIR, "data", "models")
    
    # Cache settings
    CACHE_EXPIRY: int = 86400  # 24 hours in seconds
    
    # Weather API settings
    WEATHER_API_KEY: str = os.getenv("WEATHER_API_KEY", "")
    WEATHER_API_URL: str = "https://api.openweathermap.org/data/2.5"
    
    # TimeGPT API settings
    TIMEGPT_API_KEY: str = os.getenv("TIMEGPT_API_KEY", "")
    
    # AWS S3 settings
    USE_S3_STORAGE: bool = os.getenv("USE_S3_STORAGE", "false").lower() == "true"
    AWS_ACCESS_KEY_ID: Optional[str] = os.getenv("AWS_ACCESS_KEY_ID", "")
    AWS_SECRET_ACCESS_KEY: Optional[str] = os.getenv("AWS_SECRET_ACCESS_KEY", "")
    AWS_REGION: str = os.getenv("AWS_REGION", "eu-west-2")
    AWS_S3_BUCKET_NAME: str = os.getenv("AWS_S3_BUCKET_NAME", "sales-forecast-data-files")
    S3_RAW_PREFIX: str = "raw/"
    S3_PROCESSED_PREFIX: str = "processed/"
    S3_CACHE_PREFIX: str = "cache/"
    S3_MODEL_PREFIX: str = "models/"

    
    # Location settings
    LOCATION_COORDINATES: Dict[str, Dict[str, float]] = {
        "sheffield": {
            "lat": 53.383,
            "lon": -1.4659
        }
    }
    
    # Create directories on startup
    def create_directories(self):
        """Create needed directories"""
        if not self.USE_S3_STORAGE:
            os.makedirs(self.RAW_DATA_DIR, exist_ok=True)
            os.makedirs(self.PROCESSED_DATA_DIR, exist_ok=True)
            os.makedirs(self.CACHE_DIR, exist_ok=True)
            os.makedirs(self.MODEL_DIR, exist_ok=True)
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
settings.create_directories()











