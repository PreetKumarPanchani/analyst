# app/core/config.py
import os
from pathlib import Path
from pydantic_settings import BaseSettings
from typing import Dict, Any

class Settings(BaseSettings):
    """Application settings"""
    
    # API settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Sales Forecast"
    
    # CORS settings
    BACKEND_CORS_ORIGINS: list[str] = ["http://localhost:3000", "https://localhost:8001", "http://localhost", "https://localhost"]
    
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
    
    # Location settings
    LOCATION_COORDINATES: Dict[str, Dict[str, float]] = {
        "sheffield": {
            "lat": 53.3811,
            "lon": -1.4701
        }
    }
    
    # Create directories on startup
    def create_directories(self):
        """Create needed directories"""
        os.makedirs(self.RAW_DATA_DIR, exist_ok=True)
        os.makedirs(self.PROCESSED_DATA_DIR, exist_ok=True)
        os.makedirs(self.CACHE_DIR, exist_ok=True)
        os.makedirs(self.MODEL_DIR, exist_ok=True)
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
settings.create_directories()











