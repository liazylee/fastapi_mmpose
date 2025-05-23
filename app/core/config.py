from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    APP_NAME: str = "FastAPI MMPose"
    DEBUG: bool = True
    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str = "your-secret-key-here"
    DATABASE_URL: str = "sqlite:///./app.db"

    class Config:
        case_sensitive = True

settings = Settings() 