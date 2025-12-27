"""
Configuration management for Football Analytics Platform
"""
from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    # Project paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    RAW_DATA_DIR: Path = PROJECT_ROOT / "dataset 3" / "data"
    PROCESSED_DATA_DIR: Path = PROJECT_ROOT / "data" / "processed"
    MODELS_DIR: Path = PROJECT_ROOT / "models"
    REPORTS_DIR: Path = PROJECT_ROOT / "reports"
    LOGS_DIR: Path = PROJECT_ROOT / "logs"
    
    # Selected competitions (2022-2024 with 360° data)
    SELECTED_COMPETITIONS: dict = {
        "FIFA World Cup": ["2022"],
        "Ligue 1": ["2022/2023", "2021/2022"],
        "1. Bundesliga": ["2023/2024"],  # Only 360° tracking season
        "UEFA Euro": ["2024"],
        "Major League Soccer": ["2023"]
    }
    
    # Database
    DATABASE_URL: str = "sqlite:///database/football_analytics.db"
    
    # ML Config
    RANDOM_STATE: int = 42
    TEST_SIZE: float = 0.15
    VALIDATION_SIZE: float = 0.15
    
    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    
    # Dashboard  
    DASHBOARD_PORT: int = 8501
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "allow"  # Allow extra fields from .env


# Global settings instance
settings = Settings()


# Ensure directories exist
for dir_path in [settings.DATA_DIR, settings.PROCESSED_DATA_DIR, 
                 settings.MODELS_DIR, settings.REPORTS_DIR, settings.LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)
