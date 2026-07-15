from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PORT: int = 8000
    MODEL_PATH: str = "../models"
    DATA_PATH: str = "../data"
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"

settings = Settings()
