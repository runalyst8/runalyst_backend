import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    # Database
    DATABASE_URL: str
    
    # JWT Configuration
    JWT_SECRET: str
    JWT_ALG: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    PASSWORD_RESET_TOKEN_EXPIRE_MINUTES: int = 15
    SECRET_KEY: str = "secretkey"

    # Email (Resend)
    RESEND_API_KEY: str = ""
    RESEND_FROM_EMAIL: str = "noreply@runalyst.com"

    # App Configuration
    APP_ENV: str = "development"
    DEBUG: bool = False
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"  # Ignore extra fields in .env
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Validate critical settings
        if self.APP_ENV == "production" and self.JWT_SECRET == "dev-secret":
            raise ValueError(
                "JWT_SECRET must be set to a secure value in production! "
                "Never use 'dev-secret' in production."
            )
        if not self.DATABASE_URL:
            raise ValueError("DATABASE_URL must be set")


settings = Settings()