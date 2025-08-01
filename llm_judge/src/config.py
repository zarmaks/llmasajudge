"""Configuration management for LLM-as-a-Judge."""

import os
from pathlib import Path
from typing import Optional

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv is optional


class Config:
    """Configuration settings for the application."""
    
    # API Configuration
    MISTRAL_API_KEY: Optional[str] = os.getenv("MISTRAL_API_KEY")
    DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "mistral-large-latest")
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.0"))
    
    # Output Configuration
    REPORTS_DIR: Path = Path(os.getenv("REPORTS_DIR", "reports"))
    
    # Safety Configuration
    ENABLE_SAFETY_GATE: bool = os.getenv("ENABLE_SAFETY_GATE", "true").lower() == "true"
    
    @classmethod
    def validate(cls) -> None:
        """Validate configuration settings."""
        if not cls.MISTRAL_API_KEY:
            raise ValueError(
                "MISTRAL_API_KEY environment variable not set. "
                "Please check your .env file or environment."
            )
        
        if cls.TEMPERATURE < 0.0 or cls.TEMPERATURE > 2.0:
            raise ValueError("TEMPERATURE must be between 0.0 and 2.0")


# Global config instance
config = Config()
