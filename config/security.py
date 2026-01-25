"""
Security configuration and utilities for the application.
Handles environment variables, secrets, and security settings.
"""
import os
import secrets
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


class SecurityConfig:
    """Centralized security configuration"""
    
    def __init__(self):
        # API Keys - Never expose these
        self._google_api_key = os.getenv("GOOGLE_API_KEY")
        
        # Application secrets
        self._app_secret_key = os.getenv("APP_SECRET_KEY", secrets.token_hex(32))
        
        # Default admin credentials (should be changed immediately)
        self._default_admin_username = os.getenv("DEFAULT_ADMIN_USERNAME", "admin")
        self._default_admin_password = os.getenv("DEFAULT_ADMIN_PASSWORD", "change_me_immediately")
        
        # Server configuration
        self.server_host = os.getenv("SERVER_HOST", "0.0.0.0")
        self.server_port = int(os.getenv("SERVER_PORT", "7860"))
        self.debug_mode = os.getenv("DEBUG_MODE", "False").lower() == "true"
        
        # Security settings
        self.session_timeout_minutes = int(os.getenv("SESSION_TIMEOUT_MINUTES", "30"))
        self.max_login_attempts = int(os.getenv("MAX_LOGIN_ATTEMPTS", "5"))
        self.rate_limit_per_minute = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
        
        # Data directory
        self.data_dir = os.getenv("DATA_DIR", "./data")
        
        # Vector store configuration
        self.vector_store_path = os.getenv("VECTOR_STORE_PATH", "./vector_store_google")
        self.cache_ttl_hours = int(os.getenv("CACHE_TTL_HOURS", "24"))
        
        # Validate critical settings
        self._validate_config()
    
    def _validate_config(self):
        """Validate critical configuration"""
        if not self._google_api_key:
            raise ValueError(
                "GOOGLE_API_KEY is not set. Please set it in your .env file. "
                "See .env.example for reference."
            )
        
        if self._default_admin_password == "change_me_immediately":
            import warnings
            warnings.warn(
                "WARNING: Default admin password is still set to default value. "
                "Please change DEFAULT_ADMIN_PASSWORD in your .env file immediately!",
                UserWarning
            )
    
    @property
    def google_api_key(self) -> str:
        """Get Google API key (read-only)"""
        if not self._google_api_key:
            raise ValueError("Google API key is not configured")
        return self._google_api_key
    
    @property
    def app_secret_key(self) -> str:
        """Get application secret key (read-only)"""
        return self._app_secret_key
    
    @property
    def default_admin_username(self) -> str:
        """Get default admin username"""
        return self._default_admin_username
    
    @property
    def default_admin_password(self) -> str:
        """Get default admin password"""
        return self._default_admin_password
    
    def get_safe_config(self) -> dict:
        """Get configuration without sensitive data for logging"""
        return {
            "server_host": self.server_host,
            "server_port": self.server_port,
            "debug_mode": self.debug_mode,
            "data_dir": self.data_dir,
            "vector_store_path": self.vector_store_path,
            "session_timeout_minutes": self.session_timeout_minutes,
            "max_login_attempts": self.max_login_attempts,
            "rate_limit_per_minute": self.rate_limit_per_minute,
            "google_api_key_configured": bool(self._google_api_key),
        }


# Global security config instance
security_config = SecurityConfig()
