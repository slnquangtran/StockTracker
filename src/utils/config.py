"""Configuration management module."""
import os
import yaml
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, Any

# Load environment variables
load_dotenv()

class Config:
    """Application configuration manager."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize configuration.
        
        Args:
            config_path: Path to configuration YAML file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self._load_env_vars()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Warning: Config file {self.config_path} not found. Using defaults.")
            return self._get_default_config()
    
    def _load_env_vars(self):
        """Load sensitive data from environment variables."""
        self.news_api_key = os.getenv('NEWS_API_KEY', '')
        self.smtp_email = os.getenv('SMTP_EMAIL', '')
        self.smtp_password = os.getenv('SMTP_PASSWORD', '')
        self.twilio_sid = os.getenv('TWILIO_ACCOUNT_SID', '')
        self.twilio_token = os.getenv('TWILIO_AUTH_TOKEN', '')
        self.twilio_phone = os.getenv('TWILIO_PHONE_NUMBER', '')
        self.alert_phone = os.getenv('ALERT_PHONE_NUMBER', '')
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            'portfolio': {'default_name': 'My Portfolio', 'data_file': 'data/portfolio.json'},
            'data': {'cache_duration': 300, 'default_period': '1y', 'default_interval': '1d'},
            'indicators': {
                'sma_periods': [20, 50, 200],
                'rsi_period': 14,
                'macd': {'fast_period': 12, 'slow_period': 26, 'signal_period': 9}
            },
            'lstm': {'sequence_length': 60, 'epochs': 50, 'batch_size': 32},
            'alerts': {'email': {'enabled': True}, 'sms': {'enabled': False}},
            'dashboard': {'refresh_interval': 60, 'theme': 'dark'}
        }
    
    def get(self, key: str, default=None):
        """Get configuration value by key path (e.g., 'portfolio.default_name')."""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
        return value if value is not None else default
    
    def has_news_api_key(self) -> bool:
        """Check if NewsAPI key is configured."""
        return bool(self.news_api_key)
    
    def has_email_config(self) -> bool:
        """Check if email configuration is complete."""
        return bool(self.smtp_email and self.smtp_password)
    
    def has_sms_config(self) -> bool:
        """Check if SMS configuration is complete."""
        return bool(self.twilio_sid and self.twilio_token and self.twilio_phone)

# Global config instance
config = Config()
