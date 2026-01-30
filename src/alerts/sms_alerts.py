"""SMS alert system using Twilio."""
from typing import Optional
from datetime import datetime
from src.utils.config import config

try:
    from twilio.rest import Client
    TWILIO_AVAILABLE = True
except ImportError:
    TWILIO_AVAILABLE = False
    print("Warning: Twilio not available. Install with: pip install twilio")

class SMSAlerts:
    """Send SMS notifications using Twilio."""
    
    def __init__(self):
        """Initialize SMS alert system."""
        self.account_sid = config.twilio_sid
        self.auth_token = config.twilio_token
        self.from_number = config.twilio_phone
        self.to_number = config.alert_phone
        
        if TWILIO_AVAILABLE and self.is_configured():
            self.client = Client(self.account_sid, self.auth_token)
        else:
            self.client = None
    
    def send_sms(self, message: str, to_number: str = None) -> bool:
        """Send an SMS message.
        
        Args:
            message: Message text
            to_number: Recipient phone number (defaults to configured number)
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_configured():
            print("SMS not configured. Set Twilio credentials in .env")
            return False
        
        if not TWILIO_AVAILABLE:
            print("Twilio library not installed")
            return False
        
        to_number = to_number or self.to_number
        
        try:
            message = self.client.messages.create(
                body=message,
                from_=self.from_number,
                to=to_number
            )
            return True
        except Exception as e:
            print(f"Error sending SMS: {e}")
            return False
    
    def send_price_alert(self, ticker: str, current_price: float,
                        threshold_price: float, alert_type: str) -> bool:
        """Send price alert via SMS.
        
        Args:
            ticker: Stock ticker
            current_price: Current price
            threshold_price: Threshold price
            alert_type: 'above' or 'below'
            
        Returns:
            True if successful
        """
        message = (f"🚨 PRICE ALERT: {ticker}\n"
                  f"Current: ${current_price:.2f}\n"
                  f"Threshold: ${threshold_price:.2f}\n"
                  f"Price went {alert_type} threshold")
        
        return self.send_sms(message)
    
    def send_critical_alert(self, ticker: str, alert_message: str) -> bool:
        """Send critical alert via SMS.
        
        Args:
            ticker: Stock ticker
            alert_message: Alert message
            
        Returns:
            True if successful
        """
        message = f"⚠️ CRITICAL: {ticker}\n{alert_message}"
        return self.send_sms(message)
    
    def is_configured(self) -> bool:
        """Check if SMS is configured.
        
        Returns:
            True if configured
        """
        return config.has_sms_config()
