"""Email alert system."""
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import List, Dict, Any
from datetime import datetime
from src.utils.config import config

class EmailAlerts:
    """Send email notifications for stock alerts."""
    
    def __init__(self):
        """Initialize email alert system."""
        self.smtp_email = config.smtp_email
        self.smtp_password = config.smtp_password
        self.smtp_server = "smtp.gmail.com"
        self.smtp_port = 587
    
    def send_email(self, subject: str, body: str, to_email: str = None) -> bool:
        """Send an email.
        
        Args:
            subject: Email subject
            body: Email body (can be HTML)
            to_email: Recipient email (defaults to sender)
            
        Returns:
            True if successful, False otherwise
        """
        if not config.has_email_config():
            print("Email not configured. Set SMTP_EMAIL and SMTP_PASSWORD in .env")
            return False
        
        to_email = to_email or self.smtp_email
        
        try:
            msg = MIMEMultipart('alternative')
            msg['From'] = self.smtp_email
            msg['To'] = to_email
            msg['Subject'] = subject
            
            # Add body
            html_part = MIMEText(body, 'html')
            msg.attach(html_part)
            
            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_email, self.smtp_password)
                server.send_message(msg)
            
            return True
        except Exception as e:
            print(f"Error sending email: {e}")
            return False
    
    def send_price_alert(self, ticker: str, current_price: float, 
                        threshold_price: float, alert_type: str) -> bool:
        """Send price threshold alert.
        
        Args:
            ticker: Stock ticker
            current_price: Current stock price
            threshold_price: Threshold price that was crossed
            alert_type: 'above' or 'below'
            
        Returns:
            True if successful
        """
        subject = f"🚨 Price Alert: {ticker}"
        
        body = f"""
        <html>
        <body style="font-family: Arial, sans-serif;">
            <h2>Stock Price Alert</h2>
            <p><strong>{ticker}</strong> has crossed your price threshold!</p>
            <ul>
                <li><strong>Current Price:</strong> ${current_price:.2f}</li>
                <li><strong>Threshold:</strong> ${threshold_price:.2f}</li>
                <li><strong>Alert Type:</strong> Price went {alert_type} threshold</li>
                <li><strong>Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</li>
            </ul>
            <p>Consider reviewing your position in {ticker}.</p>
        </body>
        </html>
        """
        
        return self.send_email(subject, body)
    
    def send_portfolio_summary(self, portfolio_data: Dict[str, Any]) -> bool:
        """Send daily portfolio summary.
        
        Args:
            portfolio_data: Portfolio summary data
            
        Returns:
            True if successful
        """
        subject = f"📊 Daily Portfolio Summary - {datetime.now().strftime('%Y-%m-%d')}"
        
        # Build holdings table
        holdings_html = ""
        for holding in portfolio_data.get('holdings', []):
            profit_color = 'green' if holding['profit_loss'] >= 0 else 'red'
            holdings_html += f"""
            <tr>
                <td>{holding['ticker']}</td>
                <td>${holding['current_price']:.2f}</td>
                <td style="color: {profit_color};">${holding['profit_loss']:.2f}</td>
                <td style="color: {profit_color};">{holding['profit_loss_pct']:.2f}%</td>
            </tr>
            """
        
        total_color = 'green' if portfolio_data['total_profit_loss'] >= 0 else 'red'
        
        body = f"""
        <html>
        <body style="font-family: Arial, sans-serif;">
            <h2>Portfolio Summary</h2>
            <h3>Overall Performance</h3>
            <ul>
                <li><strong>Total Value:</strong> ${portfolio_data['total_value']:.2f}</li>
                <li><strong>Total Cost:</strong> ${portfolio_data['total_cost']:.2f}</li>
                <li style="color: {total_color};"><strong>Profit/Loss:</strong> ${portfolio_data['total_profit_loss']:.2f} ({portfolio_data['total_roi']:.2f}%)</li>
                <li><strong>Daily Change:</strong> ${portfolio_data.get('daily_change', 0):.2f} ({portfolio_data.get('daily_change_pct', 0):.2f}%)</li>
            </ul>
            
            <h3>Holdings</h3>
            <table border="1" cellpadding="5" cellspacing="0" style="border-collapse: collapse;">
                <tr style="background-color: #f0f0f0;">
                    <th>Ticker</th>
                    <th>Current Price</th>
                    <th>Profit/Loss</th>
                    <th>ROI %</th>
                </tr>
                {holdings_html}
            </table>
            
            <p style="margin-top: 20px; color: #666;">
                Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            </p>
        </body>
        </html>
        """
        
        return self.send_email(subject, body)
    
    def send_anomaly_alert(self, ticker: str, anomaly_data: Dict[str, Any]) -> bool:
        """Send anomaly detection alert.
        
        Args:
            ticker: Stock ticker
            anomaly_data: Anomaly information
            
        Returns:
            True if successful
        """
        subject = f"⚠️ Anomaly Alert: {ticker}"
        
        reasons = ', '.join(anomaly_data.get('reasons', []))
        
        body = f"""
        <html>
        <body style="font-family: Arial, sans-serif;">
            <h2>Unusual Activity Detected</h2>
            <p>An anomaly has been detected for <strong>{ticker}</strong>:</p>
            <ul>
                <li><strong>Date:</strong> {anomaly_data.get('date', 'N/A')}</li>
                <li><strong>Price:</strong> ${anomaly_data.get('close', 0):.2f}</li>
                <li><strong>Return:</strong> {anomaly_data.get('return', 0)*100:.2f}%</li>
                <li><strong>Reasons:</strong> {reasons}</li>
            </ul>
            <p>This unusual activity may warrant further investigation.</p>
        </body>
        </html>
        """
        
        return self.send_email(subject, body)
    
    def send_recommendation_alert(self, ticker: str, recommendation: Dict[str, Any]) -> bool:
        """Send recommendation alert.
        
        Args:
            ticker: Stock ticker
            recommendation: Recommendation data
            
        Returns:
            True if successful
        """
        subject = f"💡 Recommendation: {recommendation['recommendation']} {ticker}"
        
        body = f"""
        <html>
        <body style="font-family: Arial, sans-serif;">
            <h2>Stock Recommendation</h2>
            <p><strong>Recommendation:</strong> {recommendation['recommendation']} {ticker}</p>
            <p><strong>Confidence:</strong> {recommendation['confidence']*100:.0f}%</p>
            
            <h3>Analysis Summary</h3>
            <pre style="background-color: #f5f5f5; padding: 10px; border-radius: 5px;">
{recommendation['summary']}
            </pre>
            
            <p style="color: #666; margin-top: 20px;">
                <em>This is an automated recommendation based on technical analysis, 
                sentiment analysis, and price predictions. Always do your own research 
                before making investment decisions.</em>
            </p>
        </body>
        </html>
        """
        
        return self.send_email(subject, body)
    
    def is_configured(self) -> bool:
        """Check if email is configured.
        
        Returns:
            True if configured
        """
        return config.has_email_config()
