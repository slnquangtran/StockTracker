"""Anomaly detection for unusual price movements."""
import numpy as np
import pandas as pd
from typing import Dict, List, Any
from sklearn.ensemble import IsolationForest
from scipy import stats
from src.data.stock_fetcher import StockFetcher

class AnomalyDetector:
    """Detect anomalies in stock price movements."""
    
    def __init__(self, contamination: float = 0.1, z_threshold: float = 3.0,
                 volume_threshold: float = 2.5):
        """Initialize anomaly detector.
        
        Args:
            contamination: Expected proportion of outliers (for Isolation Forest)
            z_threshold: Z-score threshold for anomaly detection
            volume_threshold: Volume spike threshold (multiples of average)
        """
        self.contamination = contamination
        self.z_threshold = z_threshold
        self.volume_threshold = volume_threshold
        self.fetcher = StockFetcher()
    
    def detect_price_anomalies(self, ticker: str, period: str = '3mo') -> Dict[str, Any]:
        """Detect price anomalies using multiple methods.
        
        Args:
            ticker: Stock ticker symbol
            period: Historical period to analyze
            
        Returns:
            Dictionary with anomaly information
        """
        data = self.fetcher.get_historical_data(ticker, period=period)
        
        if data is None or len(data) < 30:
            return {'error': 'Insufficient data'}
        
        # Calculate returns
        data['returns'] = data['Close'].pct_change()
        data['abs_returns'] = data['returns'].abs()
        
        # Method 1: Z-score based detection
        z_scores = np.abs(stats.zscore(data['returns'].dropna()))
        z_anomalies = z_scores > self.z_threshold
        
        # Method 2: Isolation Forest
        features = data[['returns', 'Volume']].dropna()
        iso_forest = IsolationForest(contamination=self.contamination, random_state=42)
        iso_predictions = iso_forest.fit_predict(features)
        iso_anomalies = iso_predictions == -1
        
        # Method 3: Volume spikes
        avg_volume = data['Volume'].rolling(window=20).mean()
        volume_anomalies = data['Volume'] > (avg_volume * self.volume_threshold)
        
        # Combine anomalies
        anomaly_dates = []
        for i in range(len(data)):
            date = data.index[i]
            is_anomaly = False
            reasons = []
            
            if i < len(z_anomalies) and z_anomalies[i]:
                is_anomaly = True
                reasons.append('price_zscore')
            
            if i < len(iso_anomalies) and iso_anomalies[i]:
                is_anomaly = True
                reasons.append('isolation_forest')
            
            if volume_anomalies.iloc[i]:
                is_anomaly = True
                reasons.append('volume_spike')
            
            if is_anomaly:
                anomaly_dates.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'close': float(data['Close'].iloc[i]),
                    'return': float(data['returns'].iloc[i]) if not pd.isna(data['returns'].iloc[i]) else 0,
                    'volume': int(data['Volume'].iloc[i]),
                    'reasons': reasons
                })
        
        return {
            'ticker': ticker,
            'num_anomalies': len(anomaly_dates),
            'anomalies': anomaly_dates[-10:],  # Last 10 anomalies
            'recent_anomaly': anomaly_dates[-1] if anomaly_dates else None
        }
    
    def detect_recent_anomaly(self, ticker: str) -> bool:
        """Check if there's a recent anomaly (last 5 days).
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            True if recent anomaly detected
        """
        result = self.detect_price_anomalies(ticker, period='1mo')
        
        if 'error' in result or not result['anomalies']:
            return False
        
        # Check if any anomaly in last 5 days
        recent_date = pd.Timestamp.now() - pd.Timedelta(days=5)
        
        for anomaly in result['anomalies']:
            anomaly_date = pd.Timestamp(anomaly['date'])
            if anomaly_date >= recent_date:
                return True
        
        return False
    
    def get_anomaly_summary(self, ticker: str) -> str:
        """Get human-readable anomaly summary.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Summary string
        """
        result = self.detect_price_anomalies(ticker)
        
        if 'error' in result:
            return f"Unable to analyze {ticker}"
        
        if result['num_anomalies'] == 0:
            return f"No anomalies detected for {ticker}"
        
        recent = result['recent_anomaly']
        if recent:
            reasons = ', '.join(recent['reasons'])
            return (f"{ticker} anomaly on {recent['date']}: "
                   f"{recent['return']*100:.2f}% change ({reasons})")
        
        return f"{ticker}: {result['num_anomalies']} anomalies detected"
