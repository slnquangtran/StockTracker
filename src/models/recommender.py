"""Recommendation engine for buy/sell/hold decisions."""
import numpy as np
from typing import Dict, Any, Optional
from src.data.stock_fetcher import StockFetcher
from src.analysis.technical_indicators import TechnicalIndicators
from src.models.sentiment_analyzer import SentimentAnalyzer
from src.models.anomaly_detector import AnomalyDetector
from src.models.lstm_predictor import LSTMPredictor

class Recommender:
    """Generate buy/sell/hold recommendations based on multiple signals."""
    
    def __init__(self, weights: Dict[str, float] = None):
        """Initialize recommender.
        
        Args:
            weights: Weights for different signals (technical, sentiment, prediction, anomaly)
        """
        self.weights = weights or {
            'technical': 0.3,
            'sentiment': 0.2,
            'prediction': 0.3,
            'anomaly': 0.2
        }
        
        self.fetcher = StockFetcher()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.anomaly_detector = AnomalyDetector()
        self.predictor = LSTMPredictor()
    
    def get_recommendation(self, ticker: str, company_name: str = None) -> Dict[str, Any]:
        """Get comprehensive recommendation for a stock.
        
        Args:
            ticker: Stock ticker symbol
            company_name: Company name for sentiment analysis
            
        Returns:
            Dictionary with recommendation and supporting data
        """
        # Gather signals
        technical_signal = self._get_technical_signal(ticker)
        sentiment_signal = self._get_sentiment_signal(ticker, company_name)
        prediction_signal = self._get_prediction_signal(ticker)
        anomaly_signal = self._get_anomaly_signal(ticker)
        
        # Calculate weighted score
        total_score = (
            technical_signal['score'] * self.weights['technical'] +
            sentiment_signal['score'] * self.weights['sentiment'] +
            prediction_signal['score'] * self.weights['prediction'] +
            anomaly_signal['score'] * self.weights['anomaly']
        )
        
        # Normalize to -1 to 1
        total_score = np.clip(total_score, -1, 1)
        
        # Determine recommendation
        if total_score > 0.3:
            recommendation = 'BUY'
            confidence = abs(total_score)
        elif total_score < -0.3:
            recommendation = 'SELL'
            confidence = abs(total_score)
        else:
            recommendation = 'HOLD'
            confidence = 1 - abs(total_score)
        
        return {
            'ticker': ticker,
            'recommendation': recommendation,
            'confidence': float(confidence),
            'score': float(total_score),
            'signals': {
                'technical': technical_signal,
                'sentiment': sentiment_signal,
                'prediction': prediction_signal,
                'anomaly': anomaly_signal
            },
            'summary': self._generate_summary(ticker, recommendation, confidence, 
                                             technical_signal, sentiment_signal,
                                             prediction_signal, anomaly_signal)
        }
    
    def _get_technical_signal(self, ticker: str) -> Dict[str, Any]:
        """Get technical analysis signal.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Signal dictionary with score and details
        """
        data = self.fetcher.get_historical_data(ticker, period='3mo')
        
        if data is None or len(data) < 50:
            return {'score': 0.0, 'details': 'Insufficient data'}
        
        # Add indicators
        data = TechnicalIndicators.add_all_indicators(data)
        latest = data.iloc[-1]
        
        signals = []
        
        # RSI signal
        rsi = latest['RSI']
        if rsi < 30:
            signals.append(('RSI oversold', 0.5))
        elif rsi > 70:
            signals.append(('RSI overbought', -0.5))
        else:
            signals.append(('RSI neutral', 0.0))
        
        # MACD signal
        if latest['MACD'] > latest['MACD_Signal']:
            signals.append(('MACD bullish', 0.3))
        else:
            signals.append(('MACD bearish', -0.3))
        
        # SMA trend
        if latest['Close'] > latest['SMA_50'] > latest['SMA_200']:
            signals.append(('Strong uptrend', 0.4))
        elif latest['Close'] < latest['SMA_50'] < latest['SMA_200']:
            signals.append(('Strong downtrend', -0.4))
        else:
            signals.append(('Mixed trend', 0.0))
        
        # Bollinger Bands
        if latest['Close'] < latest['BB_Lower']:
            signals.append(('Below lower BB', 0.3))
        elif latest['Close'] > latest['BB_Upper']:
            signals.append(('Above upper BB', -0.3))
        
        # Calculate average score
        avg_score = np.mean([s[1] for s in signals])
        
        return {
            'score': float(avg_score),
            'details': ', '.join([s[0] for s in signals]),
            'rsi': float(rsi)
        }
    
    def _get_sentiment_signal(self, ticker: str, company_name: str = None) -> Dict[str, Any]:
        """Get sentiment analysis signal.
        
        Args:
            ticker: Stock ticker symbol
            company_name: Company name
            
        Returns:
            Signal dictionary
        """
        if not self.sentiment_analyzer.is_configured():
            return {'score': 0.0, 'details': 'Sentiment analysis not configured'}
        
        sentiment = self.sentiment_analyzer.analyze_stock(ticker, company_name)
        
        return {
            'score': float(sentiment['sentiment_score']),
            'details': f"{sentiment['sentiment_label']} ({sentiment['num_articles']} articles)",
            'label': sentiment['sentiment_label']
        }
    
    def _get_prediction_signal(self, ticker: str) -> Dict[str, Any]:
        """Get LSTM prediction signal.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Signal dictionary
        """
        try:
            predictions = self.predictor.predict(ticker, days=7)
            
            if predictions is None:
                return {'score': 0.0, 'details': 'Prediction unavailable'}
            
            # Compare predicted price with current price
            current_price = self.fetcher.get_current_price(ticker)
            if current_price is None:
                return {'score': 0.0, 'details': 'Current price unavailable'}
            
            future_price = predictions[-1]['predicted_price']
            change_pct = (future_price - current_price) / current_price
            
            # Convert to signal (-1 to 1)
            score = np.clip(change_pct * 2, -1, 1)
            
            return {
                'score': float(score),
                'details': f"7-day prediction: {change_pct*100:.2f}% change",
                'predicted_price': float(future_price),
                'current_price': float(current_price)
            }
        except Exception as e:
            return {'score': 0.0, 'details': f'Prediction error: {str(e)}'}
    
    def _get_anomaly_signal(self, ticker: str) -> Dict[str, Any]:
        """Get anomaly detection signal.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Signal dictionary
        """
        has_anomaly = self.anomaly_detector.detect_recent_anomaly(ticker)
        
        if has_anomaly:
            # Recent anomaly suggests caution
            return {
                'score': -0.3,
                'details': 'Recent anomaly detected - caution advised'
            }
        else:
            return {
                'score': 0.0,
                'details': 'No recent anomalies'
            }
    
    def _generate_summary(self, ticker: str, recommendation: str, confidence: float,
                         technical: Dict, sentiment: Dict, prediction: Dict,
                         anomaly: Dict) -> str:
        """Generate human-readable summary.
        
        Args:
            ticker: Stock ticker
            recommendation: Recommendation (BUY/SELL/HOLD)
            confidence: Confidence score
            technical, sentiment, prediction, anomaly: Signal dictionaries
            
        Returns:
            Summary string
        """
        summary = f"{recommendation} {ticker} (confidence: {confidence:.0%})\n\n"
        summary += f"Technical: {technical['details']}\n"
        summary += f"Sentiment: {sentiment['details']}\n"
        summary += f"Prediction: {prediction['details']}\n"
        summary += f"Anomaly: {anomaly['details']}"
        
        return summary
