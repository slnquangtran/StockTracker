"""Ensemble model combining LSTM and ARIMA forecasts."""
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from src.models.lstm_predictor import LSTMPredictor
from src.models.arima_predictor import ARIMAPredictor
from src.analysis.analytics_engine import AnalyticsEngine

class EnsemblePredictor:
    """Combines Deep Learning and Statistical models for better forecasting."""
    
    def __init__(self):
        self.lstm = LSTMPredictor()
        self.arima = ARIMAPredictor()
        self.analytics = AnalyticsEngine()
        
    def predict_ensemble(self, ticker: str, days: int = 30) -> Dict:
        """Generate ensemble prediction with metrics and insights.
        
        Args:
            ticker: Stock symbol
            days: Forecast horizon
            
        Returns:
            Dictionary with unified results
        """
        # 1. Fetch data
        data = self.lstm.fetcher.get_historical_data(ticker, period='2y')
        if data is None or data.empty:
            return {'success': False, 'error': "Data fetch failed"}
            
        # 2. Get LSTM predictions
        lstm_preds = self.lstm.predict(ticker, days=days)
        
        # 3. Get ARIMA predictions
        arima_res = self.arima.predict(data['Close'], days=days)
        
        if not arima_res['success'] and lstm_preds is None:
            return {'success': False, 'error': "Both models failed"}
            
        # 4. Unify results
        results = []
        for i in range(days):
            date = arima_res['forecast'][i]['date'] if arima_res['success'] else lstm_preds[i]['date']
            
            # Weighted average (60% LSTM, 40% ARIMA)
            weights = {'lstm': 0.6, 'arima': 0.4}
            
            vals = []
            if lstm_preds: vals.append(lstm_preds[i]['predicted_price'] * weights['lstm'])
            if arima_res['success']: vals.append(arima_res['forecast'][i]['price'] * weights['arima'])
            
            # If one failed, use the other 100%
            if len(vals) == 1:
                final_price = lstm_preds[i]['predicted_price'] if lstm_preds else arima_res['forecast'][i]['price']
            else:
                final_price = sum(vals)
            
            results.append({
                'date': date,
                'price': float(final_price),
                'lstm': float(lstm_preds[i]['predicted_price']) if lstm_preds else None,
                'arima': float(arima_res['forecast'][i]['price']) if arima_res['success'] else None,
                'lower': arima_res['forecast'][i]['lower'] if arima_res['success'] else final_price * 0.95,
                'upper': arima_res['forecast'][i]['upper'] if arima_res['success'] else final_price * 1.05
            })
            
        # 5. Get Risk & Insights
        risk = self.analytics.calculate_risk_metrics(data['Close'])
        # Add RSI etc for insights
        from src.analysis.technical_indicators import TechnicalIndicators
        data_with_indicators = TechnicalIndicators.add_all_indicators(data)
        insights = self.analytics.generate_insights(ticker, data_with_indicators, results, risk)
        levels = self.analytics.get_support_resistance(data)
        
        return {
            'success': True,
            'ticker': ticker,
            'forecast': results,
            'risk_metrics': risk,
            'insights': insights,
            'levels': levels,
            'historical_data': data['Close'].tail(100).to_dict()
        }
