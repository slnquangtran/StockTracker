"""ARIMA-based statistical forecasting model."""
import numpy as np
import pandas as pd
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from typing import List, Dict, Optional, Tuple

warnings.filterwarnings('ignore')

class ARIMAPredictor:
    """Statistical ARIMA model for price forecasting."""
    
    def __init__(self, order: Tuple[int, int, int] = (5, 1, 0)):
        """Initialize ARIMA model.
        
        Args:
            order: (p, d, q) parameters for ARIMA
        """
        self.order = order
        
    def predict(self, data: pd.Series, days: int = 30) -> Dict:
        """Forecast future values using ARIMA.
        
        Args:
            data: Time series data (prices)
            days: Days to forecast
            
        Returns:
            Dictionary with forecast and confidence metrics
        """
        try:
            # Fit model
            model = SARIMAX(data, order=self.order, 
                            enforce_stationarity=False, 
                            enforce_invertibility=False)
            model_fit = model.fit(disp=False)
            
            # Forecast
            forecast_obj = model_fit.get_forecast(steps=days)
            forecast_mean = forecast_obj.summary_frame()['mean']
            forecast_conf = forecast_obj.summary_frame()[['mean_ci_lower', 'mean_ci_upper']]
            
            # Generate dates
            last_date = data.index[-1]
            forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days)
            
            # Prepare results
            results = []
            for i in range(days):
                results.append({
                    'date': forecast_dates[i].strftime('%Y-%m-%d'),
                    'price': float(forecast_mean.iloc[i]),
                    'lower': float(forecast_conf.iloc[i, 0]),
                    'upper': float(forecast_conf.iloc[i, 1])
                })
                
            return {
                'success': True,
                'forecast': results,
                'aic': float(model_fit.aic)
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def backtest(self, data: pd.Series, test_size: int = 30) -> Dict:
        """Perform simple backtest on historical data."""
        train = data[:-test_size]
        test = data[-test_size:]
        
        res = self.predict(train, days=test_size)
        if not res['success']: return res
        
        preds = np.array([p['price'] for p in res['forecast']])
        rmse = np.sqrt(np.mean((preds - test.values)**2))
        
        return {
            'success': True,
            'rmse': float(rmse),
            'actual': test.values.tolist(),
            'predicted': preds.tolist()
        }
