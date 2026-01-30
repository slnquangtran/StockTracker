"""Engine for advanced financial metrics and contextual insights."""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any

class AnalyticsEngine:
    """Calculates risk metrics and generates contextual insights."""
    
    @staticmethod
    def calculate_risk_metrics(data: pd.Series, risk_free_rate: float = 0.02) -> Dict[str, float]:
        """Calculate professional risk metrics.
        
        Args:
            data: Historical price series
            risk_free_rate: Annual risk-free rate for Sharpe calculation
            
        Returns:
            Dictionary of metrics
        """
        returns = data.pct_change().dropna()
        
        # Sharpe Ratio (Annualized)
        excess_returns = returns - (risk_free_rate / 252)
        sharpe = np.sqrt(252) * excess_returns.mean() / returns.std() if returns.std() != 0 else 0
        
        # Volatility (Annualized)
        volatility = returns.std() * np.sqrt(252)
        
        # Max Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Value at Risk (VaR) 95%
        var_95 = np.percentile(returns, 5)
        
        return {
            'sharpe_ratio': float(sharpe),
            'volatility': float(volatility),
            'max_drawdown': float(max_drawdown),
            'var_95': float(var_95),
            'avg_daily_return': float(returns.mean())
        }
    
    @staticmethod
    def generate_insights(ticker: str, data: pd.DataFrame, 
                         forecast_res: List[Dict],
                         risk_metrics: Dict[str, float]) -> List[str]:
        """Generate human-readable insights based on data.
        
        Args:
            ticker: Stock ticker
            data: Historical DataFrame with indicators
            forecast_res: List of predicted results
            risk_metrics: Calculated risk dictionary
            
        Returns:
            List of insight strings
        """
        insights = []
        
        # 1. Price Movement Insight
        last_price = data['Close'].iloc[-1]
        pred_price = forecast_res[-1]['price']
        change_pct = ((pred_price - last_price) / last_price) * 100
        days = len(forecast_res)
        
        direction = "increase" if change_pct > 0 else "decrease"
        insights.append(f"The model predicts a {abs(change_pct):.1f}% {direction} over the next {days} days.")
        
        # 2. Risk Insight
        if risk_metrics['volatility'] > 0.4:
            insights.append(f"High volatility detected ({risk_metrics['volatility']*100:.1f}% annualized). Expect price swings.")
        elif risk_metrics['volatility'] < 0.15:
            insights.append("Stock shows low volatility, suggesting stable price action.")
            
        # 3. Technical Indicator Insights (RSI)
        if 'RSI' in data.columns:
            rsi = data['RSI'].iloc[-1]
            if rsi > 70:
                insights.append(f"Current RSI of {rsi:.1f} suggests overbought conditions (potential pullback).")
            elif rsi < 30:
                insights.append(f"Current RSI of {rsi:.1f} suggests oversold conditions (potential bounce).")
        
        # 4. Uncertainty Insight
        insights.append("Prediction uncertainty typically increases significantly beyond the 10-day window.")
        
        return insights

    @staticmethod
    def get_support_resistance(data: pd.DataFrame, window: int = 20) -> Dict[str, List[float]]:
        """Identify potential support and resistance levels."""
        prices = data['Close']
        resistance = []
        support = []
        
        for i in range(window, len(prices) - window):
            # Resistance: Local peak
            if prices.iloc[i] == max(prices.iloc[i-window:i+window]):
                resistance.append(float(prices.iloc[i]))
            # Support: Local trough
            if prices.iloc[i] == min(prices.iloc[i-window:i+window]):
                support.append(float(prices.iloc[i]))
                
        # Return unique and most recent/significant levels
        return {
            'resistance': sorted(list(set(resistance)))[-3:],
            'support': sorted(list(set(support)))[:3]
        }
