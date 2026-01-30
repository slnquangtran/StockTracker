"""Backtesting framework for trading strategies."""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Callable
from datetime import datetime
from src.data.stock_fetcher import StockFetcher
from src.analysis.technical_indicators import TechnicalIndicators

class Backtester:
    """Backtest trading strategies on historical data."""
    
    def __init__(self, initial_capital: float = 10000, commission: float = 0.001,
                 slippage: float = 0.0005):
        """Initialize backtester.
        
        Args:
            initial_capital: Starting capital
            commission: Commission per trade (as fraction, e.g., 0.001 = 0.1%)
            slippage: Slippage per trade (as fraction)
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.fetcher = StockFetcher()
    
    def run_strategy(self, ticker: str, strategy: str, start_date: str = None,
                    end_date: str = None, **kwargs) -> Dict[str, Any]:
        """Run a backtesting strategy.
        
        Args:
            ticker: Stock ticker symbol
            strategy: Strategy name ('sma_crossover', 'rsi', 'macd', 'buy_hold')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            **kwargs: Strategy-specific parameters
            
        Returns:
            Dictionary with backtest results
        """
        # Get historical data
        data = self.fetcher.get_historical_data(ticker, period='2y')
        
        if data is None or data.empty:
            return {'error': f'No data available for {ticker}'}
        
        # Filter by date range
        if start_date:
            data = data[data.index >= start_date]
        if end_date:
            data = data[data.index <= end_date]
        
        # Add technical indicators
        data = TechnicalIndicators.add_all_indicators(data)
        
        # Run strategy
        if strategy == 'sma_crossover':
            signals = self._sma_crossover_strategy(data, **kwargs)
        elif strategy == 'rsi':
            signals = self._rsi_strategy(data, **kwargs)
        elif strategy == 'macd':
            signals = self._macd_strategy(data, **kwargs)
        elif strategy == 'buy_hold':
            signals = self._buy_hold_strategy(data)
        else:
            return {'error': f'Unknown strategy: {strategy}'}
        
        # Calculate returns
        results = self._calculate_returns(data, signals, ticker)
        return results
    
    def _sma_crossover_strategy(self, data: pd.DataFrame, fast: int = 20,
                                slow: int = 50) -> pd.Series:
        """SMA crossover strategy.
        
        Args:
            data: DataFrame with price data and indicators
            fast: Fast SMA period
            slow: Slow SMA period
            
        Returns:
            Series with signals (1=buy, -1=sell, 0=hold)
        """
        signals = pd.Series(0, index=data.index)
        
        sma_fast = TechnicalIndicators.sma(data, fast)
        sma_slow = TechnicalIndicators.sma(data, slow)
        
        # Buy when fast crosses above slow
        signals[sma_fast > sma_slow] = 1
        # Sell when fast crosses below slow
        signals[sma_fast < sma_slow] = -1
        
        return signals
    
    def _rsi_strategy(self, data: pd.DataFrame, oversold: int = 30,
                     overbought: int = 70) -> pd.Series:
        """RSI strategy.
        
        Args:
            data: DataFrame with price data and indicators
            oversold: RSI oversold threshold
            overbought: RSI overbought threshold
            
        Returns:
            Series with signals
        """
        signals = pd.Series(0, index=data.index)
        
        rsi = data['RSI'] if 'RSI' in data.columns else TechnicalIndicators.rsi(data)
        
        # Buy when RSI is oversold
        signals[rsi < oversold] = 1
        # Sell when RSI is overbought
        signals[rsi > overbought] = -1
        
        return signals
    
    def _macd_strategy(self, data: pd.DataFrame) -> pd.Series:
        """MACD strategy.
        
        Args:
            data: DataFrame with price data and indicators
            
        Returns:
            Series with signals
        """
        signals = pd.Series(0, index=data.index)
        
        if 'MACD' in data.columns and 'MACD_Signal' in data.columns:
            macd = data['MACD']
            signal = data['MACD_Signal']
        else:
            macd_data = TechnicalIndicators.macd(data)
            macd = macd_data['MACD']
            signal = macd_data['Signal']
        
        # Buy when MACD crosses above signal
        signals[macd > signal] = 1
        # Sell when MACD crosses below signal
        signals[macd < signal] = -1
        
        return signals
    
    def _buy_hold_strategy(self, data: pd.DataFrame) -> pd.Series:
        """Buy and hold strategy (baseline).
        
        Args:
            data: DataFrame with price data
            
        Returns:
            Series with signals
        """
        signals = pd.Series(0, index=data.index)
        signals.iloc[0] = 1  # Buy at start
        return signals
    
    def _calculate_returns(self, data: pd.DataFrame, signals: pd.Series,
                          ticker: str) -> Dict[str, Any]:
        """Calculate strategy returns and metrics.
        
        Args:
            data: DataFrame with price data
            signals: Series with trading signals
            ticker: Stock ticker
            
        Returns:
            Dictionary with performance metrics
        """
        # Initialize
        capital = self.initial_capital
        position = 0
        trades = []
        equity_curve = []
        
        for i in range(len(data)):
            date = data.index[i]
            price = data['Close'].iloc[i]
            signal = signals.iloc[i]
            
            # Execute trades
            if signal == 1 and position == 0:  # Buy
                shares = int(capital / (price * (1 + self.commission + self.slippage)))
                cost = shares * price * (1 + self.commission + self.slippage)
                if shares > 0:
                    position = shares
                    capital -= cost
                    trades.append({
                        'date': date,
                        'action': 'BUY',
                        'price': price,
                        'shares': shares,
                        'cost': cost
                    })
            
            elif signal == -1 and position > 0:  # Sell
                proceeds = position * price * (1 - self.commission - self.slippage)
                capital += proceeds
                trades.append({
                    'date': date,
                    'action': 'SELL',
                    'price': price,
                    'shares': position,
                    'proceeds': proceeds
                })
                position = 0
            
            # Calculate equity
            equity = capital + (position * price if position > 0 else 0)
            equity_curve.append({
                'date': date,
                'equity': equity,
                'price': price
            })
        
        # Close any open position
        if position > 0:
            final_price = data['Close'].iloc[-1]
            proceeds = position * final_price * (1 - self.commission - self.slippage)
            capital += proceeds
            trades.append({
                'date': data.index[-1],
                'action': 'SELL',
                'price': final_price,
                'shares': position,
                'proceeds': proceeds
            })
        
        # Calculate metrics
        final_equity = capital
        total_return = (final_equity - self.initial_capital) / self.initial_capital
        
        equity_df = pd.DataFrame(equity_curve)
        equity_df['returns'] = equity_df['equity'].pct_change()
        
        sharpe_ratio = (equity_df['returns'].mean() / equity_df['returns'].std() * 
                       np.sqrt(252)) if equity_df['returns'].std() > 0 else 0
        
        max_drawdown = self._calculate_max_drawdown(equity_df['equity'])
        
        return {
            'ticker': ticker,
            'initial_capital': self.initial_capital,
            'final_equity': final_equity,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'num_trades': len(trades),
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'trades': trades,
            'equity_curve': equity_curve,
            'summary': {
                'Total Return': f"{total_return * 100:.2f}%",
                'Sharpe Ratio': f"{sharpe_ratio:.2f}",
                'Max Drawdown': f"{max_drawdown * 100:.2f}%",
                'Number of Trades': len(trades),
                'Final Equity': f"${final_equity:.2f}"
            }
        }
    
    def _calculate_max_drawdown(self, equity: pd.Series) -> float:
        """Calculate maximum drawdown.
        
        Args:
            equity: Series of equity values
            
        Returns:
            Maximum drawdown as fraction
        """
        cummax = equity.cummax()
        drawdown = (equity - cummax) / cummax
        return abs(drawdown.min())
