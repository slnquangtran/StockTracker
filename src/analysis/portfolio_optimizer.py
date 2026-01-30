"""Portfolio optimization using Modern Portfolio Theory."""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from scipy.optimize import minimize
from src.data.stock_fetcher import StockFetcher

class PortfolioOptimizer:
    """Optimize portfolio allocation using Sharpe ratio."""
    
    def __init__(self, risk_free_rate: float = 0.02):
        """Initialize optimizer.
        
        Args:
            risk_free_rate: Annual risk-free rate (default 2%)
        """
        self.risk_free_rate = risk_free_rate
        self.fetcher = StockFetcher()
    
    def get_returns(self, tickers: List[str], period: str = '1y') -> pd.DataFrame:
        """Get historical returns for multiple stocks.
        
        Args:
            tickers: List of stock tickers
            period: Historical period
            
        Returns:
            DataFrame with daily returns
        """
        prices = pd.DataFrame()
        
        for ticker in tickers:
            data = self.fetcher.get_historical_data(ticker, period=period)
            if data is not None:
                prices[ticker] = data['Close']
        
        # Calculate daily returns
        returns = prices.pct_change().dropna()
        return returns
    
    def calculate_portfolio_stats(self, weights: np.ndarray, returns: pd.DataFrame) -> Tuple[float, float, float]:
        """Calculate portfolio statistics.
        
        Args:
            weights: Portfolio weights
            returns: Historical returns DataFrame
            
        Returns:
            Tuple of (expected_return, volatility, sharpe_ratio)
        """
        # Annualize returns (252 trading days)
        portfolio_return = np.sum(returns.mean() * weights) * 252
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_std
        
        return portfolio_return, portfolio_std, sharpe_ratio
    
    def optimize_sharpe(self, tickers: List[str], period: str = '1y',
                       constraints: Dict = None) -> Dict:
        """Optimize portfolio for maximum Sharpe ratio.
        
        Args:
            tickers: List of stock tickers
            period: Historical period for optimization
            constraints: Optional constraints (min_weight, max_weight)
            
        Returns:
            Dictionary with optimal weights and statistics
        """
        returns = self.get_returns(tickers, period)
        
        if returns.empty:
            return {'error': 'No data available for optimization'}
        
        num_assets = len(tickers)
        
        # Objective function (negative Sharpe ratio for minimization)
        def neg_sharpe(weights):
            _, _, sharpe = self.calculate_portfolio_stats(weights, returns)
            return -sharpe
        
        # Constraints
        constraints_list = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]  # Weights sum to 1
        
        # Bounds
        min_weight = constraints.get('min_weight', 0.0) if constraints else 0.0
        max_weight = constraints.get('max_weight', 1.0) if constraints else 1.0
        bounds = tuple((min_weight, max_weight) for _ in range(num_assets))
        
        # Initial guess (equal weights)
        init_weights = np.array([1.0 / num_assets] * num_assets)
        
        # Optimize
        result = minimize(
            neg_sharpe,
            init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list
        )
        
        if result.success:
            optimal_weights = result.x
            exp_return, volatility, sharpe = self.calculate_portfolio_stats(optimal_weights, returns)
            
            return {
                'tickers': tickers,
                'weights': {ticker: float(weight) for ticker, weight in zip(tickers, optimal_weights)},
                'expected_return': float(exp_return),
                'volatility': float(volatility),
                'sharpe_ratio': float(sharpe),
                'success': True
            }
        else:
            return {'error': 'Optimization failed', 'success': False}
    
    def optimize_min_volatility(self, tickers: List[str], period: str = '1y',
                                target_return: float = None) -> Dict:
        """Optimize portfolio for minimum volatility.
        
        Args:
            tickers: List of stock tickers
            period: Historical period
            target_return: Optional target return constraint
            
        Returns:
            Dictionary with optimal weights and statistics
        """
        returns = self.get_returns(tickers, period)
        
        if returns.empty:
            return {'error': 'No data available for optimization'}
        
        num_assets = len(tickers)
        
        # Objective function (portfolio volatility)
        def portfolio_volatility(weights):
            _, vol, _ = self.calculate_portfolio_stats(weights, returns)
            return vol
        
        # Constraints
        constraints_list = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        
        if target_return:
            constraints_list.append({
                'type': 'eq',
                'fun': lambda x: self.calculate_portfolio_stats(x, returns)[0] - target_return
            })
        
        bounds = tuple((0, 1) for _ in range(num_assets))
        init_weights = np.array([1.0 / num_assets] * num_assets)
        
        result = minimize(
            portfolio_volatility,
            init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list
        )
        
        if result.success:
            optimal_weights = result.x
            exp_return, volatility, sharpe = self.calculate_portfolio_stats(optimal_weights, returns)
            
            return {
                'tickers': tickers,
                'weights': {ticker: float(weight) for ticker, weight in zip(tickers, optimal_weights)},
                'expected_return': float(exp_return),
                'volatility': float(volatility),
                'sharpe_ratio': float(sharpe),
                'success': True
            }
        else:
            return {'error': 'Optimization failed', 'success': False}
    
    def efficient_frontier(self, tickers: List[str], period: str = '1y',
                          num_portfolios: int = 100) -> pd.DataFrame:
        """Generate efficient frontier.
        
        Args:
            tickers: List of stock tickers
            period: Historical period
            num_portfolios: Number of portfolios to generate
            
        Returns:
            DataFrame with portfolio statistics
        """
        returns = self.get_returns(tickers, period)
        
        if returns.empty:
            return pd.DataFrame()
        
        results = []
        num_assets = len(tickers)
        
        for _ in range(num_portfolios):
            # Random weights
            weights = np.random.random(num_assets)
            weights /= np.sum(weights)
            
            exp_return, volatility, sharpe = self.calculate_portfolio_stats(weights, returns)
            
            results.append({
                'return': exp_return,
                'volatility': volatility,
                'sharpe': sharpe
            })
        
        return pd.DataFrame(results)
