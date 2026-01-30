"""Portfolio management system."""
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from src.data.stock_fetcher import StockFetcher

class Portfolio:
    """Manage stock portfolio with tracking and analytics."""
    
    def __init__(self, name: str = "My Portfolio", data_file: str = None):
        """Initialize portfolio.
        
        Args:
            name: Portfolio name
            data_file: Path to portfolio data file
        """
        self.name = name
        self.data_file = data_file or f"data/{name.replace(' ', '_').lower()}.json"
        self.holdings: List[Dict[str, Any]] = []
        self.fetcher = StockFetcher()
        self.load()
    
    def add_stock(self, ticker: str, quantity: float, purchase_price: float, 
                  purchase_date: str = None) -> bool:
        """Add stock to portfolio.
        
        Args:
            ticker: Stock ticker symbol
            quantity: Number of shares
            purchase_price: Price per share at purchase
            purchase_date: Purchase date (YYYY-MM-DD format)
            
        Returns:
            True if successful, False otherwise
        """
        # Validate ticker
        if not self.fetcher.validate_ticker(ticker):
            print(f"Invalid ticker: {ticker}")
            return False
        
        purchase_date = purchase_date or datetime.now().strftime('%Y-%m-%d')
        
        # Check if stock already exists
        for holding in self.holdings:
            if holding['ticker'] == ticker:
                # Update existing holding (average price)
                total_shares = holding['quantity'] + quantity
                total_cost = (holding['quantity'] * holding['purchase_price'] + 
                            quantity * purchase_price)
                holding['purchase_price'] = total_cost / total_shares
                holding['quantity'] = total_shares
                self.save()
                return True
        
        # Add new holding
        self.holdings.append({
            'ticker': ticker,
            'quantity': quantity,
            'purchase_price': purchase_price,
            'purchase_date': purchase_date
        })
        self.save()
        return True
    
    def remove_stock(self, ticker: str, quantity: float = None) -> bool:
        """Remove stock from portfolio.
        
        Args:
            ticker: Stock ticker symbol
            quantity: Number of shares to remove (None = remove all)
            
        Returns:
            True if successful, False otherwise
        """
        for i, holding in enumerate(self.holdings):
            if holding['ticker'] == ticker:
                if quantity is None or quantity >= holding['quantity']:
                    self.holdings.pop(i)
                else:
                    holding['quantity'] -= quantity
                self.save()
                return True
        return False
    
    def get_current_value(self) -> Dict[str, float]:
        """Get current portfolio value.
        
        Returns:
            Dictionary with total value and individual holdings
        """
        total_value = 0
        total_cost = 0
        holdings_value = []
        
        for holding in self.holdings:
            current_price = self.fetcher.get_current_price(holding['ticker'])
            if current_price:
                value = current_price * holding['quantity']
                cost = holding['purchase_price'] * holding['quantity']
                profit_loss = value - cost
                profit_loss_pct = (profit_loss / cost) * 100 if cost > 0 else 0
                
                holdings_value.append({
                    'ticker': holding['ticker'],
                    'quantity': holding['quantity'],
                    'purchase_price': holding['purchase_price'],
                    'current_price': current_price,
                    'cost_basis': cost,
                    'current_value': value,
                    'profit_loss': profit_loss,
                    'profit_loss_pct': profit_loss_pct,
                    'purchase_date': holding['purchase_date']
                })
                
                total_value += value
                total_cost += cost
        
        total_profit_loss = total_value - total_cost
        total_roi = (total_profit_loss / total_cost * 100) if total_cost > 0 else 0
        
        return {
            'total_value': total_value,
            'total_cost': total_cost,
            'total_profit_loss': total_profit_loss,
            'total_roi': total_roi,
            'holdings': holdings_value
        }
    
    def get_daily_change(self) -> Dict[str, Any]:
        """Get daily change for portfolio.
        
        Returns:
            Dictionary with daily change information
        """
        total_change = 0
        total_value = 0
        holdings_change = []
        
        for holding in self.holdings:
            data = self.fetcher.get_historical_data(holding['ticker'], period='5d')
            if data is not None and len(data) >= 2:
                current_price = data['Close'].iloc[-1]
                previous_price = data['Close'].iloc[-2]
                change = current_price - previous_price
                change_pct = (change / previous_price) * 100
                
                position_value = current_price * holding['quantity']
                position_change = change * holding['quantity']
                
                holdings_change.append({
                    'ticker': holding['ticker'],
                    'change': change,
                    'change_pct': change_pct,
                    'position_change': position_change
                })
                
                total_change += position_change
                total_value += position_value
        
        total_change_pct = (total_change / total_value * 100) if total_value > 0 else 0
        
        return {
            'total_change': total_change,
            'total_change_pct': total_change_pct,
            'holdings': holdings_change
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive portfolio summary.
        
        Returns:
            Dictionary with complete portfolio information
        """
        current_value = self.get_current_value()
        daily_change = self.get_daily_change()
        
        return {
            'name': self.name,
            'num_holdings': len(self.holdings),
            'total_value': current_value['total_value'],
            'total_cost': current_value['total_cost'],
            'total_profit_loss': current_value['total_profit_loss'],
            'total_roi': current_value['total_roi'],
            'daily_change': daily_change['total_change'],
            'daily_change_pct': daily_change['total_change_pct'],
            'holdings': current_value['holdings']
        }
    
    def save(self):
        """Save portfolio to file."""
        Path(self.data_file).parent.mkdir(parents=True, exist_ok=True)
        data = {
            'name': self.name,
            'holdings': self.holdings,
            'last_updated': datetime.now().isoformat()
        }
        with open(self.data_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self):
        """Load portfolio from file."""
        if Path(self.data_file).exists():
            try:
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                    self.name = data.get('name', self.name)
                    self.holdings = data.get('holdings', [])
            except Exception as e:
                print(f"Error loading portfolio: {e}")
    
    def get_tickers(self) -> List[str]:
        """Get list of all tickers in portfolio.
        
        Returns:
            List of ticker symbols
        """
        return [holding['ticker'] for holding in self.holdings]
