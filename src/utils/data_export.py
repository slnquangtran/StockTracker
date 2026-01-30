"""Data export utilities for CSV and Excel formats."""
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

class DataExporter:
    """Export data to various formats."""
    
    def __init__(self, output_dir: str = "exports"):
        """Initialize exporter.
        
        Args:
            output_dir: Directory to save exported files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def export_portfolio(self, portfolio_data: Dict[str, Any], format: str = 'csv') -> str:
        """Export portfolio data.
        
        Args:
            portfolio_data: Portfolio data dictionary
            format: Export format ('csv' or 'excel')
            
        Returns:
            Path to exported file
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"portfolio_{timestamp}.{format if format == 'csv' else 'xlsx'}"
        filepath = self.output_dir / filename
        
        # Convert to DataFrame
        df = pd.DataFrame(portfolio_data.get('holdings', []))
        
        if format == 'csv':
            df.to_csv(filepath, index=False)
        else:
            df.to_excel(filepath, index=False, engine='openpyxl')
        
        return str(filepath)
    
    def export_historical_data(self, ticker: str, data: pd.DataFrame, format: str = 'csv') -> str:
        """Export historical stock data.
        
        Args:
            ticker: Stock ticker symbol
            data: Historical data DataFrame
            format: Export format ('csv' or 'excel')
            
        Returns:
            Path to exported file
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{ticker}_history_{timestamp}.{format if format == 'csv' else 'xlsx'}"
        filepath = self.output_dir / filename
        
        if format == 'csv':
            data.to_csv(filepath)
        else:
            data.to_excel(filepath, engine='openpyxl')
        
        return str(filepath)
    
    def export_predictions(self, ticker: str, predictions: List[Dict], format: str = 'csv') -> str:
        """Export price predictions.
        
        Args:
            ticker: Stock ticker symbol
            predictions: List of prediction dictionaries
            format: Export format ('csv' or 'excel')
            
        Returns:
            Path to exported file
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{ticker}_predictions_{timestamp}.{format if format == 'csv' else 'xlsx'}"
        filepath = self.output_dir / filename
        
        df = pd.DataFrame(predictions)
        
        if format == 'csv':
            df.to_csv(filepath, index=False)
        else:
            df.to_excel(filepath, index=False, engine='openpyxl')
        
        return str(filepath)
    
    def export_backtest_results(self, strategy_name: str, results: Dict[str, Any], 
                               format: str = 'excel') -> str:
        """Export backtesting results.
        
        Args:
            strategy_name: Name of the strategy
            results: Backtesting results dictionary
            format: Export format ('csv' or 'excel')
            
        Returns:
            Path to exported file
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"backtest_{strategy_name}_{timestamp}.xlsx"
        filepath = self.output_dir / filename
        
        if format == 'excel':
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                # Summary sheet
                summary_df = pd.DataFrame([results.get('summary', {})])
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Trades sheet
                if 'trades' in results:
                    trades_df = pd.DataFrame(results['trades'])
                    trades_df.to_excel(writer, sheet_name='Trades', index=False)
                
                # Equity curve
                if 'equity_curve' in results:
                    equity_df = pd.DataFrame(results['equity_curve'])
                    equity_df.to_excel(writer, sheet_name='Equity Curve', index=False)
        else:
            # For CSV, just export summary
            summary_df = pd.DataFrame([results.get('summary', {})])
            filepath = self.output_dir / f"backtest_{strategy_name}_{timestamp}.csv"
            summary_df.to_csv(filepath, index=False)
        
        return str(filepath)
