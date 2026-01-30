"""Example usage of the stock tracker components."""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.portfolio import Portfolio
from src.data.stock_fetcher import StockFetcher
from src.models.recommender import Recommender
from src.analysis.backtester import Backtester
from src.utils.config import config

def portfolio_example():
    print("\n--- Portfolio Example ---")
    portfolio = Portfolio("Test Portfolio")
    
    # Add some stocks
    print("Adding stocks to portfolio...")
    portfolio.add_stock("AAPL", quantity=10, purchase_price=180.0)
    portfolio.add_stock("MSFT", quantity=5, purchase_price=350.0)
    
    # Get summary
    summary = portfolio.get_summary()
    print(f"Portfolio: {summary['name']}")
    print(f"Total Value: ${summary['total_value']:.2f}")
    print(f"Total Profit/Loss: ${summary['total_profit_loss']:.2f} ({summary['total_roi']:.2f}%)")
    
    for holding in summary['holdings']:
        print(f"  {holding['ticker']}: {holding['quantity']} shares @ ${holding['current_price']:.2f}")

def analysis_example():
    print("\n--- Analysis & Recommendation Example ---")
    ticker = "TSLA"
    
    # Get recommendation
    print(f"Analyzing {ticker}...")
    recommender = Recommender()
    result = recommender.get_recommendation(ticker)
    
    print(f"Recommendation: {result['recommendation']}")
    print(f"Confidence: {result['confidence']:.0%}")
    print("\nDetails:")
    print(result['summary'])

def backtest_example():
    print("\n--- Backtesting Example ---")
    ticker = "AAPL"
    strategy = "sma_crossover"
    
    print(f"Backtesting {strategy} on {ticker}...")
    backtester = Backtester()
    results = backtester.run_strategy(ticker, strategy)
    
    if 'error' not in results:
        print(f"Initial Capital: ${results['initial_capital']}")
        print(f"Final Equity: ${results['final_equity']:.2f}")
        print(f"Total Return: {results['total_return_pct']:.2f}%")
        print(f"Number of Trades: {results['num_trades']}")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    else:
        print(f"Error: {results['error']}")

if __name__ == "__main__":
    # Note: These examples will attempt to fetch real data using yfinance
    try:
        portfolio_example()
        analysis_example()
        backtest_example()
    except Exception as e:
        print(f"An error occurred during examples: {e}")
        print("Note: Some features require API keys in a .env file to work fully.")
