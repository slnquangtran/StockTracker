"""Interactive charts using Plotly and Matplotlib."""
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from src.analysis.technical_indicators import TechnicalIndicators

class Charts:
    """Create interactive stock charts."""
    
    @staticmethod
    def candlestick_chart(data: pd.DataFrame, ticker: str, 
                         show_volume: bool = True) -> go.Figure:
        """Create candlestick chart with volume.
        
        Args:
            data: DataFrame with OHLCV data
            ticker: Stock ticker symbol
            show_volume: Whether to show volume subplot
            
        Returns:
            Plotly figure
        """
        if show_volume:
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                row_heights=[0.7, 0.3],
                subplot_titles=(f'{ticker} Price', 'Volume')
            )
        else:
            fig = go.Figure()
        
        # Candlestick with professional colors
        candlestick = go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='Price',
            increasing_line_color='#10B981',  # Success green
            decreasing_line_color='#EF4444',  # Danger red
            increasing_fillcolor='rgba(16, 185, 129, 0.4)',
            decreasing_fillcolor='rgba(239, 68, 68, 0.4)',
            line=dict(width=1)
        )
        
        if show_volume:
            fig.add_trace(candlestick, row=1, col=1)
            
            # Volume bars with professional theme
            colors = ['#EF4444' if data['Close'].iloc[i] < data['Open'].iloc[i] 
                     else '#10B981' for i in range(len(data))]
            volume = go.Bar(
                x=data.index,
                y=data['Volume'],
                name='Volume',
                marker_color=colors,
                opacity=0.6
            )
            fig.add_trace(volume, row=2, col=1)
        else:
            fig.add_trace(candlestick)
        
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Inter', color='#94A3B8'),
            showlegend=False,
            height=600,
            margin=dict(l=10, r=10, t=50, b=10),
            hovermode='x unified',
            xaxis=dict(
                showgrid=True, gridcolor='rgba(255,255,255,0.05)',
                rangeslider_visible=False,
                showspikes=True, spikemode='across', spikesnap='cursor',
                spikedash='dash', spikecolor='rgba(255,255,255,0.3)', spikethickness=1
            ),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', tickprefix='$')
        )
        return fig
    
    @staticmethod
    def technical_indicators_chart(data: pd.DataFrame, ticker: str) -> go.Figure:
        """Create chart with technical indicators.
        
        Args:
            data: DataFrame with price data and indicators
            ticker: Stock ticker symbol
            
        Returns:
            Plotly figure
        """
        # Add indicators if not present
        if 'SMA_20' not in data.columns:
            data = TechnicalIndicators.add_all_indicators(data)
        
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.5, 0.25, 0.25],
            subplot_titles=(f'{ticker} with Moving Averages', 'RSI', 'MACD')
        )
        
        # Price and MAs
        fig.add_trace(go.Scatter(x=data.index, y=data['Close'], 
                                name='Close', line=dict(color='white')), row=1, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data['SMA_20'], 
                                name='SMA 20', line=dict(color='orange')), row=1, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data['SMA_50'], 
                                name='SMA 50', line=dict(color='blue')), row=1, col=1)
        
        # Bollinger Bands
        fig.add_trace(go.Scatter(x=data.index, y=data['BB_Upper'], 
                                name='BB Upper', line=dict(color='gray', dash='dash')), 
                     row=1, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data['BB_Lower'], 
                                name='BB Lower', line=dict(color='gray', dash='dash'), 
                                fill='tonexty'), row=1, col=1)
        
        # RSI
        fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], 
                                name='RSI', line=dict(color='purple')), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        # MACD
        fig.add_trace(go.Scatter(x=data.index, y=data['MACD'], 
                                name='MACD', line=dict(color='blue')), row=3, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=data['MACD_Signal'], 
                                name='Signal', line=dict(color='orange')), row=3, col=1)
        fig.add_trace(go.Bar(x=data.index, y=data['MACD_Hist'], 
                            name='Histogram'), row=3, col=1)
        
        fig.update_layout(
            height=800,
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Inter', color='#94A3B8', size=14),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=14)),
            hovermode='x unified',
            xaxis=dict(
                showgrid=True, gridcolor='rgba(255,255,255,0.1)', 
                showspikes=True, title_font=dict(size=16)
            ),
            yaxis=dict(
                showgrid=True, gridcolor='rgba(255,255,255,0.1)', 
                tickprefix='$', title_font=dict(size=16)
            )
        )
        return fig
    
    @staticmethod
    def prediction_chart(historical_data: pd.DataFrame, predictions: List[Dict],
                        ticker: str) -> go.Figure:
        """Create chart showing historical data and predictions.
        
        Args:
            historical_data: Historical price data
            predictions: List of prediction dictionaries
            ticker: Stock ticker symbol
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        # Historical data with thicker line for visibility
        fig.add_trace(go.Scatter(
            x=historical_data.index,
            y=historical_data['Close'],
            name='Historical',
            line=dict(color='white', width=3)
        ))
        
        # Predictions
        pred_dates = [pd.Timestamp(p['date']) for p in predictions]
        pred_prices = [p['predicted_price'] for p in predictions]
        
        fig.add_trace(go.Scatter(
            x=pred_dates,
            y=pred_prices,
            name='Predicted',
            line=dict(color='#3B82F6', dash='dash', width=4),
            mode='lines+markers',
            marker=dict(size=10)
        ))
        
        fig.update_layout(
            title=f'{ticker} Price Prediction (7-day forecast)',
            xaxis_title='Date',
            yaxis_title='Price',
            template='plotly_dark',
            height=500
        )
        
        return fig
    
    @staticmethod
    def portfolio_performance_chart(equity_curve: List[Dict]) -> go.Figure:
        """Create portfolio performance chart.
        
        Args:
            equity_curve: List of equity values over time
            
        Returns:
            Plotly figure
        """
        df = pd.DataFrame(equity_curve)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['equity'],
            name='Portfolio Value',
            line=dict(color='green'),
            fill='tozeroy'
        ))
        
        fig.update_layout(
            title='Portfolio Performance',
            xaxis_title='Date',
            yaxis_title='Equity ($)',
            template='plotly_dark',
            height=400
        )
        
        return fig
    
    @staticmethod
    def sentiment_timeline_chart(sentiment_trend: List[Dict], ticker: str) -> go.Figure:
        """Create sentiment timeline chart.
        
        Args:
            sentiment_trend: List of daily sentiment scores
            ticker: Stock ticker symbol
            
        Returns:
            Plotly figure
        """
        df = pd.DataFrame(sentiment_trend)
        
        # Color based on sentiment
        colors = ['green' if s > 0.05 else 'red' if s < -0.05 else 'gray' 
                 for s in df['sentiment_score']]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=df['date'],
            y=df['sentiment_score'],
            name='Sentiment',
            marker_color=colors
        ))
        
        fig.add_hline(y=0, line_dash="dash", line_color="white")
        
        fig.update_layout(
            title=f'{ticker} Sentiment Trend',
            xaxis_title='Date',
            yaxis_title='Sentiment Score',
            template='plotly_dark',
            height=400
        )
        
        return fig
    
    @staticmethod
    def efficient_frontier_chart(frontier_data: pd.DataFrame) -> go.Figure:
        """Create efficient frontier chart.
        
        Args:
            frontier_data: DataFrame with portfolio statistics
            
        Returns:
            Plotly figure
        """
        fig = go.Figure()
        
        # Color by Sharpe ratio
        fig.add_trace(go.Scatter(
            x=frontier_data['volatility'],
            y=frontier_data['return'],
            mode='markers',
            marker=dict(
                size=8,
                color=frontier_data['sharpe'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Sharpe Ratio")
            ),
            text=[f"Return: {r:.2%}<br>Vol: {v:.2%}<br>Sharpe: {s:.2f}" 
                  for r, v, s in zip(frontier_data['return'], 
                                    frontier_data['volatility'],
                                    frontier_data['sharpe'])],
            hoverinfo='text'
        ))
        
        fig.update_layout(
            title='Efficient Frontier',
            xaxis_title='Volatility (Risk)',
            yaxis_title='Expected Return',
            template='plotly_dark',
            height=500
        )
        
        return fig
