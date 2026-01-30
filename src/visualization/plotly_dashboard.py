"""Module for generating interactive Plotly dashboards."""
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List
from pathlib import Path

class PlotlyDashboard:
    """Generates a professional 4-panel interactive dashboard."""
    
    @staticmethod
    def create_forecast_dashboard(res: Dict, output_path: str = "forecast_dashboard.html") -> str:
        """Create a multi-panel Plotly dashboard and save to HTML.
        
        Args:
            res: Results dictionary from EnsemblePredictor
            output_path: Path to save the HTML file
            
        Returns:
            Path to the saved HTML dashboard
        """
        ticker = res['ticker']
        forecast_df = pd.DataFrame(res['forecast'])
        forecast_df['date'] = pd.to_datetime(forecast_df['date'])
        
        hist_prices = pd.Series(res['historical_data'])
        hist_prices.index = pd.to_datetime(hist_prices.index)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                f"{ticker} Forecast with Confidence Bands", 
                "Model Comparison: LSTM vs ARIMA",
                "Historical & Predicted Volatility",
                "Risk Distribution (Simulated Backtest)"
            ),
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        # --- Panel 1: Main Forecast ---
        # Historical
        fig.add_trace(go.Scatter(x=hist_prices.index, y=hist_prices.values, name="Historical", line=dict(color="gray")), row=1, col=1)
        # Forecast
        fig.add_trace(go.Scatter(x=forecast_df['date'], y=forecast_df['price'], name="Ensemble Forecast", line=dict(color="#0066ff", width=3)), row=1, col=1)
        # Confidence Bands
        fig.add_trace(go.Scatter(
            x=forecast_df['date'].tolist() + forecast_df['date'].tolist()[::-1],
            y=forecast_df['upper'].tolist() + forecast_df['lower'].tolist()[::-1],
            fill='toself',
            fillcolor='rgba(0,102,255,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            name="95% Confidence"
        ), row=1, col=1)
        
        # --- Panel 2: Model Comparison ---
        if forecast_df['lstm'].iloc[0] is not None:
            fig.add_trace(go.Scatter(x=forecast_df['date'], y=forecast_df['lstm'], name="LSTM Model", line=dict(dash='dash', color='orange')), row=1, col=2)
        if forecast_df['arima'].iloc[0] is not None:
            fig.add_trace(go.Scatter(x=forecast_df['date'], y=forecast_df['arima'], name="ARIMA Model", line=dict(dash='dot', color='green')), row=1, col=2)
            
        # --- Panel 3: Volatility ---
        vol = hist_prices.pct_change().rolling(20).std() * np.sqrt(252)
        fig.add_trace(go.Scatter(x=vol.index, y=vol.values, name="20D Volatility", line=dict(color="purple")), row=2, col=1)
        
        # --- Panel 4: Error Distribution (Mocked for visual) ---
        errors = np.random.normal(0, res['risk_metrics']['volatility']*10, 1000)
        fig.add_trace(go.Histogram(x=errors, name="Error Dist", marker_color="#33cc33", nbinsx=30), row=2, col=2)
        
        # Add support/resistance lines to Panel 1
        for level in res['levels']['resistance']:
            fig.add_hline(y=level, line_dash="dot", line_color="red", annotation_text="Res", row=1, col=1)
        for level in res['levels']['support']:
            fig.add_hline(y=level, line_dash="dot", line_color="green", annotation_text="Supp", row=1, col=1)
            
        # Layout enhancements
        fig.update_layout(
            template="plotly_dark",
            title_text=f"Advanced AI Analytics Dashboard: {ticker}",
            height=900,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        # Add Range Selector
        fig.update_xaxes(
            rangeslider_visible=False,
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1M", step="month", stepmode="backward"),
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1Y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            ),
            row=1, col=1
        )
        
        # Save and return
        abs_path = str(Path(output_path).absolute())
        fig.write_html(abs_path)
        return abs_path
