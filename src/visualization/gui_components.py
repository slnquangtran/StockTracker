"""GUI specific charting utility for the desktop app."""
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
from typing import Optional
import tkinter as tk
from src.analysis.technical_indicators import TechnicalIndicators

class GUICharts:
    """Helper class to embed Matplotlib charts into Tkinter/CustomTkinter."""
    
    @staticmethod
    def create_stock_chart(canvas_parent: tk.Widget, data: pd.DataFrame, ticker: str):
        """Create a technical analysis chart using Matplotlib."""
        # Add indicators
        data = TechnicalIndicators.add_all_indicators(data)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
        plt.subplots_adjust(hspace=0.05)
        
        # Style
        plt.style.use('dark_background')
        fig.patch.set_facecolor('#2b2b2b')
        ax1.set_facecolor('#2b2b2b')
        ax2.set_facecolor('#2b2b2b')
        
        # Price and MAs
        ax1.plot(data.index, data['Close'], label='Close', color='white', linewidth=1)
        ax1.plot(data.index, data['SMA_20'], label='SMA 20', color='orange', alpha=0.7)
        ax1.plot(data.index, data['SMA_50'], label='SMA 50', color='cyan', alpha=0.7)
        
        # Bollinger Bands
        ax1.fill_between(data.index, data['BB_Upper'], data['BB_Lower'], color='gray', alpha=0.2, label='Bollinger Bands')
        
        ax1.set_title(f"{ticker} Technical Analysis", color='white', fontsize=12)
        ax1.legend(loc='best', fontsize=8)
        ax1.grid(True, alpha=0.1)
        
        # RSI
        ax2.plot(data.index, data['RSI'], color='purple', linewidth=1)
        ax2.axhline(70, color='red', linestyle='--', alpha=0.5)
        ax2.axhline(30, color='green', linestyle='--', alpha=0.5)
        ax2.set_ylabel('RSI', color='white', fontsize=10)
        ax2.set_ylim(0, 100)
        ax2.grid(True, alpha=0.1)
        
        # Embed in Tkinter
        canvas = FigureCanvasTkAgg(fig, master=canvas_parent)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill="both", expand=True)
        canvas.draw()
        
        return canvas
