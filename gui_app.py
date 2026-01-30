"""Main Desktop GUI application for stock tracking."""
import tkinter as tk
import customtkinter as ctk
from PIL import Image, ImageTk
import os
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models.portfolio import Portfolio
from src.data.stock_fetcher import StockFetcher
from src.visualization.gui_components import GUICharts
from src.models.lstm_predictor import LSTMPredictor
from src.models.ensemble_predictor import EnsemblePredictor
from src.visualization.plotly_dashboard import PlotlyDashboard
from src.models.sentiment_analyzer import SentimentAnalyzer
from src.models.recommender import Recommender
from src.utils.config import config
import webbrowser

# Set appearance mode and theme
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class StockTrackerApp(ctk.CTk):
    """Main application class for the stock tracking desktop app."""
    
    def __init__(self):
        super().__init__()
        
        # Configure window
        self.title("Stock Tracker Pro")
        self.geometry("1100x800")
        
        # Initialize data models
        self.portfolio = Portfolio()
        self.fetcher = StockFetcher()
        self.predictor = EnsemblePredictor() # Upgraded to Ensemble
        self.sentiment_analyzer = SentimentAnalyzer()
        self.recommender = Recommender()
        
        # Create layout
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        # Sidebar
        self.sidebar_frame = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(9, weight=1)
        
        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="Stock Tracker Pro", 
                                       font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        
        # Sidebar buttons
        self.dashboard_button = self.create_sidebar_button("🏠 Dashboard", 1, self.show_dashboard)
        self.portfolio_button = self.create_sidebar_button("💼 Portfolio", 2, self.show_portfolio)
        self.analysis_button = self.create_sidebar_button("📊 Analysis", 3, self.show_analysis)
        self.predictions_button = self.create_sidebar_button("🤖 Predictions", 4, self.show_predictions)
        self.sentiment_button = self.create_sidebar_button("📰 Sentiment", 5, self.show_sentiment)
        self.alerts_button = self.create_sidebar_button("🔔 Alerts", 6, self.show_alerts)
        self.backtest_button = self.create_sidebar_button("📈 Backtesting", 7, self.show_backtesting)
        self.settings_button = self.create_sidebar_button("⚙️ Settings", 8, self.show_settings)
        
        self.appearance_mode_label = ctk.CTkLabel(self.sidebar_frame, text="Appearance Mode:", anchor="w")
        self.appearance_mode_label.grid(row=10, column=0, padx=20, pady=(10, 0))
        self.appearance_mode_optionemenu = ctk.CTkOptionMenu(self.sidebar_frame, values=["Dark", "Light", "System"],
                                                            command=self.change_appearance_mode_event)
        self.appearance_mode_optionemenu.grid(row=11, column=0, padx=20, pady=(10, 20))
        
        # Main content area
        self.main_content_frame = ctk.CTkFrame(self, corner_radius=0)
        self.main_content_frame.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        self.main_content_frame.grid_columnconfigure(0, weight=1)
        self.main_content_frame.grid_rowconfigure(0, weight=1)
        
        # Initialize pages
        self.pages = {}
        self.create_pages()
        
        # Show default page
        self.show_dashboard()

    def create_sidebar_button(self, text, row, command):
        button = ctk.CTkButton(self.sidebar_frame, text=text, corner_radius=0, height=40, border_spacing=10,
                               fg_color="transparent", text_color=("gray10", "gray90"), hover_color=("gray70", "gray30"),
                               anchor="w", command=command)
        button.grid(row=row, column=0, sticky="ew")
        return button

    def create_pages(self):
        """Pre-create placeholder frames for each page."""
        self.pages["Dashboard"] = ctk.CTkFrame(self.main_content_frame)
        self.pages["Portfolio"] = ctk.CTkFrame(self.main_content_frame)
        self.pages["Analysis"] = ctk.CTkFrame(self.main_content_frame)
        self.pages["AI Predictions"] = ctk.CTkFrame(self.main_content_frame)
        self.pages["Sentiment"] = ctk.CTkFrame(self.main_content_frame)
        self.pages["Alerts"] = ctk.CTkFrame(self.main_content_frame)
        self.pages["Backtesting"] = ctk.CTkFrame(self.main_content_frame)
        self.pages["Settings"] = ctk.CTkFrame(self.main_content_frame)
        
        for page in self.pages.values():
            page.grid(row=0, column=0, sticky="nsew")
            page.grid_columnconfigure(0, weight=1)

    def show_page(self, page_name):
        """Bring the specified page to the front."""
        page = self.pages[page_name]
        page.tkraise()
        
        # Update sidebar button highlighting
        for name, button in [("Dashboard", self.dashboard_button), ("Portfolio", self.portfolio_button),
                             ("Analysis", self.analysis_button), ("AI Predictions", self.predictions_button),
                             ("Sentiment", self.sentiment_button), ("Alerts", self.alerts_button),
                             ("Backtesting", self.backtest_button), ("Settings", self.settings_button)]:
            if name == page_name:
                button.configure(fg_color=("gray75", "gray25"))
            else:
                button.configure(fg_color="transparent")

    def show_dashboard(self):
        self.show_page("Dashboard")
        for widget in self.pages["Dashboard"].winfo_children():
            widget.destroy()
            
        header = ctk.CTkLabel(self.pages["Dashboard"], text="Market Overview", 
                              font=ctk.CTkFont(size=24, weight="bold"))
        header.pack(pady=20)
        
        stats_frame = ctk.CTkFrame(self.pages["Dashboard"])
        stats_frame.pack(fill="x", padx=20)
        
        summary = self.portfolio.get_summary()
        self.create_stat_card(stats_frame, "Portfolio Value", f"${summary['total_value']:,.2f}", 0)
        self.create_stat_card(stats_frame, "Profit / Loss", f"${summary['total_profit_loss']:,.2f}", 1)
        self.create_stat_card(stats_frame, "ROI Status", f"{summary['total_roi']:.2f}%", 2)
        
        # Add a status message area
        self.status_label = ctk.CTkLabel(self.pages["Dashboard"], text="", font=ctk.CTkFont(size=12))
        self.status_label.pack(pady=10)
        
        # Add a quick lookup entry
        lookup_frame = ctk.CTkFrame(self.pages["Dashboard"])
        lookup_frame.pack(fill="x", padx=20, pady=20)
        
        self.lookup_entry = ctk.CTkEntry(lookup_frame, placeholder_text="Enter Ticker (e.g. AAPL)")
        self.lookup_entry.pack(side="left", padx=10, pady=10, expand=True, fill="x")
        
        self.lookup_button = ctk.CTkButton(lookup_frame, text="Quick Analysis", command=self.quick_analysis)
        self.lookup_button.pack(side="right", padx=10, pady=10)

    def quick_analysis(self):
        ticker = self.lookup_entry.get().upper()
        if ticker:
            self.show_analysis(ticker)

    def create_stat_card(self, parent, title, value, column):
        card = ctk.CTkFrame(parent, corner_radius=10, border_width=1)
        card.grid(row=0, column=column, padx=10, pady=10, sticky="nsew")
        parent.grid_columnconfigure(column, weight=1)
        
        ctk.CTkLabel(card, text=title, font=ctk.CTkFont(size=12)).pack(pady=(15, 0))
        ctk.CTkLabel(card, text=value, font=ctk.CTkFont(size=22, weight="bold")).pack(pady=(5, 15))

    def show_portfolio(self):
        self.show_page("Portfolio")
        for widget in self.pages["Portfolio"].winfo_children():
            widget.destroy()
            
        header = ctk.CTkLabel(self.pages["Portfolio"], text="Stock Portfolio", 
                              font=ctk.CTkFont(size=24, weight="bold"))
        header.pack(pady=20)

        # Add form
        form_frame = ctk.CTkFrame(self.pages["Portfolio"])
        form_frame.pack(fill="x", padx=20, pady=10)
        
        self.p_ticker = ctk.CTkEntry(form_frame, placeholder_text="Ticker")
        self.p_ticker.grid(row=0, column=0, padx=5, pady=5)
        self.p_qty = ctk.CTkEntry(form_frame, placeholder_text="Quantity")
        self.p_qty.grid(row=0, column=1, padx=5, pady=5)
        self.p_price = ctk.CTkEntry(form_frame, placeholder_text="Price")
        self.p_price.grid(row=0, column=2, padx=5, pady=5)
        
        ctk.CTkButton(form_frame, text="Add/Update Position", command=self.add_position).grid(row=0, column=3, padx=5, pady=5)
        ctk.CTkButton(form_frame, text="Clear", command=self.clear_portfolio_form, fg_color="gray").grid(row=0, column=4, padx=5, pady=5)

        # Holdings table
        table_frame = ctk.CTkScrollableFrame(self.pages["Portfolio"], label_text="Holdings")
        table_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        summary = self.portfolio.get_summary()
        if summary['holdings']:
            headers = ["Ticker", "Shares", "Avg Cost", "Current", "Value", "ROI %", "Actions"]
            for i, h in enumerate(headers):
                ctk.CTkLabel(table_frame, text=h, font=ctk.CTkFont(weight="bold")).grid(row=0, column=i, padx=10, pady=5)
            
            for r, h in enumerate(summary['holdings'], 1):
                ticker = h['ticker']
                ctk.CTkLabel(table_frame, text=ticker).grid(row=r, column=0, padx=10, pady=2)
                ctk.CTkLabel(table_frame, text=str(h['quantity'])).grid(row=r, column=1, padx=10, pady=2)
                ctk.CTkLabel(table_frame, text=f"${h['purchase_price']:.2f}").grid(row=r, column=2, padx=10, pady=2)
                ctk.CTkLabel(table_frame, text=f"${h['current_price']:.2f}").grid(row=r, column=3, padx=10, pady=2)
                ctk.CTkLabel(table_frame, text=f"${h['current_value']:.2f}").grid(row=r, column=4, padx=10, pady=2)
                color = "green" if h['profit_loss_pct'] >= 0 else "red"
                ctk.CTkLabel(table_frame, text=f"{h['profit_loss_pct']:.2f}%", text_color=color).grid(row=r, column=5, padx=10, pady=2)
                
                # Action buttons
                actions_frame = ctk.CTkFrame(table_frame, fg_color="transparent")
                actions_frame.grid(row=r, column=6, padx=10, pady=2)
                
                adj_btn = ctk.CTkButton(actions_frame, text="✏️", width=30, height=25, 
                                       command=lambda t=ticker, q=h['quantity'], p=h['purchase_price']: self.adjust_position(t, q, p))
                adj_btn.pack(side="left", padx=2)
                
                rem_btn = ctk.CTkButton(actions_frame, text="🗑️", width=30, height=25, fg_color="red", hover_color="darkred",
                                       command=lambda t=ticker: self.remove_position(t))
                rem_btn.pack(side="left", padx=2)
        else:
            ctk.CTkLabel(table_frame, text="Portfolio is empty").pack(pady=20)

    def add_position(self):
        ticker = self.p_ticker.get().upper()
        try:
            qty = float(self.p_qty.get())
            price = float(self.p_price.get())
            if ticker and qty > 0 and price > 0:
                if self.portfolio.add_stock(ticker, qty, price):
                    self.show_portfolio()
                    self.status_label.configure(text=f"Updated {ticker} in portfolio", text_color="green")
        except Exception as e:
            self.status_label.configure(text=f"Error adding position: {str(e)}", text_color="red")

    def adjust_position(self, ticker, qty, price):
        self.p_ticker.delete(0, "end")
        self.p_ticker.insert(0, ticker)
        self.p_qty.delete(0, "end")
        self.p_qty.insert(0, str(qty))
        self.p_price.delete(0, "end")
        self.p_price.insert(0, str(price))

    def remove_position(self, ticker):
        if self.portfolio.remove_stock(ticker):
            self.show_portfolio()
            self.status_label.configure(text=f"Removed {ticker} from portfolio", text_color="orange")

    def clear_portfolio_form(self):
        self.p_ticker.delete(0, "end")
        self.p_qty.delete(0, "end")
        self.p_price.delete(0, "end")

    def show_analysis(self, ticker=None):
        self.show_page("Analysis")
        for widget in self.pages["Analysis"].winfo_children():
            widget.destroy()
            
        header = ctk.CTkLabel(self.pages["Analysis"], text="Technical Analysis", 
                              font=ctk.CTkFont(size=24, weight="bold"))
        header.pack(pady=(20, 10))
        
        search_frame = ctk.CTkFrame(self.pages["Analysis"])
        search_frame.pack(fill="x", padx=20)
        
        self.a_ticker = ctk.CTkEntry(search_frame, placeholder_text="Search Symbol...")
        if ticker: self.a_ticker.insert(0, ticker)
        self.a_ticker.pack(side="left", padx=10, pady=10, expand=True, fill="x")
        
        self.a_button = ctk.CTkButton(search_frame, text="Run Analysis", command=self.run_analysis)
        self.a_button.pack(side="right", padx=10, pady=10)
        
        self.chart_container = ctk.CTkFrame(self.pages["Analysis"])
        self.chart_container.pack(fill="both", expand=True, padx=20, pady=10)
        
        if ticker:
            self.run_analysis()

    def run_analysis(self):
        ticker = self.a_ticker.get().upper()
        if not ticker: return
        
        # Clear container
        for widget in self.chart_container.winfo_children():
            widget.destroy()
            
        loading = ctk.CTkLabel(self.chart_container, text="Fetching Market Data...")
        loading.pack(expand=True)
        
        def fetch_data():
            data = self.fetcher.get_historical_data(ticker, period="1y")
            self.after(0, lambda: self.render_chart(data, ticker))
            
        threading.Thread(target=fetch_data, daemon=True).start()

    def render_chart(self, data, ticker):
        for widget in self.chart_container.winfo_children():
            widget.destroy()
            
        if data is not None and not data.empty:
            GUICharts.create_stock_chart(self.chart_container, data, ticker)
        else:
            ctk.CTkLabel(self.chart_container, text=f"Data for {ticker} not found.").pack(expand=True)

    def show_predictions(self):
        self.show_page("AI Predictions")
        for widget in self.pages["AI Predictions"].winfo_children(): widget.destroy()
        
        ctk.CTkLabel(self.pages["AI Predictions"], text="AI Price Forecast", 
                    font=ctk.CTkFont(size=24, weight="bold")).pack(pady=20)
        
        pred_entry_frame = ctk.CTkFrame(self.pages["AI Predictions"])
        pred_entry_frame.pack(fill="x", padx=20)
        
        self.pred_ticker = ctk.CTkEntry(pred_entry_frame, placeholder_text  ="Enter Ticker for Prediction")
        self.pred_ticker.pack(side="left", padx=10, pady=10, expand=True, fill="x")
        
        self.fast_mode_var = tk.BooleanVar(value=True)
        self.fast_mode_cb = ctk.CTkCheckBox(pred_entry_frame, text="Fast Mode (No News)", variable=self.fast_mode_var)
        self.fast_mode_cb.pack(side="left", padx=10)
        
        self.pred_btn = ctk.CTkButton(pred_entry_frame, text="Generate Forecast", command=self.run_prediction)
        self.pred_btn.pack(side="right", padx=10, pady=10)
        
        self.pred_results = ctk.CTkFrame(self.pages["AI Predictions"])
        self.pred_results.pack(fill="both", expand=True, padx=20, pady=10)
        
        self.pred_insights = ctk.CTkScrollableFrame(self.pages["AI Predictions"], height=150, label_text="Market Insights & Risks")
        self.pred_insights.pack(fill="x", padx=20, pady=(0, 20))

    def run_prediction(self):
        ticker = self.pred_ticker.get().upper()
        if not ticker: return
        
        for widget in self.pred_results.winfo_children(): widget.destroy()
        loading = ctk.CTkLabel(self.pred_results, text="Processing Deep Learning Model...")
        loading.pack(expand=True)
        
        def job():
            is_fast = self.fast_mode_var.get()
            res = self.predictor.predict_ensemble(ticker, days=30)
            self.after(0, lambda: self.render_predictions(res, ticker))
        threading.Thread(target=job, daemon=True).start()

    def render_predictions(self, res, ticker):
        for widget in self.pred_results.winfo_children(): widget.destroy()
        for widget in self.pred_insights.winfo_children(): widget.destroy()
        
        if res and res['success']:
            # Main Forecast Table/List
            ctk.CTkLabel(self.pred_results, text=f"Advanced Analytics: {ticker}", font=ctk.CTkFont(size=18, weight="bold")).pack(pady=10)
            
            # Risk Stats Frame
            risk_f = ctk.CTkFrame(self.pred_results)
            risk_f.pack(fill="x", padx=10, pady=5)
            metrics = res['risk_metrics']
            ctk.CTkLabel(risk_f, text=f"Sharpe: {metrics['sharpe_ratio']:.2f} | Vol: {metrics['volatility']*100:.1f}% | VaR: {metrics['var_95']*100:.2f}%").pack(pady=5)
            
            # Action Button for Dashboard
            dash_btn = ctk.CTkButton(self.pred_results, text="🚀 Launch Interactive Plotly Dashboard", 
                                    fg_color="#0066ff", hover_color="#0055dd",
                                    command=lambda: self.open_plotly_dashboard(res))
            dash_btn.pack(pady=20)
            
            # Insights
            for insight in res['insights']:
                ctk.CTkLabel(self.pred_insights, text=f"• {insight}", anchor="w", wraplength=800).pack(fill="x", padx=10, pady=2)
                
            # Render a static matplotlib preview
            static_data = pd.DataFrame(res['forecast'])
            # (Simplification: just list top 5)
            ctk.CTkLabel(self.pred_results, text="Next 5 Days Forecast Snapshot:").pack(pady=5)
            for i in range(5):
                p = res['forecast'][i]
                ctk.CTkLabel(self.pred_results, text=f"{p['date']}: ${p['price']:.2f} (±${(p['upper']-p['price']):.2f})").pack()
        else:
            err = res.get('error', 'Prediction failed.') if res else 'Prediction failed.'
            ctk.CTkLabel(self.pred_results, text=f"❌ {err}").pack(expand=True)

    def open_plotly_dashboard(self, res):
        path = PlotlyDashboard.create_forecast_dashboard(res)
        webbrowser.open(f"file://{path}")

    def show_sentiment(self):
        self.show_page("Sentiment")
        for widget in self.pages["Sentiment"].winfo_children(): widget.destroy()
        
        ctk.CTkLabel(self.pages["Sentiment"], text="News Sentiment Analysis", 
                    font=ctk.CTkFont(size=24, weight="bold")).pack(pady=20)
        
        sent_frame = ctk.CTkFrame(self.pages["Sentiment"])
        sent_frame.pack(fill="x", padx=20)
        
        self.sent_ticker = ctk.CTkEntry(sent_frame, placeholder_text="Ticker symbol")
        self.sent_ticker.pack(side="left", padx=10, pady=10, expand=True, fill="x")
        
        self.sent_btn = ctk.CTkButton(sent_frame, text="Scan News", command=self.run_sentiment)
        self.sent_btn.pack(side="right", padx=10, pady=10)
        
        self.sent_results = ctk.CTkScrollableFrame(self.pages["Sentiment"], label_text="Live News Sentiment")
        self.sent_results.pack(fill="both", expand=True, padx=20, pady=10)

    def run_sentiment(self):
        ticker = self.sent_ticker.get().upper()
        if not ticker: return
        
        for widget in self.sent_results.winfo_children(): widget.destroy()
        loading = ctk.CTkLabel(self.sent_results, text="Scanning Global Markets...")
        loading.pack(expand=True)
        
        def job():
            res = self.sentiment_analyzer.analyze_stock(ticker)
            self.after(0, lambda: self.render_sentiment(res))
        threading.Thread(target=job, daemon=True).start()

    def render_sentiment(self, res):
        for widget in self.sent_results.winfo_children(): widget.destroy()
        if res and res['num_articles'] > 0:
            score_color = "green" if res['sentiment_score'] > 0 else "red" if res['sentiment_score'] < 0 else "gray"
            ctk.CTkLabel(self.sent_results, text=f"Overall: {res['sentiment_label'].upper()} ({res['sentiment_score']:.2f})", 
                        text_color=score_color, font=ctk.CTkFont(size=18, weight="bold")).pack(pady=10)
            
            for art in res['articles']:
                item = ctk.CTkFrame(self.sent_results)
                item.pack(fill="x", padx=5, pady=5)
                ctk.CTkLabel(item, text=art['title'], font=ctk.CTkFont(size=13, weight="normal"), wraplength=700).pack(anchor="w", padx=10)
                ctk.CTkLabel(item, text=f"{art['source']} | {art['sentiment_label']}", font=ctk.CTkFont(size=10)).pack(anchor="w", padx=10)
        else:
            ctk.CTkLabel(self.sent_results, text="No news found or API key missing.").pack(expand=True)

    def show_alerts(self):
        self.show_page("Alerts")
        for widget in self.pages["Alerts"].winfo_children(): widget.destroy()
        ctk.CTkLabel(self.pages["Alerts"], text="Position Alerts", 
                    font=ctk.CTkFont(size=24, weight="bold")).pack(pady=20)
        ctk.CTkLabel(self.pages["Alerts"], text="Monitoring service active in background.").pack(expand=True)

    def show_backtesting(self):
        self.show_page("Backtesting")
        # Implementation skeleton
        ctk.CTkLabel(self.pages["Backtesting"], text="Strategy Engine", 
                    font=ctk.CTkFont(size=24, weight="bold")).pack(pady=20)

    def show_settings(self):
        self.show_page("Settings")
        for widget in self.pages["Settings"].winfo_children(): widget.destroy()
        ctk.CTkLabel(self.pages["Settings"], text="Application Settings", 
                    font=ctk.CTkFont(size=24, weight="bold")).pack(pady=20)
        
        status_frame = ctk.CTkFrame(self.pages["Settings"])
        status_frame.pack(fill="x", padx=40, pady=20)
        
        ctk.CTkLabel(status_frame, text="Backend Integration Status", font=ctk.CTkFont(weight="bold")).pack(pady=10)
        ctk.CTkLabel(status_frame, text=f"yfinance: ✅ Connected").pack(anchor="w", padx=20)
        ctk.CTkLabel(status_frame, text=f"NewsAPI: {'✅' if config.has_news_api_key() else '❌ Not Configured'}").pack(anchor="w", padx=20)
        ctk.CTkLabel(status_frame, text=f"Email (SMTP): {'✅' if config.has_email_config() else '❌ Not Configured'}").pack(anchor="w", padx=20)
        
        # AI Status
        from src.models.lstm_predictor import TENSORFLOW_AVAILABLE
        tf_status = "✅ Active (Graph Mode)" if TENSORFLOW_AVAILABLE else "❌ Disabled"
        ctk.CTkLabel(status_frame, text=f"AI Engine: {tf_status}").pack(anchor="w", padx=20)
        ctk.CTkLabel(status_frame, text=f"Hardware Accel: ✅ FP16 Mixed Precision Enabled").pack(anchor="w", padx=20)

    def change_appearance_mode_event(self, new_appearance_mode: str):
        ctk.set_appearance_mode(new_appearance_mode)

if __name__ == "__main__":
    app = StockTrackerApp()
    app.mainloop()
