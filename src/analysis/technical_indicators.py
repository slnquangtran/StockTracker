"""Technical indicators implementation."""
import pandas as pd
import numpy as np
from typing import Optional

class TechnicalIndicators:
    """Calculate technical indicators for stock analysis."""
    
    @staticmethod
    def sma(data: pd.DataFrame, period: int = 20, column: str = 'Close') -> pd.Series:
        """Calculate Simple Moving Average.
        
        Args:
            data: DataFrame with stock data
            period: Period for SMA
            column: Column to calculate SMA on
            
        Returns:
            Series with SMA values
        """
        return data[column].rolling(window=period).mean()
    
    @staticmethod
    def ema(data: pd.DataFrame, period: int = 20, column: str = 'Close') -> pd.Series:
        """Calculate Exponential Moving Average.
        
        Args:
            data: DataFrame with stock data
            period: Period for EMA
            column: Column to calculate EMA on
            
        Returns:
            Series with EMA values
        """
        return data[column].ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def rsi(data: pd.DataFrame, period: int = 14, column: str = 'Close') -> pd.Series:
        """Calculate Relative Strength Index.
        
        Args:
            data: DataFrame with stock data
            period: Period for RSI (typically 14)
            column: Column to calculate RSI on
            
        Returns:
            Series with RSI values (0-100)
        """
        delta = data[column].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def macd(data: pd.DataFrame, fast: int = 12, slow: int = 26, 
             signal: int = 9, column: str = 'Close') -> pd.DataFrame:
        """Calculate MACD (Moving Average Convergence Divergence).
        
        Args:
            data: DataFrame with stock data
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line period
            column: Column to calculate MACD on
            
        Returns:
            DataFrame with MACD, Signal, and Histogram
        """
        ema_fast = data[column].ewm(span=fast, adjust=False).mean()
        ema_slow = data[column].ewm(span=slow, adjust=False).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return pd.DataFrame({
            'MACD': macd_line,
            'Signal': signal_line,
            'Histogram': histogram
        })
    
    @staticmethod
    def bollinger_bands(data: pd.DataFrame, period: int = 20, std_dev: int = 2,
                       column: str = 'Close') -> pd.DataFrame:
        """Calculate Bollinger Bands.
        
        Args:
            data: DataFrame with stock data
            period: Period for moving average
            std_dev: Number of standard deviations
            column: Column to calculate bands on
            
        Returns:
            DataFrame with Upper, Middle, and Lower bands
        """
        middle_band = data[column].rolling(window=period).mean()
        std = data[column].rolling(window=period).std()
        
        upper_band = middle_band + (std * std_dev)
        lower_band = middle_band - (std * std_dev)
        
        return pd.DataFrame({
            'Upper': upper_band,
            'Middle': middle_band,
            'Lower': lower_band
        })
    
    @staticmethod
    def atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range.
        
        Args:
            data: DataFrame with OHLC data
            period: Period for ATR
            
        Returns:
            Series with ATR values
        """
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        return atr
    
    @staticmethod
    def stochastic_oscillator(data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate Stochastic Oscillator.
        
        Args:
            data: DataFrame with OHLC data
            period: Period for calculation
            
        Returns:
            DataFrame with %K and %D values
        """
        low_min = data['Low'].rolling(window=period).min()
        high_max = data['High'].rolling(window=period).max()
        
        k_percent = 100 * ((data['Close'] - low_min) / (high_max - low_min))
        d_percent = k_percent.rolling(window=3).mean()
        
        return pd.DataFrame({
            '%K': k_percent,
            '%D': d_percent
        })
    
    @staticmethod
    def obv(data: pd.DataFrame) -> pd.Series:
        """Calculate On-Balance Volume.
        
        Args:
            data: DataFrame with Close and Volume data
            
        Returns:
            Series with OBV values
        """
        obv = (np.sign(data['Close'].diff()) * data['Volume']).fillna(0).cumsum()
        return obv
    
    @staticmethod
    def add_all_indicators(data: pd.DataFrame) -> pd.DataFrame:
        """Add all technical indicators to DataFrame.
        
        Args:
            data: DataFrame with OHLC data
            
        Returns:
            DataFrame with all indicators added
        """
        result = data.copy()
        
        # Moving averages
        result['SMA_20'] = TechnicalIndicators.sma(data, 20)
        result['SMA_50'] = TechnicalIndicators.sma(data, 50)
        result['SMA_200'] = TechnicalIndicators.sma(data, 200)
        result['EMA_12'] = TechnicalIndicators.ema(data, 12)
        result['EMA_26'] = TechnicalIndicators.ema(data, 26)
        
        # RSI
        result['RSI'] = TechnicalIndicators.rsi(data)
        
        # MACD
        macd_data = TechnicalIndicators.macd(data)
        result['MACD'] = macd_data['MACD']
        result['MACD_Signal'] = macd_data['Signal']
        result['MACD_Hist'] = macd_data['Histogram']
        
        # Bollinger Bands
        bb_data = TechnicalIndicators.bollinger_bands(data)
        result['BB_Upper'] = bb_data['Upper']
        result['BB_Middle'] = bb_data['Middle']
        result['BB_Lower'] = bb_data['Lower']
        
        # ATR
        result['ATR'] = TechnicalIndicators.atr(data)
        
        # Stochastic
        stoch_data = TechnicalIndicators.stochastic_oscillator(data)
        result['Stoch_K'] = stoch_data['%K']
        result['Stoch_D'] = stoch_data['%D']
        
        # OBV
        result['OBV'] = TechnicalIndicators.obv(data)
        
        return result
