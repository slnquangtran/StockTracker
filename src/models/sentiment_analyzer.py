"""Sentiment analysis for news and social media."""
import numpy as np
from typing import List, Dict, Any
from datetime import datetime

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: Transformers not available. Using VADER for sentiment analysis.")

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    print("Warning: VADER not available. Installing: pip install vaderSentiment")

from src.data.news_fetcher import NewsFetcher

class SentimentAnalyzer:
    """Analyze sentiment from news and social media with caching."""
    
    def __init__(self, use_finbert: bool = False):
        """Initialize sentiment analyzer.
        
        Args:
            use_finbert: Use FinBERT model (slower but more accurate for finance)
        """
        self.news_fetcher = NewsFetcher()
        self.use_finbert = use_finbert and TRANSFORMERS_AVAILABLE
        self._cache = {} # Simple sentiment cache: {ticker: (timestamp, result)}
        
        if self.use_finbert:
            try:
                self.sentiment_pipeline = pipeline(
                    "sentiment-analysis",
                    model="ProsusAI/finbert",
                    tokenizer="ProsusAI/finbert"
                )
            except:
                print("FinBERT not available, falling back to VADER")
                self.use_finbert = False
        
        if not self.use_finbert and VADER_AVAILABLE:
            self.vader = SentimentIntensityAnalyzer()
        else:
            self.vader = None
    
    def analyze_text(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of a single text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment scores
        """
        if self.use_finbert:
            return self._analyze_with_finbert(text)
        elif self.vader:
            return self._analyze_with_vader(text)
        else:
            return {'score': 0.0, 'label': 'neutral', 'confidence': 0.0}
    
    def _analyze_with_finbert(self, text: str) -> Dict[str, float]:
        """Analyze using FinBERT model.
        
        Args:
            text: Text to analyze
            
        Returns:
            Sentiment dictionary
        """
        # Truncate text if too long
        text = text[:512]
        
        result = self.sentiment_pipeline(text)[0]
        
        # Convert to score (-1 to 1)
        label = result['label'].lower()
        confidence = result['score']
        
        if label == 'positive':
            score = confidence
        elif label == 'negative':
            score = -confidence
        else:
            score = 0.0
        
        return {
            'score': score,
            'label': label,
            'confidence': confidence
        }
    
    def _analyze_with_vader(self, text: str) -> Dict[str, float]:
        """Analyze using VADER.
        
        Args:
            text: Text to analyze
            
        Returns:
            Sentiment dictionary
        """
        scores = self.vader.polarity_scores(text)
        compound = scores['compound']
        
        if compound >= 0.05:
            label = 'positive'
        elif compound <= -0.05:
            label = 'negative'
        else:
            label = 'neutral'
        
        return {
            'score': compound,
            'label': label,
            'confidence': abs(compound)
        }
    
    def analyze_stock(self, ticker: str, company_name: str = None,
                     days: int = 7) -> Dict[str, Any]:
        """Analyze sentiment for a stock.
        
        Args:
            ticker: Stock ticker symbol
            company_name: Company name for better search
            days: Number of days to look back
            
        Returns:
            Dictionary with aggregated sentiment
        """
        # Check cache (1 hour expiry)
        now = datetime.now().timestamp()
        if ticker in self._cache:
            ts, result = self._cache[ticker]
            if now - ts < 3600: # 1 hour
                return result

        # Fetch news
        articles = self.news_fetcher.fetch_news(ticker, company_name, days)
        
        if not articles:
            res = {
                'ticker': ticker,
                'sentiment_score': 0.0,
                'sentiment_label': 'neutral',
                'num_articles': 0,
                'articles': []
            }
            self._cache[ticker] = (now, res)
            return res
        
        # Analyze each article
        sentiments = []
        analyzed_articles = []
        
        for article in articles:
            sentiment = self.analyze_text(article['text'])
            sentiments.append(sentiment['score'])
            
            analyzed_articles.append({
                'title': article['title'],
                'source': article['source'],
                'published_at': article['published_at'],
                'sentiment_score': sentiment['score'],
                'sentiment_label': sentiment['label'],
                'url': article['url']
            })
        
        # Aggregate sentiment
        avg_sentiment = np.mean(sentiments)
        
        if avg_sentiment >= 0.05:
            overall_label = 'positive'
        elif avg_sentiment <= -0.05:
            overall_label = 'negative'
        else:
            overall_label = 'neutral'
        
        res = {
            'ticker': ticker,
            'sentiment_score': float(avg_sentiment),
            'sentiment_label': overall_label,
            'num_articles': len(articles),
            'positive_count': sum(1 for s in sentiments if s > 0.05),
            'negative_count': sum(1 for s in sentiments if s < -0.05),
            'neutral_count': sum(1 for s in sentiments if -0.05 <= s <= 0.05),
            'articles': analyzed_articles[:10]  # Top 10 articles
        }
        
        self._cache[ticker] = (now, res)
        return res
    
    def get_sentiment_trend(self, ticker: str, company_name: str = None,
                           days: int = 30) -> List[Dict]:
        """Get sentiment trend over time.
        
        Args:
            ticker: Stock ticker symbol
            company_name: Company name
            days: Number of days to analyze
            
        Returns:
            List of daily sentiment scores
        """
        articles = self.news_fetcher.fetch_news(ticker, company_name, days)
        
        if not articles:
            return []
        
        # Group by date
        daily_sentiments = {}
        
        for article in articles:
            date = article['published_at'][:10]  # YYYY-MM-DD
            sentiment = self.analyze_text(article['text'])
            
            if date not in daily_sentiments:
                daily_sentiments[date] = []
            daily_sentiments[date].append(sentiment['score'])
        
        # Calculate daily averages
        trend = []
        for date, scores in sorted(daily_sentiments.items()):
            trend.append({
                'date': date,
                'sentiment_score': float(np.mean(scores)),
                'num_articles': len(scores)
            })
        
        return trend
    
    def is_configured(self) -> bool:
        """Check if sentiment analysis is properly configured.
        
        Returns:
            True if configured
        """
        return (self.use_finbert or self.vader is not None) and self.news_fetcher.is_configured()
