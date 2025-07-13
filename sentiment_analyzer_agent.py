import time
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
import statistics
import re
from datetime import datetime, timedelta
import asyncio
import json

# For NLP analysis
try:
    from textblob import TextBlob
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except ImportError:
    print("Install textblob and vaderSentiment: pip install textblob vaderSentiment")

@dataclass
class SentimentReading:
    """Single sentiment measurement with metadata"""
    timestamp: float
    score: float
    source: str
    confidence: float
    raw_data: Dict = field(default_factory=dict)
    metadata: Dict = field(default_factory=dict)

@dataclass
class SentimentEvent:
    """Event-triggered sentiment analysis"""
    trigger_type: str  # "volatility_spike", "fear_rise", "manual"
    trigger_value: float
    sentiment_reading: SentimentReading

@dataclass
class SentimentScanAgent:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        
        # Core sentiment tracking
        self.last_sentiment_score = 0.0
        self.sentiment_history = deque(maxlen=self.config['history_size'])
        
        # Multi-source system
        self.sources = {
            "news": [],
            "social": [],
            "funding": [],
            "options": [],
            "technical": []
        }
        
        # Source weights (dynamic)
        self.source_weights = {
            "funding": 0.3,
            "social": 0.4,
            "news": 0.2,
            "options": 0.1
        }
        
        # NLP processors
        self.textblob_analyzer = None
        self.vader_analyzer = None
        
        # Trust and filtering
        self.trust_thresholds = {
            "min_followers": 100,
            "min_engagement": 10,
            "spam_keywords": ["airdrop", "free", "pump", "moon", "shill"]
        }
        
        # Event triggers
        self.event_triggers = {
            "volatility_threshold": 0.03,
            "fear_threshold": 0.4,
            "sentiment_change_threshold": 0.15
        }
        
        # Historical embedding for learning
        self.performance_history = deque(maxlen=1000)
        
        # Rate limiting
        self.last_fetch_time = 0
        self.min_fetch_interval = self.config.get('min_fetch_interval', 30)  # seconds
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
    def _default_config(self) -> Dict:
        return {
            'history_size': 100,
            'min_fetch_interval': 30,
            'enable_social_scraping': False,
            'enable_news_scraping': False,
            'sentiment_smoothing': 0.1,
            'divergence_threshold': 0.3
        }
    
    def fetch_sentiment(self, force: bool = False) -> float:
        """
        Enhanced sentiment fetching with rate limiting and multi-source aggregation
        """
        current_time = time.time()
        
        # Rate limiting
        if not force and (current_time - self.last_fetch_time) < self.min_fetch_interval:
            return self.last_sentiment_score
        
        self.last_fetch_time = current_time
        
        # Fetch from all sources
        source_scores = {}
        
        # 1. Funding Rate Sentiment
        source_scores['funding'] = self._fetch_funding_sentiment()
        
        # 2. Social Media Sentiment
        source_scores['social'] = self._fetch_social_sentiment()
        
        # 3. News Sentiment
        source_scores['news'] = self._fetch_news_sentiment()
        
        # 4. Options Flow Sentiment
        source_scores['options'] = self._fetch_options_sentiment()
        
        # Aggregate with dynamic weights
        self.last_sentiment_score = self._aggregate_sentiment(source_scores)
        
        # Store in history
        reading = SentimentReading(
            timestamp=current_time,
            score=self.last_sentiment_score,
            source="aggregated",
            confidence=self._calculate_confidence(source_scores),
            raw_data=source_scores
        )
        self.sentiment_history.append(reading)
        
        return self.last_sentiment_score
    
    def _fetch_funding_sentiment(self) -> float:
        """Analyze funding rates for sentiment using real API"""
        try:
            import requests
            # Example: Binance funding rate API (replace with your exchange)
            url = "https://fapi.binance.com/fapi/v1/premiumIndex"
            response = requests.get(url, timeout=5)
            data = response.json()
            # Filter for major coins
            coins = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
            funding_rates = {item['symbol']: float(item['lastFundingRate']) for item in data if item['symbol'] in coins}
            if not funding_rates:
                return 0.0
            avg_funding = statistics.mean(funding_rates.values())
            sentiment = -avg_funding * 10  # Negative funding = bullish
            return max(-1.0, min(1.0, sentiment))
        except Exception as e:
            self.logger.warning(f"Funding rate API error: {e}")
            return 0.0
    
    def _fetch_social_sentiment(self) -> float:
        """Analyze social media sentiment using Twitter API"""
        try:
            import requests
            # Example: Twitter API v2 (requires bearer token)
            bearer_token = self.config.get('twitter_bearer_token')
            if not bearer_token:
                self.logger.warning("Twitter API token not set in config")
                return 0.0
            headers = {"Authorization": f"Bearer {bearer_token}"}
            query = "(bitcoin OR ethereum OR crypto) lang:en -is:retweet"
            url = f"https://api.twitter.com/2/tweets/search/recent?query={query}&max_results=20"
            response = requests.get(url, headers=headers, timeout=5)
            data = response.json()
            tweets = [tweet['text'] for tweet in data.get('data', [])]
            return self._analyze_social_text(tweets)
        except Exception as e:
            self.logger.warning(f"Twitter API error: {e}")
            return 0.0
    
    def _fetch_news_sentiment(self) -> float:
        """Analyze news headlines sentiment using Crypto News API"""
        try:
            import requests
            # Example: CryptoPanic API (free tier)
            api_key = self.config.get('cryptopanic_api_key')
            url = f"https://cryptopanic.com/api/v1/posts/?auth_token={api_key}&currencies=BTC,ETH,SOL"
            response = requests.get(url, timeout=5)
            data = response.json()
            headlines = [item['title'] for item in data.get('results', [])]
            return self._analyze_text_sentiment(headlines)
        except Exception as e:
            self.logger.warning(f"News API error: {e}")
            return 0.0
    
    def _fetch_options_sentiment(self) -> float:
        """Analyze options flow for sentiment using Deribit API"""
        try:
            import requests
            # Example: Deribit public options stats
            url = "https://www.deribit.com/api/v2/public/get_book_summary_by_currency?currency=BTC"
            response = requests.get(url, timeout=5)
            data = response.json()
            # Calculate put/call ratio from open interest
            summaries = data.get('result', [])
            put_oi = sum(item['open_interest'] for item in summaries if item['option_type'] == 'put')
            call_oi = sum(item['open_interest'] for item in summaries if item['option_type'] == 'call')
            if put_oi + call_oi == 0:
                return 0.0
            put_call_ratio = put_oi / (put_oi + call_oi)
            sentiment = (1.0 - put_call_ratio) * 0.5
            return max(-1.0, min(1.0, sentiment))
        except Exception as e:
            self.logger.warning(f"Options API error: {e}")
            return 0.0
    def _fetch_crypto_specific_sentiment(self) -> float:
        """Fetch sentiment from crypto-specific sources (e.g., Reddit, Glassnode, Santiment)"""
        try:
            # Example: Reddit API (pushshift)
            import requests
            url = "https://api.pushshift.io/reddit/search/comment/?q=bitcoin&size=20"
            response = requests.get(url, timeout=5)
            data = response.json()
            comments = [item['body'] for item in data.get('data', [])]
            return self._analyze_text_sentiment(comments)
        except Exception as e:
            self.logger.warning(f"Crypto-specific API error: {e}")
            return 0.0
    
    def _analyze_social_text(self, texts: List[str]) -> float:
        """Advanced social media text analysis with filtering"""
        if not texts:
            return 0.0
        
        # Filter out spam/low quality content
        filtered_texts = []
        for text in texts:
            if self._is_valid_social_content(text):
                filtered_texts.append(text)
        
        if not filtered_texts:
            return 0.0
        
        return self._analyze_text_sentiment(filtered_texts)
    
    def _analyze_text_sentiment(self, texts: List[str]) -> float:
        """Multi-method text sentiment analysis"""
        if not texts:
            return 0.0
        
        scores = []
        
        for text in texts:
            text_scores = []
            
            # TextBlob analysis
            if self.textblob_analyzer:
                try:
                    blob = self.textblob_analyzer(text)
                    text_scores.append(blob.sentiment[0] if isinstance(blob.sentiment, tuple) else getattr(blob.sentiment, 'polarity', 0.0))
                except:
                    pass
            
            # VADER analysis
            if self.vader_analyzer:
                try:
                    vader_score = self.vader_analyzer.polarity_scores(text)
                    text_scores.append(vader_score['compound'])
                except:
                    pass
            
            # Simple keyword-based fallback
            if not text_scores:
                text_scores.append(self._simple_keyword_sentiment(text))
            
            # Average the methods for this text
            if text_scores:
                scores.append(statistics.mean(text_scores))
        
        return statistics.mean(scores) if scores else 0.0
    
    def _simple_keyword_sentiment(self, text: str) -> float:
        """Simple keyword-based sentiment as fallback"""
        bullish_keywords = ['bull', 'moon', 'pump', 'rally', 'breakout', 'strong', 'buy']
        bearish_keywords = ['bear', 'dump', 'crash', 'sell', 'drop', 'weak', 'down']
        
        text_lower = text.lower()
        
        bullish_count = sum(1 for word in bullish_keywords if word in text_lower)
        bearish_count = sum(1 for word in bearish_keywords if word in text_lower)
        
        if bullish_count + bearish_count == 0:
            return 0.0
        
        return (bullish_count - bearish_count) / (bullish_count + bearish_count)
    
    def _is_valid_social_content(self, text: str) -> bool:
        """Filter out spam and low-quality content"""
        text_lower = text.lower()
        
        # Check for spam keywords
        for keyword in self.trust_thresholds['spam_keywords']:
            if keyword in text_lower:
                return False
        
        # Check minimum length
        if len(text.split()) < 3:
            return False
        
        # Check for excessive emojis or special characters
        emoji_count = len(re.findall(r'[^\w\s]', text))
        if emoji_count > len(text) * 0.3:
            return False
        
        return True
    
    def _aggregate_sentiment(self, source_scores: Dict[str, float]) -> float:
        """Aggregate sentiment from multiple sources with dynamic weights"""
        weighted_sum = 0.0
        total_weight = 0.0
        
        for source, score in source_scores.items():
            if source in self.source_weights and score is not None:
                weight = self.source_weights[source]
                weighted_sum += score * weight
                total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        return weighted_sum / total_weight
    
    def _calculate_confidence(self, source_scores: Dict[str, float]) -> float:
        """Calculate confidence based on source agreement"""
        valid_scores = [s for s in source_scores.values() if s is not None]
        
        if len(valid_scores) < 2:
            return 0.5
        
        # Calculate standard deviation - lower std = higher confidence
        std_dev = statistics.stdev(valid_scores)
        
        # Convert to confidence (0-1 scale)
        confidence = max(0.1, min(1.0, 1.0 - std_dev))
        
        return confidence
    
    def analyze(self) -> Dict:
        """Enhanced analysis with additional metrics"""
        sentiment_score = self.fetch_sentiment()
        
        # Determine label
        if sentiment_score > 0.2:
            label = "bullish"
        elif sentiment_score < -0.2:
            label = "bearish"
        else:
            label = "neutral"
        
        # Calculate additional metrics
        metrics = {
            "score": round(sentiment_score, 3),
            "label": label,
            "confidence": self._get_current_confidence(),
            "trend": self._calculate_trend(),
            "volatility": self._calculate_sentiment_volatility(),
            "divergence": self._detect_divergence(),
            "sources": self._get_source_summary()
        }
        
        return metrics
    
    def _get_current_confidence(self) -> float:
        """Get confidence of most recent reading"""
        if not self.sentiment_history:
            return 0.5
        return self.sentiment_history[-1].confidence
    
    def _calculate_trend(self) -> str:
        """Calculate sentiment trend over recent history"""
        if len(self.sentiment_history) < 5:
            return "insufficient_data"
        
        recent_scores = [r.score for r in list(self.sentiment_history)[-5:]]
        
        # Simple trend calculation
        if recent_scores[-1] > recent_scores[0] + 0.1:
            return "improving"
        elif recent_scores[-1] < recent_scores[0] - 0.1:
            return "deteriorating"
        else:
            return "stable"
    
    def _calculate_sentiment_volatility(self) -> float:
        """Calculate volatility of sentiment readings"""
        if len(self.sentiment_history) < 10:
            return 0.0
        
        scores = [r.score for r in self.sentiment_history]
        return statistics.stdev(scores)
    
    def _detect_divergence(self) -> bool:
        """Detect if sentiment is diverging from expected patterns"""
        # Mock implementation - would need market price data
        # This would compare sentiment trend vs price trend
        return False
    
    def _get_source_summary(self) -> Dict:
        """Get summary of source contributions"""
        if not self.sentiment_history:
            return {}
        
        latest = self.sentiment_history[-1]
        return latest.raw_data
    
    def should_trigger_scan(self, market_volatility: float = 0.0, fear_level: float = 0.0) -> bool:
        """Event-triggered sentiment scanning"""
        # Check volatility trigger
        if market_volatility > self.event_triggers['volatility_threshold']:
            return True
        
        # Check fear trigger
        if fear_level > self.event_triggers['fear_threshold']:
            return True
        
        # Check sentiment change trigger
        if len(self.sentiment_history) >= 2:
            recent_change = abs(self.sentiment_history[-1].score - self.sentiment_history[-2].score)
            if recent_change > self.event_triggers['sentiment_change_threshold']:
                return True
        
        return False
    
    def update_performance_feedback(self, strategy_result: Dict):
        """Update performance history for learning"""
        if not self.sentiment_history:
            return
        
        # Store sentiment + outcome for learning
        feedback = {
            'timestamp': time.time(),
            'sentiment_score': self.last_sentiment_score,
            'sentiment_confidence': self._get_current_confidence(),
            'strategy_result': strategy_result,
            'market_structure': strategy_result.get('market_structure', 'unknown')
        }
        
        self.performance_history.append(feedback)
    
    def get_sentiment_impact_score(self) -> float:
        """Calculate how much sentiment should impact trading decisions"""
        if not self.performance_history:
            return 0.5  # Default moderate impact
        
        # Analyze historical performance when sentiment was factor
        # This would be more sophisticated in practice
        recent_performance = list(self.performance_history)[-20:]
        
        if not recent_performance:
            return 0.5
        
        # Simple correlation analysis
        sentiment_wins = sum(1 for p in recent_performance 
                           if p['strategy_result'].get('success', False))
        
        impact_score = sentiment_wins / len(recent_performance)
        return max(0.1, min(0.9, impact_score))
    
    def adjust_source_weights(self, performance_data: Dict):
        """Dynamically adjust source weights based on performance"""
        # This would analyze which sources led to better predictions
        # For now, simple mock adjustment
        
        market_regime = performance_data.get('market_regime', 'normal')
        
        if market_regime == 'high_volatility':
            # In high volatility, funding rates might be more reliable
            self.source_weights['funding'] = 0.4
            self.source_weights['social'] = 0.3
        elif market_regime == 'low_volatility':
            # In low volatility, social sentiment might be more predictive
            self.source_weights['social'] = 0.5
            self.source_weights['funding'] = 0.2
    
    def get_dashboard_data(self) -> Dict:
        """Get data for real-time sentiment dashboard"""
        return {
            'current_sentiment': self.last_sentiment_score,
            'sentiment_history': [
                {
                    'timestamp': r.timestamp,
                    'score': r.score,
                    'confidence': r.confidence
                }
                for r in self.sentiment_history
            ],
            'source_weights': self.source_weights,
            'recent_analysis': self.analyze(),
            'performance_metrics': self._get_performance_metrics()
        }
    
    def _get_performance_metrics(self) -> Dict:
        """Get performance metrics for monitoring"""
        if not self.performance_history:
            return {}
        
        recent = list(self.performance_history)[-50:]
        
        return {
            'total_readings': len(self.sentiment_history),
            'avg_confidence': statistics.mean(r.confidence for r in self.sentiment_history),
            'sentiment_range': {
                'min': min(r.score for r in self.sentiment_history),
                'max': max(r.score for r in self.sentiment_history)
            },
            'recent_accuracy': len([p for p in recent if p['strategy_result'].get('success')]) / len(recent) if recent else 0
        }
# Example usage and integration
class SentimentIntegrator:
    """Integration helper for connecting sentiment to other agents"""
    
    def __init__(self, sentiment_agent: SentimentScanAgent):
        self.sentiment_agent = sentiment_agent
    
    def boost_fear_analysis(self, base_fear: float) -> float:
        """Boost fear analysis with sentiment data"""
        sentiment_analysis = self.sentiment_agent.analyze()
        
        if sentiment_analysis['label'] == 'bearish':
            # Negative sentiment amplifies fear
            fear_multiplier = 1.0 + abs(sentiment_analysis['score']) * 0.5
            return min(1.0, base_fear * fear_multiplier)
        
        return base_fear
    
    def weight_strategy_decisions(self, base_weight: float, strategy_name: str) -> float:
        """Adjust strategy weights based on sentiment"""
        sentiment_impact = self.sentiment_agent.get_sentiment_impact_score()
        sentiment_analysis = self.sentiment_agent.analyze()
        
        # Example: reduce aggressive strategies in uncertain sentiment
        if sentiment_analysis['confidence'] < 0.3:
            if 'aggressive' in strategy_name.lower():
                return base_weight * 0.5
        
        return base_weight
    
    def suppress_trades_check(self) -> bool:
        """Check if trades should be suppressed due to sentiment"""
        sentiment_analysis = self.sentiment_agent.analyze()
        
        # Suppress trades when sentiment is flat and confidence is low
        if (sentiment_analysis['label'] == 'neutral' and 
            sentiment_analysis['confidence'] < 0.4):
            return True
        
        return False


# Example usage
if __name__ == "__main__":
    # Initialize sentiment agent
    config = {
        'history_size': 50,
        'min_fetch_interval': 15,
        'enable_social_scraping': True,
        'enable_news_scraping': True,
        'sentiment_smoothing': 0.1,
        'divergence_threshold': 0.3,
        # Add your Twitter API token here
        'twitter_bearer_token': 'YOUR_TWITTER_BEARER_TOKEN',
        'cryptopanic_api_key': 'YOUR_CRYPTO_PANIC_API_KEY',
    }

    
    agent = SentimentScanAgent(config)
    
    # Basic analysis
    result = agent.analyze()
    print(f"Sentiment Analysis: {result}")
    
    # Event-triggered scanning
    if agent.should_trigger_scan(market_volatility=0.035, fear_level=0.5):
        print("Triggered sentiment scan due to market conditions")
        result = agent.analyze()
    
    # Dashboard data
    dashboard = agent.get_dashboard_data()
    print(f"Dashboard data available with {len(dashboard['sentiment_history'])} readings")