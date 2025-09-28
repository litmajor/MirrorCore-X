import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import warnings
from typing import Dict, List, Tuple, Optional
import json
from dataclasses import dataclass, field
from scipy import stats
import yfinance
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

warnings.filterwarnings('ignore')

@dataclass
class Config:
    """Configuration class for the enhanced scanner"""
    PREV_CSV: str = "scan_results_daily_20250821_172252.csv"
    CURR_CSV: str = "scan_results_daily_20250827_224149.csv"
    OUTPUT_MD: str = "enhanced_weekly_comparison.md"
    OUTPUT_JSON: str = "analysis_results.json"
    OUTPUT_HTML: str = "interactive_report.html"
    
    # Thresholds
    RSI_OVERBOUGHT: float = 75.0
    RSI_OVERSOLD: float = 25.0
    MOMENTUM_THRESHOLD: float = 0.05
    VOLUME_SPIKE_THRESHOLD: float = 2.0
    
    # Risk parameters
    VAR_CONFIDENCE: float = 0.05
    MAX_POSITION_SIZE: float = 0.1
    
    # Email settings (optional)
    EMAIL_ENABLED: bool = False
    SMTP_SERVER: str = "smtp.gmail.com"
    SMTP_PORT: int = 587
    EMAIL_USER: str = ""
    EMAIL_PASS: str = ""
    EMAIL_TO: List[str] = field(default_factory=list)

class EnhancedCryptoAnalyzer:
    """Enhanced crypto scanner comparison with advanced features"""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.prev_data: Optional[pd.DataFrame] = None
        self.curr_data: Optional[pd.DataFrame] = None
        self.analysis_results = {}
        
    def load_data(self) -> None:
        """Load and validate data from CSV files"""
        try:
            self.prev_data = pd.read_csv(self.config.PREV_CSV)
            self.curr_data = pd.read_csv(self.config.CURR_CSV)
            
            # Ensure required columns exist
            required_cols = ['symbol', 'composite_score', 'rsi']
            for col in required_cols:
                if col not in self.prev_data.columns or col not in self.curr_data.columns:
                    raise ValueError(f"Required column '{col}' missing from data")
                    
            print(f"✅ Data loaded: {len(self.prev_data)} previous, {len(self.curr_data)} current cryptos")
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            raise
    
    # === CORE ANALYSIS FUNCTIONS ===
    
    def basic_stats(self, series: pd.Series) -> Dict:
        """Enhanced statistical summary"""
        return {
            'mean': round(series.mean(), 3),
            'median': round(series.median(), 3),
            'std': round(series.std(), 3),
            'min': round(series.min(), 3),
            'max': round(series.max(), 3),
            'q25': round(series.quantile(0.25), 3),
            'q75': round(series.quantile(0.75), 3),
            'skewness': round(stats.skew(series), 3),
            'kurtosis': round(stats.kurtosis(series), 3)
        }
    
    def calculate_var(self, returns: pd.Series, confidence: float = 0.05) -> float:
        """Calculate Value at Risk"""
        val = np.percentile(returns, confidence * 100)
        return float(val) if isinstance(val, (float, int, np.floating)) else float(val.item())
    
    def max_drawdown(self, series: pd.Series) -> float:
        """Calculate maximum drawdown"""
        roll_max = series.cummax()
        drawdown = (series - roll_max) / roll_max
        return round(drawdown.min() * 100, 2)
    
    # === ADVANCED TECHNICAL ANALYSIS ===
    
    def calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict:
        """Calculate MACD indicators"""
        exp1 = prices.ewm(span=fast).mean()
        exp2 = prices.ewm(span=slow).mean()
        macd = exp1 - exp2
        macd_signal = macd.ewm(span=signal).mean()
        macd_histogram = macd - macd_signal
        
        return {
            'macd': macd.iloc[-1] if len(macd) > 0 else 0,
            'signal': macd_signal.iloc[-1] if len(macd_signal) > 0 else 0,
            'histogram': macd_histogram.iloc[-1] if len(macd_histogram) > 0 else 0
        }
    
    def williams_r(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float:
        """Calculate Williams %R"""
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        wr = -100 * (highest_high - close) / (highest_high - lowest_low)
        return wr.iloc[-1] if len(wr) > 0 else 0
    
    def stochastic_oscillator(self, high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14) -> Dict:
        """Calculate Stochastic Oscillator"""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k_percent = 100 * (close - lowest_low) / (highest_high - lowest_low)
        d_percent = k_percent.rolling(window=3).mean()
        
        return {
            'k_percent': k_percent.iloc[-1] if len(k_percent) > 0 else 0,
            'd_percent': d_percent.iloc[-1] if len(d_percent) > 0 else 0
        }
    
    # === SECTOR ANALYSIS ===
    
    def sector_analysis(self) -> Dict:
        """Analyze sector performance and rotation"""
        if self.curr_data is None or not hasattr(self.curr_data, 'columns') or 'sector' not in self.curr_data.columns:
            return {'error': 'Sector data not available'}
        sector_analysis = {}
        for sector in self.curr_data['sector'].unique():
            if pd.isna(sector):
                continue
            curr_sector = self.curr_data[self.curr_data['sector'] == sector] if self.curr_data is not None else pd.DataFrame()
            prev_sector = pd.DataFrame()
            if self.prev_data is not None and hasattr(self.prev_data, 'columns') and 'sector' in self.prev_data.columns:
                prev_sector = self.prev_data[self.prev_data['sector'] == sector]
            sector_stats = {
                'current_count': len(curr_sector),
                'avg_score': round(curr_sector['composite_score'].mean(), 3) if 'composite_score' in curr_sector.columns else 0,
                'avg_rsi': round(curr_sector['rsi'].mean(), 3) if 'rsi' in curr_sector.columns else 0,
                'top_performer': curr_sector.loc[curr_sector['composite_score'].idxmax(), 'symbol'] if len(curr_sector) > 0 and 'composite_score' in curr_sector.columns and 'symbol' in curr_sector.columns else None
            }
            if len(prev_sector) > 0 and 'composite_score' in prev_sector.columns and 'composite_score' in curr_sector.columns:
                prev_avg = prev_sector['composite_score'].mean()
                curr_avg = curr_sector['composite_score'].mean()
                sector_stats['score_change'] = round(((curr_avg - prev_avg) / prev_avg) * 100, 2) if prev_avg != 0 else 0
                sector_stats['momentum'] = 'Strong' if sector_stats['score_change'] > 5 else 'Weak' if sector_stats['score_change'] < -5 else 'Neutral'
            sector_analysis[sector] = sector_stats
        return sector_analysis
    
    # === MARKET BREADTH ===
    
    def market_breadth(self) -> Dict:
        """Calculate market breadth indicators"""
        if self.curr_data is None or not hasattr(self.curr_data, 'columns'):
            return {'error': 'No current data'}
        curr_advancing = (self.curr_data['price_change'] > 0).sum() if 'price_change' in self.curr_data.columns else 0
        curr_declining = (self.curr_data['price_change'] < 0).sum() if 'price_change' in self.curr_data.columns else 0
        breadth = {
            'advance_decline_ratio': round(curr_advancing / max(curr_declining, 1), 3),
            'advancing_cryptos': curr_advancing,
            'declining_cryptos': curr_declining,
            'new_highs': (self.curr_data['rsi'] > 70).sum() if 'rsi' in self.curr_data.columns else 0,
            'new_lows': (self.curr_data['rsi'] < 30).sum() if 'rsi' in self.curr_data.columns else 0,
            'cryptos_above_ma': (self.curr_data['price'] > self.curr_data['ma_20']).sum() if 'ma_20' in self.curr_data.columns and 'price' in self.curr_data.columns else 0
        }
        return breadth
    
    # === MACHINE LEARNING FEATURES ===
    
    def clustering_analysis(self, n_clusters: int = 5) -> Dict:
        """Perform clustering analysis on current cryptos"""
        if self.curr_data is None or not hasattr(self.curr_data, 'columns'):
            return {'error': 'No current data'}
        feature_cols = ['composite_score', 'rsi', 'momentum_short', 'volume_ratio']
        available_cols = [col for col in feature_cols if col in self.curr_data.columns]
        if len(available_cols) < 2:
            return {'error': 'Insufficient features for clustering'}
        X = self.curr_data[available_cols].fillna(0)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        cluster_analysis = {}
        for i in range(n_clusters):
            cluster_cryptos = self.curr_data[clusters == i]
            cluster_analysis[f'cluster_{i}'] = {
                'count': len(cluster_cryptos),
                'avg_score': round(cluster_cryptos['composite_score'].mean(), 3) if 'composite_score' in cluster_cryptos.columns else 0,
                'characteristics': self.basic_stats(cluster_cryptos['composite_score']) if 'composite_score' in cluster_cryptos.columns else {},
                'top_cryptos': cluster_cryptos.nlargest(3, 'composite_score')['symbol'].tolist() if 'composite_score' in cluster_cryptos.columns and 'symbol' in cluster_cryptos.columns else []
            }
        return cluster_analysis
    
    # === RISK ANALYSIS ===
    
    def risk_analysis(self) -> Dict:
        """Comprehensive risk analysis"""
        risk_metrics = {}
        if self.curr_data is None or self.prev_data is None:
            return {'error': 'No data for risk analysis'}
        # Basic risk metrics
        risk_metrics['volatility'] = {
            'current': round(self.curr_data['composite_score'].std(), 3) if 'composite_score' in self.curr_data.columns else 0,
            'previous': round(self.prev_data['composite_score'].std(), 3) if 'composite_score' in self.prev_data.columns else 0
        }
        # Drawdown analysis
        risk_metrics['drawdown'] = {
            'current': self.max_drawdown(self.curr_data['composite_score']) if 'composite_score' in self.curr_data.columns else 0,
            'previous': self.max_drawdown(self.prev_data['composite_score']) if 'composite_score' in self.prev_data.columns else 0
        }
        # Risk warnings
        risk_metrics['warnings'] = {
            'overbought_cryptos': (self.curr_data['rsi'] > self.config.RSI_OVERBOUGHT).sum() if 'rsi' in self.curr_data.columns else 0,
            'oversold_cryptos': (self.curr_data['rsi'] < self.config.RSI_OVERSOLD).sum() if 'rsi' in self.curr_data.columns else 0,
            'high_momentum': (self.curr_data['momentum_short'] > 0.1).sum() if 'momentum_short' in self.curr_data.columns else 0,
            'volume_spikes': (self.curr_data['volume_ratio'] > self.config.VOLUME_SPIKE_THRESHOLD).sum() if 'volume_ratio' in self.curr_data.columns else 0
        }
        # Position sizing recommendations
        if 'volatility' in self.curr_data.columns:
            risk_metrics['position_sizing'] = {}
            for _, crypto in self.curr_data.head(10).iterrows():
                vol = crypto.get('volatility', 0.2)
                recommended_size = min(self.config.MAX_POSITION_SIZE, 0.02 / max(vol, 0.01))
                risk_metrics['position_sizing'][crypto['symbol']] = round(recommended_size, 4)
        return risk_metrics
    
    # === PERFORMANCE ATTRIBUTION ===
    
    def performance_attribution(self) -> Dict:
        """Analyze performance attribution"""
        if self.prev_data is None or self.curr_data is None:
            return {'error': 'No data for attribution analysis'}
        if not all(col in self.prev_data.columns for col in ['symbol', 'composite_score', 'rsi']) or not all(col in self.curr_data.columns for col in ['symbol', 'composite_score', 'rsi']):
            return {'error': 'Missing columns for attribution analysis'}
        merged = pd.merge(
            self.prev_data[['symbol', 'composite_score', 'rsi']],
            self.curr_data[['symbol', 'composite_score', 'rsi']],
            on='symbol', suffixes=('_prev', '_curr')
        )
        if len(merged) == 0:
            return {'error': 'No common symbols for attribution analysis'}
        merged['score_change'] = merged['composite_score_curr'] - merged['composite_score_prev']
        merged['rsi_change'] = merged['rsi_curr'] - merged['rsi_prev']
        attribution = {
            'best_performers': merged.nlargest(5, 'score_change')[['symbol', 'score_change']].to_dict('records'),
            'worst_performers': merged.nsmallest(5, 'score_change')[['symbol', 'score_change']].to_dict('records'),
            'consistency_score': round(merged['score_change'].std(), 3),
            'mean_reversion_candidates': merged[
                (merged['rsi_prev'] > 70) & (merged['rsi_curr'] < 60)
            ]['symbol'].tolist()
        }
        return attribution
    
    # === ALERT SYSTEM ===
    
    def generate_alerts(self) -> List[Dict]:
        """Generate trading alerts based on conditions"""
        alerts = []
        if self.curr_data is None:
            return alerts
        for _, crypto in self.curr_data.iterrows():
            symbol = crypto['symbol']
            # RSI alerts
            if 'rsi' in crypto and crypto['rsi'] > self.config.RSI_OVERBOUGHT:
                alerts.append({
                    'type': 'OVERBOUGHT',
                    'symbol': symbol,
                    'message': f"RSI at {crypto['rsi']:.1f} - Consider taking profits",
                    'priority': 'HIGH' if crypto['rsi'] > 80 else 'MEDIUM'
                })
            elif 'rsi' in crypto and crypto['rsi'] < self.config.RSI_OVERSOLD:
                alerts.append({
                    'type': 'OVERSOLD',
                    'symbol': symbol,
                    'message': f"RSI at {crypto['rsi']:.1f} - Potential buying opportunity",
                    'priority': 'HIGH' if crypto['rsi'] < 20 else 'MEDIUM'
                })
            # Volume spike alerts
            if 'volume_ratio' in crypto and crypto['volume_ratio'] > self.config.VOLUME_SPIKE_THRESHOLD:
                alerts.append({
                    'type': 'VOLUME_SPIKE',
                    'symbol': symbol,
                    'message': f"Volume spike: {crypto['volume_ratio']:.1f}x average",
                    'priority': 'MEDIUM'
                })
            # Momentum alerts
            if 'momentum_short' in crypto and crypto['momentum_short'] > 0.15:
                alerts.append({
                    'type': 'STRONG_MOMENTUM',
                    'symbol': symbol,
                    'message': f"Strong momentum: {crypto['momentum_short']*100:.1f}%",
                    'priority': 'MEDIUM'
                })
        return sorted(alerts, key=lambda x: {'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}[x['priority']], reverse=True)
    
    # === VISUALIZATION ===
    
    def create_interactive_charts(self) -> str:
        """Create interactive charts using Plotly"""
        if self.curr_data is None:
            return "<p>No data available for charts.</p>"
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Composite Score Distribution', 'RSI vs Momentum', 
                          'Volume Analysis', 'Sector Performance'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        # Composite Score Distribution
        if 'composite_score' in self.curr_data.columns:
            fig.add_trace(
                go.Histogram(x=self.curr_data['composite_score'], name='Current', opacity=0.7),
                row=1, col=1
            )
        # RSI vs Momentum scatter
        if 'momentum_short' in self.curr_data.columns and 'rsi' in self.curr_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=self.curr_data['rsi'],
                    y=self.curr_data['momentum_short']*100,
                    mode='markers',
                    text=self.curr_data['symbol'] if 'symbol' in self.curr_data.columns else None,
                    name='Cryptos'
                ),
                row=1, col=2
            )
        # Volume Analysis
        if 'volume_ratio' in self.curr_data.columns:
            fig.add_trace(
                go.Box(y=self.curr_data['volume_ratio'], name='Volume Ratio'),
                row=2, col=1
            )
        # Update layout
        fig.update_layout(
            title_text="Crypto Scanner Analysis Dashboard",
            showlegend=True,
            height=800
        )
        return fig.to_html(include_plotlyjs='cdn')
    
    # === MAIN ANALYSIS ===
    
    def run_comprehensive_analysis(self) -> Dict:
        """Run all analysis components"""
        print("🔄 Running comprehensive analysis...")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'basic_comparison': self.basic_comparison(),
            'sector_analysis': self.sector_analysis(),
            'market_breadth': self.market_breadth(),
            'clustering_analysis': self.clustering_analysis(),
            'risk_analysis': self.risk_analysis(),
            'performance_attribution': self.performance_attribution(),
            'alerts': self.generate_alerts(),
            'recommendations': self.generate_recommendations()
        }
        
        self.analysis_results = results
        return results
    
    def basic_comparison(self) -> Dict:
        """Basic comparison metrics (original functionality)"""
        if self.prev_data is None or self.curr_data is None:
            return {'error': 'No data for basic comparison'}
        if not all(col in self.prev_data.columns for col in ['symbol', 'composite_score', 'rsi']) or not all(col in self.curr_data.columns for col in ['symbol', 'composite_score', 'rsi']):
            return {'error': 'Missing columns for basic comparison'}
        prev_top30 = set(self.prev_data.nlargest(30, 'composite_score')['symbol'])
        curr_top30 = set(self.curr_data.nlargest(30, 'composite_score')['symbol'])
        return {
            'avg_rsi': {
                'previous': round(self.prev_data['rsi'].mean(), 2),
                'current': round(self.curr_data['rsi'].mean(), 2)
            },
            'top_score': {
                'previous': round(self.prev_data['composite_score'].max(), 2),
                'current': round(self.curr_data['composite_score'].max(), 2)
            },
            'new_leaders': list(curr_top30 - prev_top30),
            'fallen_leaders': list(prev_top30 - curr_top30),
            'consistent_performers': list(prev_top30 & curr_top30),
            'turnover_rate': round(len(curr_top30 - prev_top30) / 30 * 100, 1)
        }
    
    def generate_recommendations(self) -> Dict:
        """Generate trading recommendations"""
        recommendations = {
            'buy_candidates': [],
            'sell_candidates': [],
            'watch_list': [],
            'risk_warnings': []
        }
        if self.curr_data is None:
            return recommendations
        if not all(col in self.curr_data.columns for col in ['symbol', 'composite_score', 'rsi']):
            return recommendations
        # Buy candidates: High score, not overbought, good momentum
        buy_criteria = (
            (self.curr_data['composite_score'] > self.curr_data['composite_score'].quantile(0.8)) &
            (self.curr_data['rsi'] < 70) &
            (self.curr_data['rsi'] > 40)
        )
        if 'momentum_short' in self.curr_data.columns:
            buy_criteria = buy_criteria & (self.curr_data['momentum_short'] > 0.02)
        recommendations['buy_candidates'] = self.curr_data[buy_criteria].nlargest(5, 'composite_score')[
            ['symbol', 'composite_score', 'rsi']
        ].to_dict('records')
        # Sell candidates: Overbought with declining momentum
        sell_criteria = (self.curr_data['rsi'] > 75)
        recommendations['sell_candidates'] = self.curr_data[sell_criteria][
            ['symbol', 'composite_score', 'rsi']
        ].to_dict('records')
        # Watch list: High potential but need confirmation
        watch_criteria = (
            (self.curr_data['composite_score'] > self.curr_data['composite_score'].median()) &
            (self.curr_data['rsi'] > 70) & (self.curr_data['rsi'] < 80)
        )
        recommendations['watch_list'] = self.curr_data[watch_criteria][
            ['symbol', 'composite_score', 'rsi']
        ].head(5).to_dict('records')
        return recommendations
    
    # === OUTPUT GENERATION ===
    
    def generate_enhanced_report(self) -> str:
        """Generate enhanced markdown report"""
        results = self.analysis_results
        
        # Create enhanced markdown report
        md_content = f"""
# 🚀 Enhanced Weekly Scanner Analysis
*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*

---

## 📊 Executive Summary

### Key Metrics Comparison
| Metric | Previous | Current | Change |
|--------|----------|---------|---------|
| Avg RSI | {results['basic_comparison']['avg_rsi']['previous']} | {results['basic_comparison']['avg_rsi']['current']} | {'⬆️' if results['basic_comparison']['avg_rsi']['current'] > results['basic_comparison']['avg_rsi']['previous'] else '⬇️'} |
| Top Score | {results['basic_comparison']['top_score']['previous']} | {results['basic_comparison']['top_score']['current']} | {'⬆️' if results['basic_comparison']['top_score']['current'] > results['basic_comparison']['top_score']['previous'] else '⬇️'} |
| Turnover Rate | - | {results['basic_comparison']['turnover_rate']}% | 🔄 |

### Market Breadth
- **Advance/Decline Ratio**: {results['market_breadth']['advance_decline_ratio']}
- **New Highs**: {results['market_breadth']['new_highs']} | **New Lows**: {results['market_breadth']['new_lows']}
- **Cryptos Above MA**: {results['market_breadth']['cryptos_above_ma']}

---

## 🎯 Trading Recommendations

### 🟢 BUY Candidates
"""
        
        for stock in results['recommendations']['buy_candidates']:
            md_content += f"- **{stock['symbol']}** (Score: {stock['composite_score']}, RSI: {stock['rsi']})\n"
        
        md_content += "\n### 🔴 SELL Candidates\n"
        for stock in results['recommendations']['sell_candidates']:
            md_content += f"- **{stock['symbol']}** (Score: {stock['composite_score']}, RSI: {stock['rsi']})\n"
        
        md_content += f"""

---

## 🏭 Sector Analysis
"""
        
        if 'error' not in results['sector_analysis']:
            for sector, data in results['sector_analysis'].items():
                momentum = data.get('momentum', 'Unknown')
                emoji = '🚀' if momentum == 'Strong' else '📉' if momentum == 'Weak' else '➡️'
                md_content += f"### {emoji} {sector}\n"
                md_content += f"- **Stocks**: {data['current_count']} | **Avg Score**: {data['avg_score']} | **Momentum**: {momentum}\n"
                md_content += f"- **Top Performer**: {data['top_performer']}\n\n"
        
        md_content += f"""

---

## ⚠️ Risk Analysis

### Current Risk Metrics
- **Volatility**: {results['risk_analysis']['volatility']['current']}
- **Max Drawdown**: {results['risk_analysis']['drawdown']['current']}%
- **Overbought Cryptos**: {results['risk_analysis']['warnings']['overbought_cryptos']}
- **Volume Spikes**: {results['risk_analysis']['warnings']['volume_spikes']}

---

## 🔔 Active Alerts ({len(results['alerts'])})
"""
        
        for alert in results['alerts'][:10]:  # Show top 10 alerts
            priority_emoji = '🚨' if alert['priority'] == 'HIGH' else '⚡' if alert['priority'] == 'MEDIUM' else '💡'
            md_content += f"- {priority_emoji} **{alert['symbol']}**: {alert['message']}\n"
        
        md_content += f"""

---

## 📈 Performance Attribution

### Best Performers
"""
        for performer in results['performance_attribution']['best_performers']:
            md_content += f"- **{performer['symbol']}**: +{performer['score_change']:.2f}\n"
        
        md_content += "\n### Worst Performers\n"
        for performer in results['performance_attribution']['worst_performers']:
            md_content += f"- **{performer['symbol']}**: {performer['score_change']:.2f}\n"
        
        md_content += f"""

---

## 🤖 Machine Learning Insights

### Crypto Clustering Analysis
"""
        
        if 'error' not in results['clustering_analysis']:
            for cluster, data in results['clustering_analysis'].items():
                md_content += f"#### {cluster.replace('_', ' ').title()}\n"
                md_content += f"- **Size**: {data['count']} cryptos | **Avg Score**: {data['avg_score']}\n"
                md_content += f"- **Top Cryptos**: {', '.join(data['top_cryptos'])}\n\n"
        
        md_content += f"""

---

*Report generated by Enhanced Crypto  Scanner v2.0*
*Next update: {(datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')}*
"""
        
        return md_content
    
    def save_results(self) -> None:
        """Save all analysis results"""
        # Save markdown report
        markdown_report = self.generate_enhanced_report()
        Path(self.config.OUTPUT_MD).write_text(markdown_report, encoding='utf-8')
        
        # Save JSON results
        with open(self.config.OUTPUT_JSON, 'w') as f:
            json.dump(self.analysis_results, f, indent=2, default=str)
        
        # Save HTML interactive report
        html_charts = self.create_interactive_charts()
        Path(self.config.OUTPUT_HTML).write_text(html_charts, encoding='utf-8')
        
        print(f"✅ Reports saved:")
        print(f"   📄 Markdown: {self.config.OUTPUT_MD}")
        print(f"   📊 JSON: {self.config.OUTPUT_JSON}")
        print(f"   🌐 HTML: {self.config.OUTPUT_HTML}")
    
    def send_email_alert(self, alerts: List[Dict]) -> None:
        """Send email alerts (optional feature)"""
        if not self.config.EMAIL_ENABLED or not alerts:
            return
        
        try:
            msg = MIMEMultipart()
            msg['From'] = self.config.EMAIL_USER
            msg['To'] = ', '.join(self.config.EMAIL_TO)
            msg['Subject'] = f"Crypto Scanner Alerts - {datetime.now().strftime('%Y-%m-%d')}"
            
            # Create email body
            body = "High Priority Trading Alerts:\n\n"
            for alert in alerts:
                if alert['priority'] == 'HIGH':
                    body += f"• {alert['symbol']}: {alert['message']}\n"
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            server = smtplib.SMTP(self.config.SMTP_SERVER, self.config.SMTP_PORT)
            server.starttls()
            server.login(self.config.EMAIL_USER, self.config.EMAIL_PASS)
            server.send_message(msg)
            server.quit()
            
            print("📧 Email alerts sent successfully")
            
        except Exception as e:
            print(f"❌ Failed to send email: {e}")

def main():
    """Main execution function"""
    print("🚀 Starting Enhanced Crypto Scanner Analysis...")
    
    # Initialize analyzer
    config = Config()
    analyzer = EnhancedCryptoAnalyzer(config)
    
    try:
        # Load data
        analyzer.load_data()
        
        # Run comprehensive analysis
        results = analyzer.run_comprehensive_analysis()
        
        # Save all results
        analyzer.save_results()
        
        # Send email alerts for high priority items
        high_priority_alerts = [alert for alert in results['alerts'] if alert['priority'] == 'HIGH']
        if high_priority_alerts:
            analyzer.send_email_alert(high_priority_alerts)
        
        print("\n🎉 Analysis completed successfully!")
        print(f"📈 Found {len(results['alerts'])} alerts")
        print(f"🎯 {len(results['recommendations']['buy_candidates'])} buy candidates identified")
        
    except Exception as e:
        print(f"💥 Analysis failed: {e}")
        raise

if __name__ == "__main__":
    main()
