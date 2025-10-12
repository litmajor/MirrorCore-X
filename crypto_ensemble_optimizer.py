
"""
MirrorCore-X: Crypto-Specific Ensemble Optimizer
Optimizations for 24/7 markets, extreme volatility, and on-chain intelligence

Key Adaptations:
1. 24/7 continuous trading (no market close)
2. Extreme volatility regime handling
3. On-chain metrics integration
4. Exchange-specific dynamics
5. Flash crash protection
6. Funding rate arbitrage
7. Cross-exchange arbitrage
8. MEV awareness

Author: MirrorCore-X Team
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings
import logging

logger = logging.getLogger(__name__)


@dataclass
class CryptoOptimizationConfig:
    """Configuration specific to crypto markets"""
    # Basic metadata
    optimization_metric: str = "SHARPE_RATIO"
    risk_profile: str = "moderate"
    # Backwards-compatible optional fields
    objective: Optional[str] = None
    max_position_size: float = 0.1
    turnover_penalty: float = 0.0
    
    # Volatility handling
    extreme_vol_threshold: float = 0.10  # 10% hourly volatility = extreme
    vol_scaling_factor: float = 2.0  # Scale lambda by this during extreme vol
    flash_crash_threshold: float = 0.15  # 15% drop in 5min = flash crash
    
    # 24/7 market adjustments
    continuous_rebalance: bool = True
    rebalance_frequency_minutes: int = 60  # Hourly default
    night_trading_multiplier: float = 0.7  # Reduce position size at night (low liquidity)
    
    # Crypto-specific risks
    funding_rate_threshold: float = 0.01  # 1% funding = consider arb
    cross_exchange_spread_threshold: float = 0.005  # 0.5% = arb opportunity
    liquidation_buffer: float = 0.30  # Stay 30% away from liquidation
    
    # On-chain metrics
    use_on_chain_data: bool = True
    whale_wallet_threshold: float = 1000  # BTC/ETH amounts that matter
    exchange_flow_weight: float = 0.15  # Weight for exchange flow signals
    
    # Gas/transaction costs
    gas_cost_btc: float = 0.0001  # ~$5 in BTC
    slippage_bps: float = 5.0  # 5 bps average slippage
    min_trade_size_usd: float = 100  # Minimum to overcome costs
    
    # Risk management
    max_drawdown_stop: float = 0.20  # Stop trading at 20% drawdown
    circuit_breaker_vol: float = 0.25  # Halt at 25% hourly vol
    max_leverage: float = 3.0  # Maximum leverage allowed
    
    # Market microstructure
    orderbook_depth_levels: int = 10  # Levels to analyze
    min_liquidity_ratio: float = 0.1  # Min ratio of size to daily volume

    def __post_init__(self):
        # Allow 'objective' to be passed as a legacy alias for optimization_metric
        if self.objective:
            # Support both enum-like and string inputs (e.g., OptimizationObjective.SHARPE_RATIO)
            try:
                # If objective is an enum value with 'name', use that
                self.optimization_metric = getattr(self.objective, 'name', str(self.objective))
            except Exception:
                self.optimization_metric = str(self.objective)


class CryptoRegimeDetector:
    """
    Detects crypto-specific market regimes beyond traditional ones.
    """
    
    def __init__(self):
        self.regime_history = []
    
    def detect_crypto_regime(
        self, 
        df: pd.DataFrame,
        on_chain_data: Optional[Dict] = None
    ) -> str:
        """
        Detect crypto-specific market regimes.
        
        Returns:
            - BULL_RUN: Strong uptrend with increasing volume
            - BEAR_CAPITULATION: Panic selling, high vol
            - ACCUMULATION: Low vol, sideways, whale buying
            - DISTRIBUTION: High vol, sideways, whale selling
            - FLASH_CRASH: Extreme sudden drop
            - PUMP_AND_DUMP: Coordinated manipulation
            - NORMAL_VOLATILE: Standard crypto volatility
            - WEEKEND_LULL: Low weekend liquidity
        """
        
        # Calculate metrics
        returns = df['close'].pct_change()
        volatility = returns.rolling(24).std()  # 24-hour vol
        volume_ma = df['volume'].rolling(168).mean()  # 7-day MA
        volume_ratio = df['volume'] / volume_ma
        
        recent_return = returns.iloc[-24:].sum()  # Last 24h
        recent_vol = volatility.iloc[-1]
        recent_volume = volume_ratio.iloc[-1]
        
        # Price momentum
        sma_20 = df['close'].rolling(20).mean()
        sma_50 = df['close'].rolling(50).mean()
        price_trend = (df['close'].iloc[-1] - sma_20.iloc[-1]) / sma_20.iloc[-1]
        
        # Flash crash detection
        max_drop_5min = returns.rolling(5).min().iloc[-1]
        if max_drop_5min < -0.15:
            return 'FLASH_CRASH'
        
        # Weekend detection (crypto-specific)
        current_time = datetime.now()
        is_weekend = current_time.weekday() >= 5
        
        if is_weekend and recent_volume < 0.7:
            return 'WEEKEND_LULL'
        
        # Bull run characteristics
        if (recent_return > 0.15 and 
            recent_volume > 1.5 and 
            sma_20.iloc[-1] > sma_50.iloc[-1]):
            return 'BULL_RUN'
        
        # Bear capitulation
        if (recent_return < -0.20 and 
            recent_vol > 0.08 and 
            recent_volume > 2.0):
            return 'BEAR_CAPITULATION'
        
        # Pump and dump pattern
        if (recent_return > 0.30 and 
            recent_vol > 0.10 and 
            volume_ratio.iloc[-1] > 5.0):
            return 'PUMP_AND_DUMP'
        
        # Accumulation (on-chain confirmation)
        if on_chain_data:
            exchange_netflow = on_chain_data.get('exchange_netflow', 0)
            if (abs(price_trend) < 0.05 and 
                recent_vol < 0.03 and 
                exchange_netflow < -100):  # Net outflow from exchanges
                return 'ACCUMULATION'
        
        # Distribution
        if on_chain_data:
            exchange_netflow = on_chain_data.get('exchange_netflow', 0)
            if (abs(price_trend) < 0.05 and 
                recent_vol > 0.04 and 
                exchange_netflow > 100):  # Net inflow to exchanges
                return 'DISTRIBUTION'
        
        # Default: normal volatile
        return 'NORMAL_VOLATILE'


class OnChainAnalyzer:
    """
    Analyzes on-chain metrics for additional alpha.
    """
    
    def __init__(self, config: CryptoOptimizationConfig):
        self.config = config
    
    def analyze_exchange_flows(self, exchange_data: pd.DataFrame) -> Dict:
        """
        Analyze exchange inflows/outflows.
        
        Signals:
        - Large inflows → Potential selling pressure
        - Large outflows → Accumulation, bullish
        """
        
        inflow = exchange_data['inflow'].sum()
        outflow = exchange_data['outflow'].sum()
        netflow = inflow - outflow
        
        # Normalize by market cap or circulating supply
        netflow_normalized = netflow / exchange_data['circulating_supply'].iloc[0]
        
        signal = None
        if netflow_normalized > 0.01:  # 1% of supply flowing to exchanges
            signal = 'BEARISH'
        elif netflow_normalized < -0.01:
            signal = 'BULLISH'
        else:
            signal = 'NEUTRAL'
        
        return {
            'netflow': netflow,
            'netflow_normalized': netflow_normalized,
            'signal': signal,
            'confidence': min(abs(netflow_normalized) * 100, 1.0)
        }
    
    def detect_whale_activity(self, whale_wallets: List[Dict]) -> Dict:
        """
        Track large wallet movements.
        """
        
        total_whale_movement = 0
        accumulating_whales = 0
        distributing_whales = 0
        
        for wallet in whale_wallets:
            balance_change = wallet['current_balance'] - wallet['previous_balance']
            
            if abs(balance_change) > self.config.whale_wallet_threshold:
                total_whale_movement += abs(balance_change)
                
                if balance_change > 0:
                    accumulating_whales += 1
                else:
                    distributing_whales += 1
        
        # Determine signal
        whale_ratio = (accumulating_whales - distributing_whales) / len(whale_wallets) if whale_wallets else 0
        
        if whale_ratio > 0.3:
            signal = 'BULLISH'
        elif whale_ratio < -0.3:
            signal = 'BEARISH'
        else:
            signal = 'NEUTRAL'
        
        return {
            'whale_movement': total_whale_movement,
            'accumulating': accumulating_whales,
            'distributing': distributing_whales,
            'signal': signal,
            'confidence': abs(whale_ratio)
        }
    
    def analyze_funding_rates(self, funding_data: pd.DataFrame) -> Dict:
        """
        Funding rate analysis for futures markets.
        
        High positive funding → Overleveraged longs → Potential crash
        High negative funding → Overleveraged shorts → Potential squeeze
        """
        
        current_funding = funding_data['funding_rate'].iloc[-1]
        avg_funding = funding_data['funding_rate'].rolling(168).mean().iloc[-1]  # 7-day
        
        funding_extremity = (current_funding - avg_funding) / funding_data['funding_rate'].std()
        
        signal = None
        if funding_extremity > 2.0:  # 2 std above average
            signal = 'OVERLEVERAGED_LONG'  # Bearish contrarian
        elif funding_extremity < -2.0:
            signal = 'OVERLEVERAGED_SHORT'  # Bullish contrarian
        else:
            signal = 'NEUTRAL'
        
        return {
            'current_funding': current_funding,
            'avg_funding': avg_funding,
            'extremity': funding_extremity,
            'signal': signal,
            'arb_opportunity': abs(current_funding) > self.config.funding_rate_threshold
        }


class CryptoFlashCrashProtector:
    """
    Protects against crypto flash crashes and wick events.
    """
    
    def __init__(self, config: CryptoOptimizationConfig):
        self.config = config
        self.crash_history = []
    
    def detect_flash_crash(self, df: pd.DataFrame) -> bool:
        """
        Detect if a flash crash is occurring.
        """
        
        # 5-minute rolling minimum return
        returns = df['close'].pct_change()
        min_return_5min = returns.rolling(5).min().iloc[-1]
        
        # Volume spike
        volume_ratio = df['volume'].iloc[-1] / df['volume'].rolling(24).mean().iloc[-1]
        
        # Orderbook imbalance (if available)
        if 'spread' in df.columns:
            spread_widening = df['spread'].iloc[-1] / df['spread'].rolling(24).mean().iloc[-1]
        else:
            spread_widening = 1.0
        
        is_flash_crash = (
            min_return_5min < -self.config.flash_crash_threshold and
            volume_ratio > 3.0 and
            spread_widening > 2.0
        )
        
        if is_flash_crash:
            self.crash_history.append({
                'timestamp': datetime.now(),
                'drop_magnitude': min_return_5min,
                'volume_spike': volume_ratio
            })
        
        return is_flash_crash
    
    def get_protection_action(self, is_flash_crash: bool) -> str:
        """
        Determine protective action during flash crash.
        """
        
        if not is_flash_crash:
            return 'NORMAL_TRADING'
        
        # Check if this is a real crash or just a wick
        recent_crashes = [
            c for c in self.crash_history 
            if datetime.now() - c['timestamp'] < timedelta(hours=1)
        ]
        
        if len(recent_crashes) >= 3:
            return 'HALT_TRADING'  # Multiple crashes = systemic issue
        elif len(recent_crashes) == 1:
            return 'REDUCE_POSITION'  # Single crash = reduce by 50%
        else:
            return 'BUY_THE_DIP'  # Classic wick = opportunity


class CryptoCrossExchangeArbitrage:
    """
    Identifies and exploits cross-exchange arbitrage opportunities.
    """
    
    def __init__(self, config: CryptoOptimizationConfig):
        self.config = config
    
    def find_arbitrage_opportunities(
        self, 
        exchange_prices: Dict[str, float],
        exchange_fees: Dict[str, float]
    ) -> List[Dict]:
        """
        Find profitable arbitrage between exchanges.
        """
        
        opportunities = []
        
        exchanges = list(exchange_prices.keys())
        for i, exchange_a in enumerate(exchanges):
            for exchange_b in exchanges[i+1:]:
                price_a = exchange_prices[exchange_a]
                price_b = exchange_prices[exchange_b]
                fee_a = exchange_fees.get(exchange_a, 0.001)
                fee_b = exchange_fees.get(exchange_b, 0.001)
                
                # Calculate net profit after fees
                if price_a < price_b:
                    buy_cost = price_a * (1 + fee_a)
                    sell_revenue = price_b * (1 - fee_b)
                    profit_pct = (sell_revenue - buy_cost) / buy_cost
                    
                    if profit_pct > self.config.cross_exchange_spread_threshold:
                        opportunities.append({
                            'buy_exchange': exchange_a,
                            'sell_exchange': exchange_b,
                            'profit_pct': profit_pct,
                            'buy_price': price_a,
                            'sell_price': price_b
                        })
        
        return opportunities


class CryptoEnsembleOptimizer:
    """
    Extended ensemble optimizer with crypto-specific features.
    Accepts strategies and a crypto config. A base optimizer may be provided
    for mathematical optimization; otherwise a lightweight fallback is used.
    """

    def __init__(
        self,
        strategies: Optional[List] = None,
        crypto_config: Optional[CryptoOptimizationConfig] = None,
        base_optimizer: Optional[object] = None,
        **kwargs,
    ):
        # Registered strategy objects (could be agents or callable evaluators)
        self.strategies = strategies or []

        # Crypto configuration: accept either `crypto_config` or legacy `config` kw
        if crypto_config is None:
            crypto_config = kwargs.get('config', None) or kwargs.get('crypto_config', None)
            if crypto_config is None:
                crypto_config = CryptoOptimizationConfig()
        self.config = crypto_config

        # Base optimizer (optional). Provide a simple fallback if not given.
        if base_optimizer is None:
            # Lightweight fallback optimizer shim
            from types import SimpleNamespace

            class _FallbackOptimizer:
                def __init__(self):
                    cfg = SimpleNamespace()
                    cfg.lambda_risk = 1.0
                    cfg.eta_turnover = 0.0
                    self.config = cfg

                def estimate_parameters(self, strategy_returns: pd.DataFrame):
                    # Estimate mu and Sigma using simple sample moments
                    if strategy_returns is None or strategy_returns.empty:
                        return pd.Series(dtype=float), pd.DataFrame()
                    mu = strategy_returns.mean()
                    Sigma = strategy_returns.cov()
                    return mu, Sigma

                def optimize(self, mu, Sigma, regime=None):
                    # Simple equal-weight fallback
                    try:
                        n = len(mu)
                    except Exception:
                        n = 0
                    if n <= 0:
                        weights = np.array([])
                    else:
                        weights = np.ones(n) / float(n)
                    return SimpleNamespace(weights=np.array(weights))

            base_optimizer = _FallbackOptimizer()

        self.base_optimizer = base_optimizer
        self.regime_detector = CryptoRegimeDetector()
        self.on_chain_analyzer = OnChainAnalyzer(self.config)
        self.flash_crash_protector = CryptoFlashCrashProtector(self.config)
        self.arb_detector = CryptoCrossExchangeArbitrage(self.config)

    def register_strategies(self, strategies: List):
        """Register strategies after initialization if needed."""
        self.strategies = strategies or []
    
    def optimize_crypto_ensemble(
        self,
        strategy_returns: pd.DataFrame,
        market_data: pd.DataFrame,
        on_chain_data: Optional[Dict] = None,
        exchange_data: Optional[Dict] = None
    ) -> Dict:
        """
        Crypto-optimized ensemble weighting.
        """
        
        # 1. Detect crypto regime
        regime = self.regime_detector.detect_crypto_regime(
            market_data, on_chain_data
        )
        
        # 2. Check for flash crash
        is_flash_crash = self.flash_crash_protector.detect_flash_crash(market_data)
        protection_action = self.flash_crash_protector.get_protection_action(is_flash_crash)
        
        # 3. Analyze on-chain if available
        on_chain_signal = None
        if on_chain_data and self.config.use_on_chain_data:
            exchange_flow = self.on_chain_analyzer.analyze_exchange_flows(
                on_chain_data.get('exchange_flows')
            )
            whale_activity = self.on_chain_analyzer.detect_whale_activity(
                on_chain_data.get('whale_wallets', [])
            )
            funding = self.on_chain_analyzer.analyze_funding_rates(
                on_chain_data.get('funding_rates')
            )
            
            # Combine signals
            on_chain_score = (
                0.4 * (1 if exchange_flow['signal'] == 'BULLISH' else -1 if exchange_flow['signal'] == 'BEARISH' else 0) +
                0.3 * (1 if whale_activity['signal'] == 'BULLISH' else -1 if whale_activity['signal'] == 'BEARISH' else 0) +
                0.3 * (-1 if funding['signal'] == 'OVERLEVERAGED_LONG' else 1 if funding['signal'] == 'OVERLEVERAGED_SHORT' else 0)
            )
            
            on_chain_signal = {
                'score': on_chain_score,
                'exchange_flow': exchange_flow,
                'whale_activity': whale_activity,
                'funding': funding
            }
        
        # 4. Adjust parameters for crypto regime
        lambda_risk = self.base_optimizer.config.lambda_risk
        eta_turnover = self.base_optimizer.config.eta_turnover
        
        regime_adjustments = {
            'BULL_RUN': {'lambda_mult': 0.6, 'eta_mult': 1.5},
            'BEAR_CAPITULATION': {'lambda_mult': 2.0, 'eta_mult': 2.0},
            'FLASH_CRASH': {'lambda_mult': 3.0, 'eta_mult': 3.0},
            'ACCUMULATION': {'lambda_mult': 1.0, 'eta_mult': 0.8},
            'DISTRIBUTION': {'lambda_mult': 1.5, 'eta_mult': 1.2},
            'PUMP_AND_DUMP': {'lambda_mult': 2.5, 'eta_mult': 2.5},
            'WEEKEND_LULL': {'lambda_mult': 1.2, 'eta_mult': 1.5},
            'NORMAL_VOLATILE': {'lambda_mult': 1.0, 'eta_mult': 1.0}
        }
        
        adj = regime_adjustments.get(regime, {'lambda_mult': 1.0, 'eta_mult': 1.0})
        lambda_risk *= adj['lambda_mult']
        eta_turnover *= adj['eta_mult']
        
        # 5. Check for arbitrage opportunities
        arb_opportunities = []
        if exchange_data:
            arb_opportunities = self.arb_detector.find_arbitrage_opportunities(
                exchange_data.get('prices', {}),
                exchange_data.get('fees', {})
            )
        
        # 6. Run base optimization with adjusted parameters
        self.base_optimizer.config.lambda_risk = lambda_risk
        self.base_optimizer.config.eta_turnover = eta_turnover
        
        mu, Sigma = self.base_optimizer.estimate_parameters(strategy_returns)
        result = self.base_optimizer.optimize(mu, Sigma, regime=regime)
        
        # 7. Apply position scaling based on protection action
        position_scalar = 1.0
        if protection_action == 'HALT_TRADING':
            position_scalar = 0.0
        elif protection_action == 'REDUCE_POSITION':
            position_scalar = 0.5
        elif protection_action == 'BUY_THE_DIP':
            position_scalar = 1.2  # Increase slightly
        
        # 8. Apply on-chain adjustment
        if on_chain_signal and self.config.use_on_chain_data:
            on_chain_adjustment = 1.0 + (on_chain_signal['score'] * self.config.exchange_flow_weight)
            position_scalar *= on_chain_adjustment
        
        # Scale weights
        adjusted_weights = result.weights * position_scalar
        
        # Renormalize if needed
        if adjusted_weights.sum() > 0:
            adjusted_weights /= adjusted_weights.sum()
        
        return {
            'weights': adjusted_weights,
            'regime': regime,
            'protection_action': protection_action,
            'position_scalar': position_scalar,
            'on_chain_signal': on_chain_signal,
            'arbitrage_opportunities': arb_opportunities,
            'base_result': result,
            'is_flash_crash': is_flash_crash
        }
