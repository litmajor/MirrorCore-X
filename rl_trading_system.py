import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import torch
import torch.nn as nn
from stable_baselines3 import PPO, SAC, A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from typing import Dict, List, Optional, Union, Tuple, Any
import logging
import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
import pickle
import warnings
warnings.filterwarnings('ignore')

# Import your existing scanner
from scanner import MomentumScanner, TradingConfig, get_dynamic_config

logger = logging.getLogger(__name__)

@dataclass
class RLConfig:
    """Configuration for RL trading agent"""
    lookback_window: int = 20
    max_position_size: float = 1.0
    transaction_cost: float = 0.001
    slippage: float = 0.0005
    reward_scaling: float = 1.0
    risk_penalty: float = 0.1
    drawdown_penalty: float = 0.2
    sharpe_reward_weight: float = 0.3
    return_reward_weight: float = 0.7
    learning_rate: float = 3e-4
    total_timesteps: int = 100000
    eval_freq: int = 5000
    save_freq: int = 10000

class SignalEncoder:
    """Enhanced signal encoder for RL features"""
    
    def __init__(self, scaling_method: str = 'robust'):
        self.scaling_method = scaling_method
        if scaling_method == 'robust':
            self.scaler = RobustScaler()
        else:
            self.scaler = MinMaxScaler()
        self.feature_names = [
            'price', 'volume', 'momentum_short', 'momentum_long', 
            'rsi', 'macd', 'bb_position', 'volume_ratio',
            'composite_score', 'trend_score', 'confidence_score',
            'fear_greed', 'btc_dominance', 'volatility',
            'ichimoku_bullish', 'vwap_bullish', 'ema_crossover'
        ]
        self.fitted = False
    
    def encode(self, df: pd.DataFrame) -> np.ndarray:
        """Encode DataFrame into normalized feature matrix"""
        # Ensure required columns exist
        for col in self.feature_names:
            if col not in df.columns:
                # Provide sensible defaults
                if 'bullish' in col or 'crossover' in col:
                    df[col] = 0.0  # Boolean features
                elif col in ['fear_greed', 'btc_dominance']:
                    df[col] = 50.0  # Neutral values
                elif col == 'rsi':
                    df[col] = 50.0  # Neutral RSI
                else:
                    df[col] = 0.0
        
        # Extract features
        features = df[self.feature_names].fillna(0.0)
        
        # Normalize
        if not self.fitted:
            normalized = self.scaler.fit_transform(features)
            self.fitted = True
        else:
            normalized = self.scaler.transform(features)
        
        return normalized.astype(np.float32)
    
    def transform_live(self, latest_data: Dict) -> np.ndarray:
        """Transform single observation for live trading"""
        row = np.array([[latest_data.get(k, self._get_default_value(k)) 
                        for k in self.feature_names]])
        return self.scaler.transform(row).astype(np.float32)
    
    def _get_default_value(self, feature: str) -> float:
        """Get default value for missing features"""
        if 'bullish' in feature or 'crossover' in feature:
            return 0.0
        elif feature in ['fear_greed', 'btc_dominance', 'rsi']:
            return 50.0
        else:
            return 0.0

class TradingEnvironment(gym.Env):
    """Enhanced trading environment for RL training"""
    
    def __init__(self, 
                 market_data: np.ndarray,
                 price_data: np.ndarray,
                 config: Optional[RLConfig] = None,
                 initial_balance: float = 10000.0):
        super().__init__()
        
        self.config = config or RLConfig()
        self.market_data = market_data
        self.price_data = price_data
        self.initial_balance = initial_balance
        
        # Environment parameters
        self.lookback = self.config.lookback_window
        self.max_steps = len(market_data) - self.lookback - 1
        
        # Action space: [position_change] where position_change ∈ [-1, 1]
        # -1 = max short, 0 = neutral, 1 = max long
        self.action_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(1,), 
            dtype=np.float32
        )
        
        # Observation space: market features + portfolio state
        n_features = market_data.shape[1]
        portfolio_features = 4  # position, balance, pnl, drawdown
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.lookback, n_features + portfolio_features),
            dtype=np.float32
        )
        
        # Initialize state
        self.reset()
    
    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        self.current_step = self.lookback
        self.balance = self.initial_balance
        self.position = 0.0  # Current position size (-1 to 1)
        self.entry_price = 0.0
        self.total_pnl = 0.0
        self.max_balance = self.initial_balance
        self.trade_count = 0
        self.win_count = 0
        
        # Performance tracking
        self.balance_history = [self.initial_balance]
        self.position_history = [0.0]
        self.action_history = []
        self.reward_history = []
        
        return self._get_observation(), {}
    
    def step(self, action: np.ndarray):
        """Execute one step in the environment"""
        if self.current_step >= self.max_steps:
            return self._get_observation(), 0.0, True, False, {}
        
        # Parse action
        target_position = np.clip(action[0], -1.0, 1.0)
        position_change = target_position - self.position
        
        # Calculate current price
        current_price = self.price_data[self.current_step]
        
        # Execute trade if position changes
        reward = 0.0
        if abs(position_change) > 0.01:  # Minimum trade threshold
            reward += self._execute_trade(position_change, current_price)
        
        # Calculate unrealized PnL
        if self.position != 0:
            price_change = (current_price - self.entry_price) / self.entry_price
            unrealized_pnl = self.position * price_change * self.balance
            self.total_pnl = unrealized_pnl
        
        # Update balance
        self.balance = self.initial_balance + self.total_pnl
        self.max_balance = max(self.max_balance, self.balance)
        
        # Calculate reward components
        reward += self._calculate_reward(current_price)
        
        # Update state
        self.current_step += 1
        self.balance_history.append(self.balance)
        self.position_history.append(self.position)
        self.action_history.append(target_position)
        self.reward_history.append(reward)
        
        # Check if done
        done = (self.current_step >= self.max_steps or 
                self.balance <= self.initial_balance * 0.5)  # 50% drawdown limit
        
        info = {
            'balance': self.balance,
            'position': self.position,
            'pnl': self.total_pnl,
            'trade_count': self.trade_count,
            'win_rate': self.win_count / max(self.trade_count, 1),
            'drawdown': (self.max_balance - self.balance) / self.max_balance
        }
        
        return self._get_observation(), reward, done, False, info
    
    def _execute_trade(self, position_change: float, current_price: float) -> float:
        """Execute trade and return immediate reward/penalty"""
        # Transaction costs
        trade_cost = abs(position_change) * self.config.transaction_cost * self.balance
        
        # Update position
        old_position = self.position
        self.position = np.clip(self.position + position_change, -1.0, 1.0)
        
        # Update entry price (volume-weighted)
        if abs(self.position) > 0.01:
            if abs(old_position) < 0.01:  # New position
                self.entry_price = current_price
            elif np.sign(self.position) == np.sign(old_position):  # Adding to position
                # Volume-weighted average entry price
                old_value = abs(old_position) * self.entry_price
                new_value = abs(position_change) * current_price
                total_volume = abs(old_position) + abs(position_change)
                self.entry_price = (old_value + new_value) / total_volume
        
        # Close position tracking
        if abs(old_position) > 0.01 and abs(self.position) < 0.01:
            # Position closed
            price_change = (current_price - self.entry_price) / self.entry_price
            trade_pnl = old_position * price_change * self.balance
            
            self.trade_count += 1
            if trade_pnl > 0:
                self.win_count += 1
        
        return -trade_cost / self.initial_balance  # Normalize cost
    
    def _calculate_reward(self, current_price: float) -> float:
        """Calculate step reward based on multiple factors"""
        reward = 0.0
        
        # Return-based reward
        if len(self.balance_history) > 1:
            balance_return = (self.balance - self.balance_history[-2]) / self.balance_history[-2]
            reward += balance_return * self.config.return_reward_weight
        
        # Risk-adjusted reward (Sharpe-like)
        if len(self.balance_history) > 10:
            returns = np.diff(self.balance_history[-10:]) / np.array(self.balance_history[-11:-1])
            if len(returns) > 1 and np.std(returns) > 0:
                sharpe_approx = np.mean(returns) / np.std(returns)
                reward += sharpe_approx * self.config.sharpe_reward_weight
        
        # Drawdown penalty
        drawdown = (self.max_balance - self.balance) / self.max_balance
        if drawdown > 0.1:  # Penalty for >10% drawdown
            reward -= drawdown * self.config.drawdown_penalty
        
        # Risk penalty for extreme positions
        risk_penalty = abs(self.position) ** 2 * self.config.risk_penalty
        reward -= risk_penalty
        
        return float(reward * self.config.reward_scaling)
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation"""
        # Market features for lookback window
        start_idx = max(0, self.current_step - self.lookback)
        end_idx = self.current_step
        
        market_features = self.market_data[start_idx:end_idx]
        
        # Portfolio features
        portfolio_state = np.array([
            [self.position, 
             self.balance / self.initial_balance - 1,  # Normalized balance change
             self.total_pnl / self.initial_balance,    # Normalized PnL
             (self.max_balance - self.balance) / self.max_balance]  # Drawdown
        ])
        
        # Repeat portfolio state for each timestep in lookback
        portfolio_features = np.repeat(portfolio_state, self.lookback, axis=0)
        
        # Pad market features if needed
        if market_features.shape[0] < self.lookback:
            padding = np.zeros((self.lookback - market_features.shape[0], market_features.shape[1]))
            market_features = np.vstack([padding, market_features])
        
        # Combine features
        observation = np.concatenate([market_features, portfolio_features], axis=1)
        
        return observation.astype(np.float32)
    
    def render(self, mode='human'):
        """Render environment state"""
        if mode == 'human':
            print(f"Step: {self.current_step}, Balance: ${self.balance:.2f}, "
                  f"Position: {self.position:.2f}, PnL: ${self.total_pnl:.2f}")

class RLTradingAgent:
    """Main RL trading agent class"""
    
    def __init__(self, 
                 algorithm: str = 'PPO',
                 config: Optional[RLConfig] = None,
                 model_path: Optional[str] = None):
        self.algorithm = algorithm
        self.config = config or RLConfig()
        self.model = None
        self.env = None
        self.encoder = SignalEncoder()
        self.is_trained = False
        
        if model_path:
            self.load_model(model_path)
    
    def prepare_training_data(self, scanner_results: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data from scanner results"""
        # Add market sentiment features if available
        if hasattr(scanner_results, 'fear_greed_history') and hasattr(scanner_results.fear_greed_history, '__len__') and len(scanner_results.fear_greed_history) > 0:
            scanner_results['fear_greed'] = scanner_results.fear_greed_history[-1]
        else:
            scanner_results['fear_greed'] = 50
            
        if hasattr(scanner_results, 'btc_dominance_history') and hasattr(scanner_results.btc_dominance_history, '__len__') and len(scanner_results.btc_dominance_history) > 0:
            scanner_results['btc_dominance'] = scanner_results.btc_dominance_history[-1]
        else:
            scanner_results['btc_dominance'] = 50
        
        # Calculate volatility
        scanner_results['volatility'] = scanner_results['momentum_short'].rolling(5).std().fillna(0)
        
        # Add technical indicators as boolean features
        if 'ichimoku_bullish' in scanner_results.columns:
            scanner_results['ichimoku_bullish'] = scanner_results['ichimoku_bullish'].astype(float)
        else:
            scanner_results['ichimoku_bullish'] = pd.Series([False] * len(scanner_results)).astype(float)
        if 'vwap_bullish' in scanner_results.columns:
            scanner_results['vwap_bullish'] = scanner_results['vwap_bullish'].astype(float)
        else:
            scanner_results['vwap_bullish'] = 0.0
        scanner_results['ema_crossover'] = np.array(scanner_results.get('ema_5_13_bullish', False), dtype=float)
        
        # Encode features
        market_features = self.encoder.encode(scanner_results)
        price_data = np.array(scanner_results['price'].values, dtype=np.float32)
        
        return market_features, price_data
    
    def create_environment(self, market_data: np.ndarray, price_data: np.ndarray) -> TradingEnvironment:
        """Create trading environment"""
        env = TradingEnvironment(
            market_data=market_data,
            price_data=price_data,
            config=self.config
        )
        return env
    
    def train(self, 
              scanner_results: pd.DataFrame,
              save_path: str = 'rl_trading_model',
              verbose: int = 1) -> Dict[str, Any]:
        """Train the RL agent"""
        logger.info(f"Starting RL training with {self.algorithm}")
        
        # Prepare data
        market_data, price_data = self.prepare_training_data(scanner_results)
        
        # Create environment
        env = self.create_environment(market_data, price_data)
        
        # Create model based on algorithm
        if self.algorithm == 'PPO':
            self.model = PPO(
                'MlpPolicy',
                env,
                learning_rate=self.config.learning_rate,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                clip_range_vf=None,
                ent_coef=0.01,
                vf_coef=0.5,
                max_grad_norm=0.5,
                verbose=verbose
            )
        elif self.algorithm == 'SAC':
            self.model = SAC(
                'MlpPolicy',
                env,
                learning_rate=self.config.learning_rate,
                buffer_size=100000,
                learning_starts=1000,
                batch_size=256,
                tau=0.005,
                gamma=0.99,
                verbose=verbose
            )
        elif self.algorithm == 'A2C':
            self.model = A2C(
                'MlpPolicy',
                env,
                learning_rate=self.config.learning_rate,
                n_steps=5,
                gamma=0.99,
                gae_lambda=0.95,
                ent_coef=0.01,
                vf_coef=0.5,
                max_grad_norm=0.5,
                verbose=verbose
            )
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
        
        # Training callback
        callback = TrainingCallback(
            eval_freq=self.config.eval_freq,
            save_freq=self.config.save_freq,
            save_path=save_path
        )
        
        # Train model
        self.model.learn(
            total_timesteps=self.config.total_timesteps,
            callback=callback
        )
        
        # Save final model
        self.save_model(save_path)
        self.is_trained = True
        
        logger.info("RL training completed")
        
        return {
            'training_completed': True,
            'total_timesteps': self.config.total_timesteps,
            'algorithm': self.algorithm,
            'final_model_path': save_path
        }
    
    def predict(self, observation: np.ndarray, deterministic: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Predict action for given observation"""
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        action, state = self.model.predict(observation, deterministic=deterministic)
        return action, state
    
    def evaluate(self, 
                 test_data: pd.DataFrame,
                 episodes: int = 10) -> Dict[str, float]:
        """Evaluate trained model on test data"""
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        market_data, price_data = self.prepare_training_data(test_data)
        env = self.create_environment(market_data, price_data)
        
        episode_rewards = []
        episode_returns = []
        win_rates = []
        
        for episode in range(episodes):
            obs, _ = env.reset()
            total_reward = 0.0
            done = False
            info = {}  # Ensure info is always defined
            
            while not done:
                action, _ = self.predict(obs, deterministic=True)
                obs, reward, done, _, info = env.step(action)
                total_reward += reward
            
            episode_rewards.append(total_reward)
            episode_returns.append((info.get('balance', env.initial_balance) - env.initial_balance) / env.initial_balance)
            win_rates.append(info.get('win_rate', 0.0))
        
        return {
            'mean_reward': float(np.mean(episode_rewards)),
            'std_reward': float(np.std(episode_rewards)),
            'mean_return': float(np.mean(episode_returns)),
            'std_return': float(np.std(episode_returns)),
            'mean_win_rate': float(np.mean(win_rates)),
            'sharpe_ratio': float(np.mean(episode_returns) / np.std(episode_returns)) if np.std(episode_returns) > 0 else 0.0
        }
    
    def save_model(self, path: str):
        """Save model and encoder"""
        if self.model is None:
            raise ValueError("No model to save")
        
        self.model.save(path)
        
        # Save encoder separately
        with open(f"{path}_encoder.pkl", 'wb') as f:
            pickle.dump(self.encoder, f)
        
        logger.info(f"Model and encoder saved to {path}")
    
    def load_model(self, path: str):
        """Load model and encoder"""
        try:
            # Load model
            if self.algorithm == 'PPO':
                self.model = PPO.load(path)
            elif self.algorithm == 'SAC':
                self.model = SAC.load(path)
            elif self.algorithm == 'A2C':
                self.model = A2C.load(path)
            
            # Load encoder
            with open(f"{path}_encoder.pkl", 'rb') as f:
                self.encoder = pickle.load(f)
            
            self.is_trained = True
            logger.info(f"Model and encoder loaded from {path}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

class TrainingCallback(BaseCallback):
    """Callback for monitoring training progress"""
    
    def __init__(self, 
                 eval_freq: int = 5000,
                 save_freq: int = 10000,
                 save_path: str = 'model_checkpoint'):
        super().__init__()
        self.eval_freq = eval_freq
        self.save_freq = save_freq
        self.save_path = save_path
    
    def _on_step(self) -> bool:
        # Save model checkpoint
        if self.n_calls % self.save_freq == 0:
            self.model.save(f"{self.save_path}_step_{self.n_calls}")
            logger.info(f"Model checkpoint saved at step {self.n_calls}")
        
        # Log progress
        if self.n_calls % self.eval_freq == 0:
            logger.info(f"Training step: {self.n_calls}")
        
        return True

class MetaController:
    """Meta-controller for combining rule-based and RL decisions"""
    
    def __init__(self, strategy: str = "confidence_blend"):
        self.strategy = strategy
        self.performance_history = []
    
    def decide(self, 
               rule_decision: str,
               rl_action: np.ndarray,
               confidence_score: float = 0.5,
               market_regime: str = "normal") -> Dict[str, Any]:
        """
        Combine rule-based and RL decisions
        
        Args:
            rule_decision: Rule-based signal ('Strong Buy', 'Buy', etc.)
            rl_action: RL action (position size)
            confidence_score: Confidence in the signal
            market_regime: Current market regime
        """
        
        # Convert rule decision to position size
        rule_position = self._rule_to_position(rule_decision)
        rl_position = float(rl_action[0]) if isinstance(rl_action, np.ndarray) else rl_action
        
        if self.strategy == "rule_only":
            final_position = rule_position
            method = "rule_based"
            
        elif self.strategy == "rl_only":
            final_position = rl_position
            method = "reinforcement_learning"
            
        elif self.strategy == "confidence_blend":
            # Blend based on confidence score
            if confidence_score > 0.7:
                # High confidence: prefer RL
                final_position = 0.7 * rl_position + 0.3 * rule_position
                method = "rl_weighted"
            elif confidence_score > 0.4:
                # Medium confidence: equal weight
                final_position = 0.5 * rl_position + 0.5 * rule_position
                method = "balanced"
            else:
                # Low confidence: prefer rules
                final_position = 0.3 * rl_position + 0.7 * rule_position
                method = "rule_weighted"
                
        elif self.strategy == "regime_adaptive":
            # Adapt based on market regime
            if market_regime == "trending":
                # In trending markets, prefer RL
                final_position = 0.8 * rl_position + 0.2 * rule_position
                method = "trend_adaptive"
            elif market_regime == "volatile":
                # In volatile markets, prefer rules
                final_position = 0.2 * rl_position + 0.8 * rule_position
                method = "volatility_adaptive"
            else:
                # Normal markets: balanced
                final_position = 0.6 * rl_position + 0.4 * rule_position
                method = "regime_balanced"
        
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        # Clip final position
        final_position = np.clip(final_position, -1.0, 1.0)
        
        decision = {
            'final_position': final_position,
            'rule_position': rule_position,
            'rl_position': rl_position,
            'confidence_score': confidence_score,
            'method': method,
            'market_regime': market_regime
        }
        
        self.performance_history.append(decision)
        return decision
    
    def _rule_to_position(self, rule_decision: str) -> float:
        """Convert rule-based signal to position size"""
        signal_map = {
            'Strong Buy': 1.0,
            'Buy': 0.6,
            'Weak Buy': 0.3,
            'Neutral': 0.0,
            'Weak Sell': -0.3,
            'Sell': -0.6,
            'Strong Sell': -1.0,
            'Overbought': -0.4,
            'Oversold': 0.4
        }
        return signal_map.get(rule_decision, 0.0)

class IntegratedTradingSystem:
    """Main system integrating momentum scanner with RL agent"""
    
    def __init__(self, 
                 scanner: 'MomentumScanner',
                 rl_agent: RLTradingAgent,
                 meta_controller: MetaController):
        self.scanner = scanner
        self.rl_agent = rl_agent
        self.meta_controller = meta_controller
        self.trading_history = []
    
    async def generate_signals(self, timeframe: str = 'daily') -> pd.DataFrame:
        """Generate trading signals using both rule-based and RL approaches"""
        # Get scanner results
        scanner_results = await self.scanner.scan_market(timeframe=timeframe)
        
        if scanner_results.empty:
            logger.warning("No scanner results available")
            return pd.DataFrame()
        
        # Prepare signals DataFrame
        signals_df = scanner_results.copy()
        
        # Add RL predictions if agent is trained
        if self.rl_agent.is_trained:
            rl_positions = []
            
            for _, row in scanner_results.iterrows():
                # Prepare observation for RL agent
                observation_data = {
                    'price': row['price'],
                    'volume': row.get('volume_usd', 0),
                    'momentum_short': row['momentum_short'],
                    'momentum_long': row['momentum_long'],
                    'rsi': row['rsi'],
                    'macd': row['macd'],
                    'bb_position': row.get('bb_position', 0.5),
                    'volume_ratio': row['volume_ratio'],
                    'composite_score': row['composite_score'],
                    'trend_score': row['trend_score'],
                    'confidence_score': row['confidence_score'],
                    'fear_greed': 50,  # Default if not available
                    'btc_dominance': 50,  # Default if not available
                    'volatility': row.get('volatility', 0),
                    'ichimoku_bullish': float(row.get('ichimoku_bullish', False)),
                    'vwap_bullish': float(row.get('vwap_bullish', False)),
                    'ema_crossover': float(row.get('ema_5_13_bullish', False))
                }
                
                # Get RL prediction
                obs = self.rl_agent.encoder.transform_live(observation_data)
                # Create dummy observation with lookback dimension
                dummy_obs = np.repeat(obs, self.rl_agent.config.lookback_window, axis=0)
                rl_action, _ = self.rl_agent.predict(dummy_obs)
                rl_positions.append(float(rl_action[0]))
            
            signals_df['rl_position'] = rl_positions
        else:
            signals_df['rl_position'] = 0.0
        
        # Generate meta-controller decisions
        meta_decisions = []
        for _, row in signals_df.iterrows():
            decision = self.meta_controller.decide(
                rule_decision=row['signal'],
                rl_action=np.array([row['rl_position']]),
                confidence_score=row['confidence_score'],
                market_regime=self._detect_market_regime(row)
            )
            meta_decisions.append(decision)
        
        # Add meta-controller results
        meta_df = pd.DataFrame(meta_decisions)
        signals_df = pd.concat([signals_df, meta_df], axis=1)
        
        return signals_df
    
    def _detect_market_regime(self, row: pd.Series) -> str:
        """Detect current market regime based on indicators"""
        volatility = abs(row['momentum_short'])
        trend_strength = row['trend_score']
        
        if volatility > 0.05:
            return "volatile"
        elif trend_strength > 7:
            return "trending"
        else:
            return "normal"
    
    async def execute_trading_session(self, 
                                    timeframe: str = 'daily',
                                    dry_run: bool = True) -> Dict[str, Any]:
        """Execute a complete trading session"""
        logger.info(f"Starting integrated trading session - timeframe: {timeframe}, dry_run: {dry_run}")
        
        try:
            # Generate signals
            signals = await self.generate_signals(timeframe)
            
            if signals.empty:
                return {'status': 'no_signals', 'trades': []}
            
            # Filter for actionable signals
            actionable = signals[abs(signals['final_position']) > 0.1]  # Minimum 10% position
            
            trades = []
            for _, signal in actionable.iterrows():
                trade = {
                    'symbol': signal['symbol'],
                    'signal_type': signal['signal'],
                    'rule_position': signal['rule_position'],
                    'rl_position': signal['rl_position'],
                    'final_position': signal['final_position'],
                    'confidence': signal['confidence_score'],
                    'method': signal['method'],
                    'price': signal['price'],
                    'composite_score': signal['composite_score'],
                    'timestamp': datetime.now(timezone.utc),
                    'dry_run': dry_run
                }
                
                if not dry_run:
                    # Execute actual trade (implement exchange integration here)
                    trade['executed'] = await self._execute_real_trade(trade)
                else:
                    trade['executed'] = True  # Simulate successful execution
                
                trades.append(trade)
                self.trading_history.append(trade)
            
            session_result = {
                'status': 'completed',
                'timeframe': timeframe,
                'total_signals': len(signals),
                'actionable_signals': len(actionable),
                'trades_executed': len([t for t in trades if t['executed']]),
                'trades': trades,
                'dry_run': dry_run,
                'session_timestamp': datetime.now(timezone.utc)
            }
            
            logger.info(f"Trading session completed: {session_result['trades_executed']} trades executed")
            return session_result
            
        except Exception as e:
            logger.error(f"Trading session failed: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def _execute_real_trade(self, trade: Dict) -> bool:
        """Execute real trade on exchange (placeholder for implementation)"""
        # This would integrate with your exchange API
        # For now, return True to simulate successful execution
        logger.info(f"Would execute real trade: {trade['symbol']} position: {trade['final_position']:.2f}")
        return True
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate performance report for the integrated system"""
        if not self.trading_history:
            return {'status': 'no_trades'}
        
        df = pd.DataFrame(self.trading_history)
        
        # Performance metrics
        total_trades = len(df)
        successful_trades = len(df[df['executed'] == True])
        
        # Method distribution
        method_dist = df['method'].value_counts().to_dict()
        
        # Confidence analysis
        avg_confidence = df['confidence'].mean()
        high_confidence_trades = len(df[df['confidence'] > 0.7])
        
        # Position analysis
        long_positions = len(df[df['final_position'] > 0])
        short_positions = len(df[df['final_position'] < 0])
        
        return {
            'total_trades': total_trades,
            'successful_executions': successful_trades,
            'execution_rate': successful_trades / total_trades if total_trades > 0 else 0,
            'method_distribution': method_dist,
            'average_confidence': avg_confidence,
            'high_confidence_trades': high_confidence_trades,
            'long_short_ratio': f"{long_positions}/{short_positions}",
            'meta_controller_performance': self.meta_controller.performance_history[-10:] if self.meta_controller.performance_history else []
        }
# Usage Example and Training Pipeline
class RLTrainingPipeline:
    """Complete pipeline for training and deploying the RL trading system"""
    
    def __init__(self, scanner: 'MomentumScanner'):
        self.scanner = scanner
        self.rl_agent = None
        self.meta_controller = None
        self.integrated_system = None
    
    async def run_full_pipeline(self, 
                               training_timeframe: str = 'daily',
                               algorithm: str = 'PPO',
                               total_timesteps: int = 100000) -> Dict[str, Any]:
        """Run the complete training and deployment pipeline"""
        
        # Step 1: Collect training data
        logger.info("Step 1: Collecting training data...")
        training_data = await self._collect_training_data(training_timeframe)
        
        if training_data.empty:
            raise ValueError("No training data available")
        
        # Step 2: Initialize and train RL agent
        logger.info("Step 2: Training RL agent...")
        config = RLConfig(total_timesteps=total_timesteps)
        self.rl_agent = RLTradingAgent(algorithm=algorithm, config=config)
        
        training_results = self.rl_agent.train(
            training_data, 
            save_path=f'models/rl_{algorithm.lower()}_model'
        )
        
        # Step 3: Evaluate on test data
        logger.info("Step 3: Evaluating trained model...")
        test_data = await self._collect_test_data(training_timeframe)
        evaluation_results = self.rl_agent.evaluate(test_data)
        
        # Step 4: Initialize meta-controller
        logger.info("Step 4: Initializing meta-controller...")
        self.meta_controller = MetaController(strategy="confidence_blend")
        
        # Step 5: Create integrated system
        logger.info("Step 5: Creating integrated trading system...")
        self.integrated_system = IntegratedTradingSystem(
            scanner=self.scanner,
            rl_agent=self.rl_agent,
            meta_controller=self.meta_controller
        )
        
        # Step 6: Run test trading session
        logger.info("Step 6: Running test trading session...")
        test_session = await self.integrated_system.execute_trading_session(
            timeframe=training_timeframe,
            dry_run=True
        )
        
        pipeline_results = {
            'training_results': training_results,
            'evaluation_results': evaluation_results,
            'test_session': test_session,
            'model_ready': True,
            'pipeline_timestamp': datetime.now(timezone.utc)
        }
        
        logger.info("Pipeline completed successfully!")
        return pipeline_results
    
    async def _collect_training_data(self, timeframe: str) -> pd.DataFrame:
        """Collect historical data for training"""
        # Scan market with extended history
        training_data = await self.scanner.scan_market(
            timeframe=timeframe,
            full_analysis=True
        )
        
        # Add some synthetic historical data if needed
        if len(training_data) < 100:  # Minimum required for training
            logger.warning("Limited training data available, consider collecting more historical data")
        
        return training_data
    
    async def _collect_test_data(self, timeframe: str) -> pd.DataFrame:
        """Collect test data (could be out-of-sample)"""
        # For now, use a subset of recent data
        # In practice, you'd use completely separate historical periods
        test_data = await self.scanner.scan_market(timeframe=timeframe)
        return test_data.tail(50)  # Use last 50 observations for testing

# Automated Optimization System
class HyperparameterOptimizer:
    """Optimize RL hyperparameters using the scanner's OptimizableAgent interface"""
    
    def __init__(self, 
                 scanner: 'MomentumScanner',
                 search_space: Optional[Dict[str, Tuple[float, float]]] = None):
        self.scanner = scanner
        self.search_space = search_space or {
            'learning_rate': (1e-5, 1e-2),
            'lookback_window': (10, 50),
            'transaction_cost': (0.0001, 0.01),
            'reward_scaling': (0.1, 10.0)
        }
        self.best_params = None
        self.best_score = float('-inf')
        self.optimization_history = []
    
    def optimize(self, 
                 training_data: pd.DataFrame,
                 n_trials: int = 20,
                 algorithm: str = 'PPO') -> Dict[str, Any]:
        """Optimize hyperparameters using random search"""
        
        logger.info(f"Starting hyperparameter optimization with {n_trials} trials")
        
        for trial in range(n_trials):
            # Sample hyperparameters
            params = self._sample_hyperparameters()
            
            try:
                # Create and train agent with sampled parameters
                config = RLConfig(
                    learning_rate=params['learning_rate'],
                    lookback_window=int(params['lookback_window']),
                    transaction_cost=params['transaction_cost'],
                    reward_scaling=params['reward_scaling'],
                    total_timesteps=20000  # Shorter for optimization
                )
                
                agent = RLTradingAgent(algorithm=algorithm, config=config)
                agent.train(training_data, save_path=f'temp_model_{trial}', verbose=0)
                
                # Evaluate agent
                eval_results = agent.evaluate(training_data.tail(100))
                score = eval_results['sharpe_ratio']  # Use Sharpe ratio as optimization metric
                
                # Update best parameters
                if score > self.best_score:
                    self.best_score = score
                    self.best_params = params.copy()
                
                self.optimization_history.append({
                    'trial': trial,
                    'params': params,
                    'score': score,
                    'eval_results': eval_results
                })
                
                logger.info(f"Trial {trial + 1}/{n_trials}: Score = {score:.4f}")
                
            except Exception as e:
                logger.error(f"Trial {trial + 1} failed: {e}")
                continue
        
        optimization_results = {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'optimization_history': self.optimization_history,
            'total_trials': len(self.optimization_history)
        }
        
        logger.info(f"Optimization completed. Best score: {self.best_score:.4f}")
        return optimization_results
    
    def _sample_hyperparameters(self) -> Dict[str, float]:
        """Sample hyperparameters from search space"""
        params = {}
        for param, (low, high) in self.search_space.items():
            if param == 'lookback_window':
                params[param] = np.random.uniform(low, high)
            else:
                params[param] = np.random.uniform(low, high)
        return params

# Example usage and integration
async def demo_rl_trading_system():
    """Demonstration of the complete RL trading system"""
    
    # This assumes you have the MomentumScanner available
    # from your existing scanner.py file
    
    try:
        # Initialize exchange (placeholder - use your existing exchange setup)
        import ccxt.async_support as ccxt_async
        exchange = ccxt_async.binance({
            'enableRateLimit': True,
            'rateLimit': 100,
        })
        
        # Initialize scanner
        scanner = MomentumScanner(
            exchange=exchange,
            market_type='crypto',
            quote_currency='USDT',
            min_volume_usd=1_000_000,
            top_n=20
        )
        
        # Run training pipeline
        pipeline = RLTrainingPipeline(scanner)
        results = await pipeline.run_full_pipeline(
            algorithm='PPO',
            total_timesteps=50000
        )
        
        print("=== RL TRADING SYSTEM RESULTS ===")
        print(f"Training completed: {results['training_results']['training_completed']}")
        print(f"Evaluation Sharpe Ratio: {results['evaluation_results']['sharpe_ratio']:.3f}")
        print(f"Test session trades: {results['test_session']['trades_executed']}")
        
        # Get performance report
        if pipeline.integrated_system is not None:
            performance = pipeline.integrated_system.get_performance_report()
            print(f"System performance: {performance}")
        else:
            print("Integrated trading system is not initialized; cannot get performance report.")
        
        # Run hyperparameter optimization (optional)
        if len(scanner.scan_results) > 100:  # Only if we have enough data
            optimizer = HyperparameterOptimizer(scanner)
            training_data = await scanner.scan_market('daily')
            opt_results = optimizer.optimize(training_data, n_trials=10)
            print(f"Best hyperparameters: {opt_results['best_params']}")
        
        # Close exchange
        await exchange.close()
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")

# Main execution
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Run demo
    asyncio.run(demo_rl_trading_system())