# MirrorCore-X

## Overview

MirrorCore-X is an advanced multi-agent cognitive trading system designed for cryptocurrency and forex markets. The system employs a novel **SyncBus** architecture that coordinates multiple specialized agents to analyze markets, manage risk, execute trades, and optimize strategies. The architecture integrates cognitive psychology principles with algorithmic trading, featuring emotion-aware decision-making and sophisticated multi-timeframe market analysis.

Key capabilities include:
- Multi-agent coordination via high-performance asynchronous SyncBus
- Real-time market scanning across multiple exchanges and timeframes
- Bayesian belief systems for strategy optimization
- Reinforcement learning for adaptive trading
- Counterfactual scenario generation ("Imagination Engine")
- Advanced risk management with circuit breakers and emergency controls
- Real-time web dashboard with WebSocket updates
- Support for both paper trading and live execution

## User Preferences

Preferred communication style: Simple, everyday language.

## Brand Identity System

**Brand Theme**: Futuristic cyberpunk aesthetic with neon glows, glass morphism effects, and neural network visualization

**Brand Colors**:
- **Primary Cyan**: `#00D9FF` - High-tech, digital intelligence
- **Secondary Purple**: `#7B2FFF` - Cognitive processing, AI
- **Success Green**: `#00FF88` - Profitable trades
- **Warning Amber**: `#FFB800` - Caution states
- **Error Red**: `#FF3366` - Risk alerts

**Typography**:
- **Display Font**: Orbitron (headings, brand name)
- **UI Font**: Rajdhani (buttons, labels, metrics)
- **Body Font**: Inter (paragraphs, descriptions)
- **Code Font**: JetBrains Mono (technical data)

**Brand Assets** (all in `brand/` directory):
- Logo variants (primary, horizontal, icon) in SVG format with light/dark versions
- Social media assets (profile images, banners for Twitter/LinkedIn/GitHub)
- Complete favicon package for all platforms (web, iOS, Android)
- Interactive brand showcase at http://0.0.0.0:5000/
- Complete style guide in JSON format

**Frontend Theme Implementation**:
- Unified theme system integrated with Tailwind CSS (`tailwind.config.js`)
- Custom CSS utilities for components (`src/index.css`)
- Brand tokens in TypeScript (`src/theme/brand.ts`)
- Consistent styling across all React pages (Dashboard, Trading, Analytics, RiskManagement, Settings, Strategies)
- Glass morphism cards with cyan/purple glows
- Neon text effects and gradient backgrounds

## System Architecture

### Core Coordination Layer: HighPerformanceSyncBus

**Problem Addressed**: Traditional multi-agent systems suffer from excessive communication overhead, lack of fault isolation, and poor scalability when coordinating 10+ specialized agents in real-time trading scenarios.

**Solution**: The HighPerformanceSyncBus implements a custom async coordination pattern with:
- **Delta Updates**: Only transmits changed state between agents (reduces bandwidth 70-90%)
- **Interest-Based Routing**: Agents declare data interests on registration and only receive relevant updates
- **Circuit Breaker Protection**: Isolates failing agents to prevent cascade failures (5+ failures triggers circuit open)
- **Command Queue**: Supports pause/resume and real-time agent control
- **Separated Pipelines**: Market data flows independently from agent state management

**Alternatives Considered**: Traditional event buses, actor model frameworks, shared memory approaches

**Pros**: Low latency (10-50ms vs 100-500ms), scales to 100+ agents, fault-resistant
**Cons**: Custom implementation requires maintenance, learning curve for developers

### Agent Architecture

**Problem Addressed**: Need specialized processing for different aspects of trading (technical analysis, risk, execution, psychology, optimization) with clear separation of concerns.

**Solution**: Multi-agent system with specialized agents:

1. **MomentumScanner**: Multi-timeframe market analysis (1m, 5m, 1h, 4h, 1d) with 15+ technical indicators, mean reversion detection, and pattern recognition
2. **StrategyTrainerAgent**: Dynamic strategy weighting using performance-based grading and Bayesian optimization
3. **ARCH_CTRL**: Emotional state management and trade gating based on fear, stress, confidence metrics
4. **TradeAnalyzerAgent**: Performance tracking, P&L analysis, and trade logging
5. **RiskSentinel**: Portfolio risk management with VaR calculation, correlation monitoring, and position limits
6. **ExecutionDaemon**: Order execution with slippage protection and position management
7. **TradingOracleEngine**: Meta-analysis and strategic directives (Bayesian enhancement)
8. **ImaginationEngine**: Counterfactual scenario generation for strategy stress-testing

**Design Pattern**: Each agent implements `update(data: Dict) -> Dict` interface and declares data interests for efficient routing.

**Pros**: Clear separation of concerns, easy to add/remove agents, fault-isolated
**Cons**: Inter-agent dependencies must be carefully managed

### Market Data Pipeline

**Problem Addressed**: Need continuous, multi-source market data with minimal latency and fallback mechanisms.

**Solution**: 
- **Primary Scanner**: MomentumScanner with configurable timeframes and 500+ symbols/min throughput
- **Parallel Exchange Scanner**: Concurrent data collection from multiple exchanges (Binance, Coinbase, Kraken) with connection pooling and health monitoring
- **YFinance Adapter**: Forex data integration for cross-market analysis
- **Data Persistence**: DataPersistenceManager for ML/RL training with historical data storage

**Architecture**: Producer-consumer pattern with async data ingestion → SyncBus distribution → agent consumption

**Pros**: High throughput, multi-source redundancy, continuous data flow
**Cons**: Requires careful rate limit management, potential for data staleness

### Strategy Optimization Layer

**Problem Addressed**: Strategies need continuous adaptation to changing market conditions without manual intervention.

**Solution**: Three-tier optimization system:

1. **Bayesian Optimization** (MirrorOptimizerAgent): Parameter tuning using Gaussian Processes with safety constraints
2. **Bayesian Belief Tracking** (BayesianOracleEnhancement): Strategy confidence scoring based on historical success/failure with regime-specific decay
3. **Reinforcement Learning** (RLTradingAgent): PPO/SAC agents trained on live market data with custom reward shaping

**Safety Mechanisms**: 
- Parameter change limits (max 20% per iteration)
- Minimum evaluation trades (10+) before changes
- Rollback on failure
- Validation periods after parameter updates

**Pros**: Adaptive to market regimes, data-driven optimization, safety-first design
**Cons**: Requires significant historical data, potential for overfitting

### Risk Management System

**Problem Addressed**: Prevent catastrophic losses through multi-layered risk controls and emergency shutdown mechanisms.

**Solution**: 
- **RiskSentinel**: Real-time monitoring of drawdown (15% max), position sizing (10% max per position), portfolio volatility (25% max), correlation (80% max)
- **Emergency Controls**: Kill-switch with <1s response time, circuit breakers, automatic position flattening
- **Audit Logger**: Immutable audit trail with checksums for all trading activities
- **Stress Testing**: Monte Carlo scenario generation (1000+ scenarios) for strategy robustness

**Trigger Conditions**: Drawdown threshold, latency spikes (>500ms), API errors, position limits, correlation risk

**Pros**: Multiple safety layers, fast emergency response, comprehensive logging
**Cons**: May be overly conservative in some market conditions

### Web Dashboard & Monitoring

**Problem Addressed**: Need real-time visibility into multi-agent system state and ability to intervene during live trading.

**Solution**:
- **Frontend**: React/TypeScript dashboard with real-time charts, agent status, P&L tracking
- **Backend**: FastAPI server with WebSocket support (Socket.IO)
- **Visualization**: Recharts for performance metrics, Lightweight Charts for price data
- **Control Interface**: Command injection through SyncBus (pause/resume agents, adjust parameters)

**Data Flow**: SyncBus → WebSocket → React state → UI components (50ms update cycle)

**Pros**: Real-time monitoring, responsive UI, remote control capability
**Cons**: Additional system overhead, potential security concerns for remote access

### Configuration Management

**Problem Addressed**: System configuration spread across multiple files with no centralized management or environment-specific settings.

**Solution**:
- **ConfigManager**: Centralized configuration with dataclass-based structure
- **Secrets Management**: Environment variable-based API key storage with optional encryption (Fernet)
- **Dynamic Config**: Runtime parameter adjustment without system restart
- **Multi-Environment**: Support for development, staging, production configurations

**Pros**: Single source of truth, secure credential handling, environment flexibility
**Cons**: Requires migration of hardcoded values

### Advanced Features (Enhancement Layer)

**Imagination Engine**: Generates synthetic future market scenarios (20+ scenarios, 50+ steps) to stress-test strategies against "what-if" conditions including volatility spikes, trend reversals, and liquidity crises.

**Oracle System**: Meta-analysis layer that synthesizes signals from all agents to produce high-level trading directives with confidence scoring and regime-specific recommendations.

**Parallel Scanner**: Multi-exchange data collection with intelligent rate limiting, connection pooling, and circuit breakers for failed exchanges.

**Vectorized Backtesting**: NumPy/Numba-accelerated backtesting engine with realistic fee modeling (0.075%), slippage simulation (0.05%), and regime-specific performance analysis.

## External Dependencies

### Third-Party Services & APIs

- **Cryptocurrency Exchanges**: 
  - CCXT library for unified exchange API (Binance, Coinbase, Kraken)
  - WebSocket connections for real-time data
  - REST fallback for reliability
  
- **Market Data Providers**:
  - YFinance for forex data
  - Exchange native APIs for crypto
  - Historical data persistence for training

- **Machine Learning & Optimization**:
  - Stable-Baselines3 for RL agents (PPO, SAC, A2C)
  - Bayesian-Optimization for hyperparameter tuning
  - XGBoost for predictive models
  - Scikit-learn for preprocessing and metrics

### Key Python Libraries

**Trading & Market Data**:
- `ccxt` (v4.0.0+): Exchange connectivity
- `ta` (v0.10.0+): Technical analysis indicators
- `pandas` (v1.3.0+): Data manipulation
- `numpy` (v1.21.0+): Numerical computing

**Machine Learning**:
- `stable-baselines3` (v2.0.0+): Reinforcement learning
- `gymnasium` (v0.29.0+): RL environment interface
- `bayesian-optimization` (v1.4.0+): Hyperparameter optimization
- `scikit-learn` (v0.24.0+): ML utilities

**Web & API**:
- `fastapi`: REST API backend
- `flask` + `flask-socketio` (v5.0.0+): WebSocket dashboard
- `eventlet` (v0.31.0+): Async server
- `aiohttp` (v3.7.4+): Async HTTP client

**Visualization**:
- `plotly` (v5.3.0+): Interactive charts
- `matplotlib` (v3.4.0+): Static plotting
- React + Recharts (frontend): Real-time dashboard

**Data & Security**:
- `cryptography`: Secrets encryption
- `pydantic` (v1.8.2+): Data validation
- `psutil` (v7.1.0+): System monitoring

### Database & Storage

**Current**: File-based storage (CSV, JSON) for scan results, trades, and configurations

**Architecture Note**: The system uses Drizzle ORM patterns in some code but does not currently require PostgreSQL. Future database integration would use:
- PostgreSQL for relational data (trades, performance metrics)
- TimescaleDB for time-series market data
- Redis for caching and session management

### Frontend Stack

- **Framework**: React 18.2.0 with TypeScript
- **Build Tool**: Vite 5.0.8
- **Styling**: TailwindCSS 3.3.6
- **Charts**: Recharts 2.10.3, Lightweight Charts
- **Icons**: Lucide React 0.300.0
- **Routing**: React Router DOM 6.30.1
- **WebSocket**: Socket.IO Client 4.6.0

### Infrastructure Requirements

- **Runtime**: Python 3.9+ with asyncio support
- **Memory**: Minimum 4GB RAM (8GB+ recommended for RL training)
- **CPU**: Multi-core processor for parallel scanning
- **Network**: Stable connection with <100ms latency to exchanges
- **Storage**: 10GB+ for historical data and model checkpoints