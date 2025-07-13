
# MirrorCore-X: Cognitive Trading Organism

MirrorCore-X is a modular, self-aware, emotionally gated trading simulation and research platform for algorithmic trading, agent-based modeling, and cognitive finance. It features a multi-agent architecture, emotional engine, live dashboard visualization, DRY_RUN simulation mode, full strategy training, trade analysis, and real-time sentiment integration.

## Key Features

- **Multi-Agent Architecture:** Includes MirrorAgent, ARCH_CTRL, EgoProcessor, FearAnalyzer, SelfAwarenessAgent, DecisionMirror, ExecutionDaemon, ReflectionCore, RiskSentinel, StrategyTrainerAgent, TradeAnalyzerAgent, SentimentScanAgent, and more.
- **Emotional Engine (ARCH_CTRL):** Gating of trading actions and emotional state updates per tick.
- **Self-Narration & Introspection:** Agents narrate decisions and introspect on behavioral drift, trust, and emotional volatility.
- **Live Dashboard Visualization:** Real-time PnL, emotional metrics, trade logs, and session grades via matplotlib and Flask web dashboard.
- **DRY_RUN Mode:** Safe simulation with virtual portfolio, simulated trades, and live PnL tracking. No real trades are placed unless DRY_RUN is disabled.
- **Strategy Training:** StrategyTrainerAgent evaluates, selects, and grades strategies per tick, with performance feedback and grading.
- **Trade Analysis:** TradeAnalyzerAgent records trades, prints summaries every 10 ticks, and supports deep trade analytics.
- **Sentiment Analysis:** SentimentScanAgent integrates with real APIs (Binance, Twitter, CryptoPanic, Deribit, Reddit) for funding, social, news, options, and crypto sentiment. Twitter bearer token and other API keys are supported.
- **Web Server Integration:** `web_server.py` runs a Flask dashboard, collects live agent states, and supports DRY_RUN mode for safe visualization.
- **Comprehensive Documentation:** README and requirements.txt for onboarding and customization.

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure API keys:**
   - Edit `sentiment_analyzer_agent.py` to add your Twitter bearer token and other API credentials as needed.

3. **Run the main agent simulation:**
   ```bash
   python mirrorcore_x.py
   ```

4. **Run the web dashboard server:**
   ```bash
   python web_server.py
   ```

## Main Components

- `mirrorcore_x.py`: Main agent system, tick loop, dashboard visualization, agent state extraction, narration logic, DRY_RUN agent loop, virtual portfolio, strategy trainer integration, TradeAnalyzerAgent integration, SentimentScanAgent with real APIs.
- `arch_ctrl.py`: Emotional engine, gating, override logic, advanced emotional throttling.
- `momentum_scanner.py`: Market scanning, signal filtering, strong signal extraction.
- `strategy_trainer_agent.py`: StrategyTrainerAgent, wrapper agents, strategy registration, evaluation, performance tracking, grading.
- `trade_analyzer_agent.py`: TradeAnalyzerAgent, trade recording, summary reporting, hooks for deep analysis.
- `sentiment_analyzer_agent.py`: SentimentScanAgent, real API integration for funding, social, news, options, crypto-specific sentiment. Configure API keys/tokens here.
- `web_server.py`: Flask app, SocketIO, eventlet, simulation background task, dashboard data collection, DRY_RUN integration.
- `README.md`, `requirements.txt`: Project overview and dependencies.

## DRY_RUN Mode

Set `DRY_RUN = True` in `ExecutionDaemon` (default) for safe simulation. All trades are simulated, and a virtual portfolio is tracked. Trade logs and PnL are printed live and visualized in the dashboard. Set to `False` to enable live trading (requires exchange credentials).

## Sentiment Analysis

SentimentScanAgent integrates with real APIs:
- **Binance:** Funding rates
- **Twitter:** Social sentiment (requires bearer token)
- **CryptoPanic:** News sentiment
- **Deribit:** Options sentiment
- **Reddit:** Crypto-specific sentiment
Configure API keys/tokens in `sentiment_analyzer_agent.py` as needed.

## Strategy Training & Analysis

StrategyTrainerAgent evaluates registered strategies per tick, selects the best, and updates performance. TradeAnalyzerAgent records trades and prints summaries every 10 ticks. Grades and insights are printed and visualized.

## Dashboard & Visualization

Live dashboard visualizes agent states, emotional metrics, PnL, trade logs, and session grades. Run `web_server.py` for web dashboard. Matplotlib plots are shown at the end of each session.

## Usage

1. Edit `requirements.txt` and install dependencies
2. Configure API keys in `sentiment_analyzer_agent.py`
3. Run `mirrorcore_x.py` for simulation
4. Run `web_server.py` for dashboard
5. Review live output and dashboard for agent states, trades, and performance

## Customization

Extend agents, add new strategies, or integrate additional APIs as needed. The modular architecture supports rapid experimentation and research. All agent classes are designed for easy extension.

## License

MIT License

## License

MIT License

## Authors

- Architect: James Kagua

---

For questions, enhancements, or contributions, open an issue or contact the project owner.
