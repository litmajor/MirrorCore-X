// marketDataProvider.ts
export interface OHLCV {
  timestamp: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}


// --- Hybrid MarketFrame: Combines advanced analytics fields from temporal_precognition_engine and v2 ---
interface HybridMarketFrame {
  timestamp: number;
  symbol: string;
  price: {
    open: number;
    high: number;
    low: number;
    close: number;
  };
  volume: number;
  volatility: number;
  momentum_score: number;
  rsi: number;
  trend_score: number;
  confidence_score: number;
  composite_score: number;
  volume_score: number;
  anchor_index: number;
  price_range: number;
  predicted_return: number;
  predicted_consistent: number;
  indicators: {
    macd: { line: number; signal: number; histogram: number };
    bb: { upper: number; middle: number; lower: number };
    ema20: number;
    ema50: number;
    vwap: number;
    atr: number;
  };
  orderFlow: {
    bidVolume: number;
    askVolume: number;
    netFlow: number;
    largeOrders: number;
    smallOrders: number;
  };
  marketMicrostructure: {
    spread: number;
    depth: number;
    imbalance: number;
    toxicity: number;
  };
  temporal_ghost: {
    next_moves: Array<{
      timestamp: number;
      predicted_price: number;
      confidence: number;
      probability: number;
      pattern_match: number;
    }>;
    certainty_river: number;
    time_distortion: number;
    pattern_resonance: number;
    quantum_coherence: number;
    fractal_depth: number;
  };
  intention_field: {
    accumulation_pressure: number;
    breakout_membrane: number;
    momentum_vector: {
      direction: 'up' | 'down' | 'sideways';
      strength: number;
      timing: number;
      acceleration: number;
    };
    whale_presence: number;
    institutional_flow: number;
    retail_sentiment: number;
    liquidity_depth: number;
  };
  power_level: {
    edge_percentage: number;
    opportunity_intensity: number;
    risk_shadow: number;
    profit_magnetism: number;
    certainty_coefficient: number;
    godmode_factor: number;
    quantum_advantage: number;
    reality_bend_strength: number;
  };
  market_layers: {
    fear_greed_index: number;
    volatility_regime: 'low' | 'medium' | 'high' | 'extreme';
    trend_strength: number;
    support_resistance: {
      nearest_support: number;
      nearest_resistance: number;
      strength: number;
    };
    market_phase: 'accumulation' | 'markup' | 'distribution' | 'markdown';
  };
  consciousness_matrix: {
    awareness_level: number;
    collective_intelligence: number;
    prediction_accuracy: number;
    reality_stability: number;
    temporal_variance: number;
  };
}

export async function fetchHybridMarketFrames(timeframe: string = '1d') {
  const res = await fetch(`http://localhost:8000/frames/${timeframe}`);
  if (!res.ok) throw new Error('Failed to fetch market data');
  return res.json();
}