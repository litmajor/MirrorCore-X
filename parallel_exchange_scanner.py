
"""
Parallel Exchange Scanner - Optimized Multi-Exchange Market Data Collection

Features:
- Parallel scanning across multiple exchanges
- Connection pooling and reuse
- Intelligent rate limit management
- Data deduplication and normalization
- Adaptive scheduling based on exchange performance
- Circuit breakers for failed exchanges
"""

import asyncio
import ccxt.async_support as ccxt
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
import time
import logging
from datetime import datetime
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class ExchangeConfig:
    """Configuration for each exchange"""
    exchange_id: str
    enabled: bool = True
    rate_limit_per_second: float = 10.0
    max_retries: int = 3
    timeout_seconds: int = 30
    priority: int = 1  # Higher priority = scanned first
    api_key: Optional[str] = None
    api_secret: Optional[str] = None

@dataclass
class ExchangeHealth:
    """Track exchange health metrics"""
    exchange_id: str
    success_count: int = 0
    failure_count: int = 0
    avg_latency_ms: float = 0.0
    last_success: float = 0.0
    last_failure: float = 0.0
    circuit_open: bool = False
    
    @property
    def health_score(self) -> float:
        """Calculate health score 0-1"""
        total = self.success_count + self.failure_count
        if total == 0:
            return 1.0
        return self.success_count / total

@dataclass
class ScanResult:
    """Normalized scan result from any exchange"""
    symbol: str
    exchange: str
    price: float
    volume_24h: float
    high_24h: float
    low_24h: float
    timestamp: float
    bid: Optional[float] = None
    ask: Optional[float] = None
    spread_pct: Optional[float] = None
    raw_data: Dict = field(default_factory=dict)

class ConnectionPool:
    """Manages exchange connections with pooling"""
    
    def __init__(self, max_connections_per_exchange: int = 5):
        self.pools: Dict[str, List[ccxt.Exchange]] = defaultdict(list)
        self.in_use: Dict[str, Set[ccxt.Exchange]] = defaultdict(set)
        self.max_connections = max_connections_per_exchange
        self.lock = asyncio.Lock()
    
    async def acquire(self, exchange_id: str, config: ExchangeConfig) -> ccxt.Exchange:
        """Acquire an exchange connection from pool"""
        async with self.lock:
            # Try to get from pool
            if self.pools[exchange_id]:
                exchange = self.pools[exchange_id].pop()
                self.in_use[exchange_id].add(exchange)
                return exchange
            
            # Create new if under limit
            if len(self.in_use[exchange_id]) < self.max_connections:
                exchange = self._create_exchange(exchange_id, config)
                self.in_use[exchange_id].add(exchange)
                return exchange
            
            # Wait for available connection
            while not self.pools[exchange_id]:
                await asyncio.sleep(0.1)
            
            exchange = self.pools[exchange_id].pop()
            self.in_use[exchange_id].add(exchange)
            return exchange
    
    async def release(self, exchange_id: str, exchange: ccxt.Exchange):
        """Release connection back to pool"""
        async with self.lock:
            if exchange in self.in_use[exchange_id]:
                self.in_use[exchange_id].remove(exchange)
                self.pools[exchange_id].append(exchange)
    
    def _create_exchange(self, exchange_id: str, config: ExchangeConfig) -> ccxt.Exchange:
        """Create new exchange instance"""
        exchange_class = getattr(ccxt, exchange_id)
        exchange_config = {
            'enableRateLimit': True,
            'timeout': config.timeout_seconds * 1000,
        }
        
        if config.api_key and config.api_secret:
            exchange_config['apiKey'] = config.api_key
            exchange_config['secret'] = config.api_secret
        
        return exchange_class(exchange_config)
    
    async def close_all(self):
        """Close all connections"""
        for exchange_id in list(self.pools.keys()):
            for exchange in self.pools[exchange_id]:
                try:
                    await exchange.close()
                except Exception as e:
                    logger.warning(f"Error closing {exchange_id}: {e}")
            
            for exchange in self.in_use[exchange_id]:
                try:
                    await exchange.close()
                except Exception as e:
                    logger.warning(f"Error closing {exchange_id}: {e}")

class ParallelExchangeScanner:
    """
    Optimized parallel scanner for multiple exchanges
    """
    
    def __init__(self, 
                 exchange_configs: List[ExchangeConfig],
                 quote_currency: str = 'USDT',
                 min_volume_usd: float = 100000,
                 max_concurrent_per_exchange: int = 3,
                 use_per_worker_loop: bool = False):
        
        self.configs = {cfg.exchange_id: cfg for cfg in exchange_configs}
        self.quote_currency = quote_currency
        self.min_volume_usd = min_volume_usd
        self.max_concurrent = max_concurrent_per_exchange
        
        # Connection pooling
        self.connection_pool = ConnectionPool(max_connections_per_exchange=5)
        
        # Health tracking
        self.health: Dict[str, ExchangeHealth] = {
            cfg.exchange_id: ExchangeHealth(cfg.exchange_id)
            for cfg in exchange_configs
        }
        
        # Rate limiting
        self.rate_limiters: Dict[str, asyncio.Semaphore] = {}
        self.last_request_time: Dict[str, float] = defaultdict(float)
        
        # Deduplication
        self.symbol_cache: Dict[str, Set[str]] = defaultdict(set)
        self.price_cache: Dict[Tuple[str, str], ScanResult] = {}
        
        # Results
        self.latest_results: List[ScanResult] = []
        
        # Initialize rate limiters
        for cfg in exchange_configs:
            self.rate_limiters[cfg.exchange_id] = asyncio.Semaphore(
                int(cfg.rate_limit_per_second)
            )

        # Optional per-exchange worker loops for persistent exchanges
        self.use_per_worker_loop = use_per_worker_loop
        self.workers: Dict[str, Dict] = {}
        if self.use_per_worker_loop:
            # Start a worker loop and persistent exchange for each configured exchange
            for cfg in exchange_configs:
                self._start_worker(cfg.exchange_id, cfg)
    
    async def scan_all_exchanges(self, 
                                 symbols: Optional[List[str]] = None,
                                 prioritize_by_health: bool = True) -> pd.DataFrame:
        """
        Scan all exchanges in parallel with optimization
        """
        start_time = time.time()
        
        # Sort exchanges by priority and health
        sorted_exchanges = self._get_sorted_exchanges(prioritize_by_health)
        
        # Create scanning tasks
        tasks = []
        for exchange_id in sorted_exchanges:
            if not self.health[exchange_id].circuit_open:
                task = asyncio.create_task(
                    self._scan_exchange(exchange_id, symbols),
                    name=f"scan_{exchange_id}"
                )
                tasks.append(task)
        
        # Execute all scans in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        all_scan_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Scan failed: {result}")
            elif result:
                all_scan_results.extend(result)
        
        # Deduplicate and normalize
        deduplicated = self._deduplicate_results(all_scan_results)
        
        # Convert to DataFrame
        df = self._results_to_dataframe(deduplicated)
        
        elapsed = time.time() - start_time
        logger.info(f"Scanned {len(sorted_exchanges)} exchanges in {elapsed:.2f}s, "
                   f"found {len(df)} unique opportunities")
        
        self.latest_results = deduplicated
        return df
    
    async def _scan_exchange(self, 
                            exchange_id: str, 
                            symbols: Optional[List[str]] = None) -> List[ScanResult]:
        """
        Scan a single exchange with rate limiting and error handling
        """
        config = self.configs[exchange_id]
        health = self.health[exchange_id]
        results = []
        
        # If per-worker loop is enabled and a persistent worker exists, delegate
        if self.use_per_worker_loop and exchange_id in self.workers:
            # Schedule the worker coroutine on the worker loop and return its result
            worker = self.workers[exchange_id]
            cf = asyncio.run_coroutine_threadsafe(self._worker_scan_exchange(exchange_id, symbols), worker['loop'])
            # Wrap concurrent.futures.Future into asyncio.Future for awaiting
            return await asyncio.wrap_future(cf)

        exchange = None
        try:
            # Acquire connection from pool
            exchange = await self.connection_pool.acquire(exchange_id, config)
            
            # Load markets if needed
            if not exchange.markets:
                await exchange.load_markets()
            
            # Get symbols to scan
            scan_symbols = symbols or self._get_exchange_symbols(exchange)
            
            # Scan with concurrency control
            semaphore = asyncio.Semaphore(self.max_concurrent)
            
            async def scan_symbol(symbol: str):
                async with semaphore:
                    return await self._fetch_ticker_with_retry(
                        exchange, exchange_id, symbol
                    )
            
            # Execute concurrent symbol scans
            ticker_tasks = [scan_symbol(sym) for sym in scan_symbols]
            ticker_results = await asyncio.gather(*ticker_tasks, return_exceptions=True)
            
            # Process ticker results
            for ticker_result in ticker_results:
                if isinstance(ticker_result, ScanResult):
                    results.append(ticker_result)
                elif isinstance(ticker_result, Exception):
                    logger.debug(f"Ticker fetch failed: {ticker_result}")
            
            # Update health on success
            health.success_count += 1
            health.last_success = time.time()
            
        except Exception as e:
            logger.error(f"Exchange {exchange_id} scan failed: {e}")
            health.failure_count += 1
            health.last_failure = time.time()
            
            # Open circuit breaker if too many failures
            if health.health_score < 0.3:
                health.circuit_open = True
                logger.warning(f"Circuit breaker opened for {exchange_id}")
                asyncio.create_task(self._reset_circuit_breaker(exchange_id, delay=60))
        
        finally:
            # Release connection back to pool
            if exchange:
                await self.connection_pool.release(exchange_id, exchange)
        
        return results
    
    async def _fetch_ticker_with_retry(self, 
                                      exchange: ccxt.Exchange,
                                      exchange_id: str,
                                      symbol: str) -> Optional[ScanResult]:
        """
        Fetch ticker with rate limiting and retries
        """
        config = self.configs[exchange_id]
        
        for attempt in range(config.max_retries):
            try:
                # Rate limiting
                await self._wait_for_rate_limit(exchange_id)
                
                # Fetch ticker
                start = time.time()
                ticker = await exchange.fetch_ticker(symbol)
                latency = (time.time() - start) * 1000
                
                # Update latency metric
                health = self.health[exchange_id]
                health.avg_latency_ms = (health.avg_latency_ms * 0.9 + latency * 0.1)
                
                # Validate and normalize
                if ticker and ticker.get('last'):
                    return self._normalize_ticker(exchange_id, symbol, ticker)
                
            except Exception as e:
                if attempt == config.max_retries - 1:
                    logger.debug(f"Ticker fetch failed for {symbol} on {exchange_id}: {e}")
                else:
                    await asyncio.sleep(0.5 * (attempt + 1))
        
        return None

    def _start_worker(self, exchange_id: str, config: ExchangeConfig):
        """Start a dedicated thread with an asyncio loop and persistent exchange instance."""
        import threading
        import concurrent.futures

        loop = asyncio.new_event_loop()

        def _run_loop():
            try:
                asyncio.set_event_loop(loop)
                loop.run_forever()
            finally:
                try:
                    loop.close()
                except Exception:
                    pass

        thread = threading.Thread(target=_run_loop, daemon=True)
        thread.start()

        # Initialize the exchange instance inside the worker loop
        async def _init_exchange():
            exchange_class = getattr(ccxt, exchange_id)
            exchange_conf = {
                'enableRateLimit': True,
                'timeout': config.timeout_seconds * 1000,
            }
            if config.api_key and config.api_secret:
                exchange_conf['apiKey'] = config.api_key
                exchange_conf['secret'] = config.api_secret

            exch = exchange_class(exchange_conf)
            try:
                await exch.load_markets()
            except Exception:
                # Not fatal at init; worker will attempt loads later
                pass
            return exch

        cf = asyncio.run_coroutine_threadsafe(_init_exchange(), loop)
        exch = cf.result()

        self.workers[exchange_id] = {
            'loop': loop,
            'thread': thread,
            'exchange': exch,
            'config': config
        }

    async def _worker_scan_exchange(self, exchange_id: str, symbols: Optional[List[str]] = None) -> List[ScanResult]:
        """Coroutine that runs inside a worker loop and scans using the persistent exchange."""
        worker = self.workers.get(exchange_id)
        if not worker:
            return []

        exchange = worker['exchange']
        config = worker['config']
        health = self.health[exchange_id]
        results: List[ScanResult] = []

        try:
            # Ensure markets are loaded
            if not getattr(exchange, 'markets', None):
                try:
                    await exchange.load_markets()
                except Exception:
                    pass

            scan_symbols = symbols or self._get_exchange_symbols(exchange)

            semaphore = asyncio.Semaphore(self.max_concurrent)

            async def scan_symbol(sym: str):
                async with semaphore:
                    return await self._fetch_ticker_with_retry(exchange, exchange_id, sym)

            ticker_tasks = [scan_symbol(sym) for sym in scan_symbols]
            ticker_results = await asyncio.gather(*ticker_tasks, return_exceptions=True)

            for ticker_result in ticker_results:
                if isinstance(ticker_result, ScanResult):
                    results.append(ticker_result)

            health.success_count += 1
            health.last_success = time.time()

        except Exception as e:
            logger.error(f"Worker exchange {exchange_id} scan failed: {e}")
            health.failure_count += 1
            health.last_failure = time.time()

        return results
    
    async def _wait_for_rate_limit(self, exchange_id: str):
        """
        Enforce rate limiting per exchange
        """
        config = self.configs[exchange_id]
        min_interval = 1.0 / config.rate_limit_per_second
        
        now = time.time()
        elapsed = now - self.last_request_time[exchange_id]
        
        if elapsed < min_interval:
            await asyncio.sleep(min_interval - elapsed)
        
        self.last_request_time[exchange_id] = time.time()
    
    def _normalize_ticker(self, 
                         exchange_id: str, 
                         symbol: str, 
                         ticker: Dict) -> ScanResult:
        """
        Normalize ticker data across exchanges
        """
        price = ticker.get('last', 0)
        bid = ticker.get('bid')
        ask = ticker.get('ask')
        
        spread_pct = None
        if bid and ask and price > 0:
            spread_pct = ((ask - bid) / price) * 100
        
        return ScanResult(
            symbol=symbol,
            exchange=exchange_id,
            price=price,
            volume_24h=ticker.get('quoteVolume', 0),
            high_24h=ticker.get('high', price),
            low_24h=ticker.get('low', price),
            timestamp=ticker.get('timestamp', time.time()),
            bid=bid,
            ask=ask,
            spread_pct=spread_pct,
            raw_data=ticker
        )
    
    def _get_exchange_symbols(self, exchange: ccxt.Exchange) -> List[str]:
        """
        Get tradable symbols for exchange
        """
        symbols = []
        for symbol, market in exchange.markets.items():
            if (market.get('quote') == self.quote_currency and 
                market.get('active', True) and
                market.get('type') in ['spot', 'swap']):
                symbols.append(symbol)
        
        return symbols[:100]  # Limit for performance
    
    def _deduplicate_results(self, results: List[ScanResult]) -> List[ScanResult]:
        """
        Remove duplicate symbols, keeping best price/volume
        """
        symbol_best: Dict[str, ScanResult] = {}
        
        for result in results:
            base_symbol = result.symbol.split('/')[0] if '/' in result.symbol else result.symbol
            
            if base_symbol not in symbol_best:
                symbol_best[base_symbol] = result
            else:
                # Keep result with higher volume
                if result.volume_24h > symbol_best[base_symbol].volume_24h:
                    symbol_best[base_symbol] = result
        
        return list(symbol_best.values())
    
    def _results_to_dataframe(self, results: List[ScanResult]) -> pd.DataFrame:
        """
        Convert results to DataFrame with filtering
        """
        if not results:
            return pd.DataFrame()
        
        df = pd.DataFrame([
            {
                'symbol': r.symbol,
                'exchange': r.exchange,
                'price': r.price,
                'volume_24h_usd': r.volume_24h,
                'high_24h': r.high_24h,
                'low_24h': r.low_24h,
                'spread_pct': r.spread_pct,
                'timestamp': datetime.fromtimestamp(r.timestamp / 1000)
            }
            for r in results
        ])
        
        # Filter by volume
        df = df[df['volume_24h_usd'] >= self.min_volume_usd]
        
        # Calculate additional metrics
        df['price_range_pct'] = ((df['high_24h'] - df['low_24h']) / df['price']) * 100
        df['quality_score'] = (
            (df['volume_24h_usd'] / df['volume_24h_usd'].max()) * 0.6 +
            (1 - df['spread_pct'].fillna(1) / 10) * 0.4
        )
        
        return df.sort_values('quality_score', ascending=False)
    
    def _get_sorted_exchanges(self, by_health: bool = True) -> List[str]:
        """
        Get exchanges sorted by priority and health
        """
        exchanges = [
            (ex_id, cfg.priority, self.health[ex_id].health_score)
            for ex_id, cfg in self.configs.items()
            if cfg.enabled
        ]
        
        if by_health:
            # Sort by health score then priority
            exchanges.sort(key=lambda x: (x[2], x[1]), reverse=True)
        else:
            # Sort by priority only
            exchanges.sort(key=lambda x: x[1], reverse=True)
        
        return [ex[0] for ex in exchanges]
    
    async def _reset_circuit_breaker(self, exchange_id: str, delay: int = 60):
        """
        Reset circuit breaker after cooldown
        """
        await asyncio.sleep(delay)
        self.health[exchange_id].circuit_open = False
        logger.info(f"Circuit breaker reset for {exchange_id}")
    
    async def get_best_opportunities(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get best opportunities across all exchanges
        """
        if not self.latest_results:
            return pd.DataFrame()
        
        df = self._results_to_dataframe(self.latest_results)
        return df.head(top_n)
    
    def get_health_report(self) -> Dict[str, Dict]:
        """
        Get health report for all exchanges
        """
        return {
            ex_id: {
                'health_score': health.health_score,
                'success_count': health.success_count,
                'failure_count': health.failure_count,
                'avg_latency_ms': health.avg_latency_ms,
                'circuit_open': health.circuit_open
            }
            for ex_id, health in self.health.items()
        }
    
    async def close(self):
        """
        Close all connections
        """
        await self.connection_pool.close_all()
        # Close worker exchanges and stop their loops
        for ex_id, w in list(self.workers.items()):
            try:
                cf = asyncio.run_coroutine_threadsafe(w['exchange'].close(), w['loop'])
                cf.result(timeout=5)
            except Exception:
                pass
            try:
                w['loop'].call_soon_threadsafe(w['loop'].stop)
            except Exception:
                pass
            try:
                w['thread'].join(timeout=2)
            except Exception:
                pass


# Example usage
async def demo():
    """
    Demo parallel exchange scanning
    """
    
    # Configure exchanges
    configs = [
        ExchangeConfig('binance', priority=3, rate_limit_per_second=10),
        ExchangeConfig('coinbase', priority=2, rate_limit_per_second=5),
        ExchangeConfig('kraken', priority=2, rate_limit_per_second=3),
        ExchangeConfig('kucoin', priority=1, rate_limit_per_second=5),
    ]
    
    scanner = ParallelExchangeScanner(
        exchange_configs=configs,
        quote_currency='USDT',
        min_volume_usd=500000
    )
    
    try:
        # Scan all exchanges
        results = await scanner.scan_all_exchanges()
        
        print(f"\nFound {len(results)} opportunities across exchanges:")
        print(results.head(10))
        
        # Get best opportunities
        best = await scanner.get_best_opportunities(top_n=5)
        print("\nTop 5 opportunities:")
        print(best[['symbol', 'exchange', 'price', 'volume_24h_usd', 'quality_score']])
        
        # Health report
        health = scanner.get_health_report()
        print("\nExchange Health:")
        for ex, metrics in health.items():
            print(f"{ex}: {metrics['health_score']:.2%} health, "
                  f"{metrics['avg_latency_ms']:.0f}ms latency")
        
    finally:
        await scanner.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(demo())
