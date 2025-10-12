
"""
Parallel Exchange Scanner Integration
Connects parallel scanner to main MirrorCore-X system
"""

import asyncio
import logging
from typing import Dict, Any, List
from parallel_exchange_scanner import ParallelExchangeScanner, ExchangeConfig
import pandas as pd

logger = logging.getLogger(__name__)

class ParallelScannerIntegration:
    """Integrates parallel exchange scanner with MirrorCore-X"""
    
    def __init__(self, sync_bus, scanner):
        self.sync_bus = sync_bus
        self.primary_scanner = scanner
        self.parallel_scanner = None
        self.is_enabled = False
        
    async def initialize(self, exchange_configs: List[ExchangeConfig]):
        """Initialize parallel scanner with multiple exchanges"""
        try:
            self.parallel_scanner = ParallelExchangeScanner(
                exchange_configs=exchange_configs,
                quote_currency='USDT',
                min_volume_usd=500000,
                max_concurrent_per_exchange=3,
                use_per_worker_loop=True
            )
            self.is_enabled = True
            logger.info("✅ Parallel Exchange Scanner initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize parallel scanner: {e}")
            return False
    
    async def scan_and_update(self, symbols: List[str] = None) -> pd.DataFrame:
        """Scan all exchanges and update sync bus"""
        if not self.is_enabled or not self.parallel_scanner:
            logger.warning("Parallel scanner not enabled")
            return pd.DataFrame()
        
        try:
            # Scan all exchanges in parallel
            results = await self.parallel_scanner.scan_all_exchanges(
                symbols=symbols,
                prioritize_by_health=True
            )
            
            # Update sync bus with results
            if not results.empty:
                scanner_data = results.to_dict('records')
                await self.sync_bus.update_state('parallel_scanner_data', scanner_data)
                
                # Merge with primary scanner data
                primary_data = await self.sync_bus.get_state('scanner_data') or []
                merged_data = self._merge_scanner_data(primary_data, scanner_data)
                await self.sync_bus.update_state('scanner_data', merged_data)
                
                logger.info(f"Parallel scan: {len(results)} opportunities across exchanges")
            
            return results
            
        except Exception as e:
            logger.error(f"Parallel scan failed: {e}")
            return pd.DataFrame()
    
    def _merge_scanner_data(self, primary: List[Dict], parallel: List[Dict]) -> List[Dict]:
        """Merge data from primary and parallel scanners"""
        # Create lookup for primary data
        primary_symbols = {d['symbol']: d for d in primary}
        
        merged = list(primary)
        
        # Add parallel data that's not in primary
        for p_data in parallel:
            symbol = p_data['symbol']
            if symbol not in primary_symbols:
                merged.append(p_data)
            else:
                # Update with better volume data from parallel scan
                if p_data.get('volume_24h_usd', 0) > primary_symbols[symbol].get('average_volume_usd', 0):
                    primary_symbols[symbol]['average_volume_usd'] = p_data['volume_24h_usd']
                    primary_symbols[symbol]['parallel_exchange'] = p_data.get('exchange')
        
        return merged
    
    async def get_health_report(self) -> Dict[str, Any]:
        """Get health status of all exchanges"""
        if not self.parallel_scanner:
            return {}
        
        return self.parallel_scanner.get_health_report()
    
    async def close(self):
        """Close all exchange connections"""
        if self.parallel_scanner:
            await self.parallel_scanner.close()


async def add_parallel_scanner_to_mirrorcore(sync_bus, scanner, enable: bool = True):
    """
    Add parallel exchange scanner to MirrorCore-X
    
    Usage:
        parallel_integration = await add_parallel_scanner_to_mirrorcore(sync_bus, scanner)
        results = await parallel_integration.scan_and_update()
    """
    
    if not enable:
        return None
    
    # Configure exchanges
    configs = [
        ExchangeConfig('binance', priority=3, rate_limit_per_second=10),
        ExchangeConfig('coinbase', priority=2, rate_limit_per_second=5),
        ExchangeConfig('kraken', priority=2, rate_limit_per_second=3),
    ]
    
    integration = ParallelScannerIntegration(sync_bus, scanner)
    success = await integration.initialize(configs)
    
    if success:
        logger.info("✨ Parallel scanner integrated into MirrorCore-X")
        return integration
    else:
        logger.warning("⚠️ Parallel scanner integration failed")
        return None
