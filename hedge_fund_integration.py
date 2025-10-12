
"""
MirrorCore-X 3.0 Hedge Fund Integration
Connects DeFo layer with existing MirrorCore agents
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from hedge_fund_core import (
    VaultConfig, DeFoOrchestrator, HedgeFundManager
)
from defi_execution_layer import DeFiRouter, VaultExecutor, CapitalAllocationManager

logger = logging.getLogger(__name__)


class MirrorCoreHedgeFundBridge:
    """Bridges MirrorCore-X signals to hedge fund operations"""
    
    def __init__(self, sync_bus, defo_orchestrator: DeFoOrchestrator):
        self.sync_bus = sync_bus
        self.defo = defo_orchestrator
        
        # Execution components
        self.defi_router = DeFiRouter()
        self.capital_allocator = CapitalAllocationManager()
        
        # Vault executors
        self.vault_executors = {
            name: VaultExecutor(connector, self.defi_router)
            for name, connector in defo_orchestrator.vault_connectors.items()
        }
    
    async def update(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Main update cycle - bridge MirrorCore to DeFo"""
        
        # 1. Get MirrorCore signals
        scanner_data = await self.sync_bus.get_state('scanner_data') or []
        rl_signals = await self.sync_bus.get_state('rl_signals') or []
        meta_signals = await self.sync_bus.get_state('meta_controller_signals') or []
        
        # 2. Combine signals
        combined_signals = self._combine_signals(scanner_data, rl_signals, meta_signals)
        
        # 3. Update hedge fund layer
        fund_update = await self.defo.update({'trading_signals': combined_signals})
        
        # 4. Optimize capital allocation
        strategy_perf = await self.sync_bus.get_state('strategy_performance') or {}
        allocations = await self.capital_allocator.optimize_allocations(
            fund_update['aggregate'],
            strategy_perf
        )
        
        # 5. Execute rebalancing
        execution_results = await self._execute_rebalancing(
            fund_update['individual_funds'],
            allocations
        )
        
        # 6. Update state
        await self.sync_bus.update_state('hedge_fund_allocations', {
            k: float(v) for k, v in allocations.items()
        })
        await self.sync_bus.update_state('hedge_fund_executions', execution_results)
        
        return {
            'fund_update': fund_update,
            'allocations': allocations,
            'executions': execution_results
        }
    
    def _combine_signals(self, scanner, rl, meta) -> list:
        """Combine signals from different MirrorCore agents"""
        
        signals_map = {}
        
        # Add scanner signals
        for signal in scanner:
            symbol = signal.get('symbol')
            if symbol:
                signals_map[symbol] = signal
        
        # Enhance with RL
        for signal in rl:
            symbol = signal.get('symbol')
            if symbol and symbol in signals_map:
                signals_map[symbol]['rl_position'] = signal.get('position', 0)
        
        # Enhance with meta controller
        for signal in meta:
            symbol = signal.get('symbol')
            if symbol and symbol in signals_map:
                signals_map[symbol]['final_position'] = signal.get('final_position', 0)
                signals_map[symbol]['confidence'] = signal.get('confidence', 0.5)
        
        return list(signals_map.values())
    
    async def _execute_rebalancing(self, funds: Dict, allocations: Dict) -> list:
        """Execute rebalancing trades across vaults"""
        
        execution_results = []
        
        for fund_name, fund_data in funds.items():
            rebalance_actions = fund_data.get('rebalance_actions', [])
            
            if fund_name in self.vault_executors:
                executor = self.vault_executors[fund_name]
                
                for action in rebalance_actions:
                    # Execute trade
                    result = await executor.execute_trade(action)
                    execution_results.append(result)
        
        return execution_results


# Integration function
async def add_hedge_fund_to_mirrorcore(
    sync_bus,
    vault_address: str = "0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb0",  # maonovault
    rpc_url: str = ""
) -> MirrorCoreHedgeFundBridge:
    """Add hedge fund layer to MirrorCore-X system"""
    
    # Create vault config
    vault_config = VaultConfig(
        vault_address=vault_address,
        vault_name="maonovault",
        rpc_url=rpc_url or "https://eth-mainnet.g.alchemy.com/v2/YOUR_KEY"
    )
    
    # Create DeFo orchestrator
    defo = DeFoOrchestrator([vault_config], sync_bus)
    
    # Create bridge
    bridge = MirrorCoreHedgeFundBridge(sync_bus, defo)
    
    # Register with sync bus
    await sync_bus.register_agent('hedge_fund_bridge', bridge)
    
    logger.info("✅ Hedge fund layer integrated: maonovault connected")
    
    return bridge


# Standalone mode
async def run_standalone_hedge_fund():
    """Run hedge fund layer in standalone mode"""
    
    from mirrorcore_x import create_mirrorcore_system
    
    # Create full MirrorCore system
    sync_bus, components = await create_mirrorcore_system(
        dry_run=False,
        use_testnet=False,
        enable_oracle=True,
        enable_bayesian=True,
        enable_advanced_strategies=True
    )
    
    # Add hedge fund layer
    hedge_fund = await add_hedge_fund_to_mirrorcore(
        sync_bus,
        vault_address="0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb0"
    )
    
    # Run system
    logger.info("🚀 MirrorCore-X 3.0 Hedge Fund Layer running...")
    
    while True:
        await sync_bus.tick()
        await asyncio.sleep(1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_standalone_hedge_fund())
