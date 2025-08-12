"""
Imagination Integration for MirrorCore-X

This module integrates the ImaginationEngine into the main system, layering counterfactual scenario synthesis and robustness scoring on top of existing agents. All core agents continue to operate as before, but now have access to synthetic scenario generation, stress-testing, and robustness evaluation.
"""
import asyncio
import logging
from imagination_engine import (
    ScenarioGenerator,
    CounterfactualSimulator,
    RobustnessScorer,
)

logger = logging.getLogger(__name__)

class ImaginationIntegration:
    """
    Wraps the main system with ImaginationEngine capabilities.
    Agents continue to operate as before, but can now:
    - Generate synthetic market scenarios
    - Simulate strategies against counterfactual futures
    - Score robustness and stress resistance
    """
    def __init__(self, sync_bus, strategy_trainer, execution_daemon, oracle_engine):
        self.sync_bus = sync_bus
        self.strategy_trainer = strategy_trainer
        self.execution_daemon = execution_daemon
        self.oracle_engine = oracle_engine
        self.scenario_generator = None
        self.simulator = None
        self.robustness_scorer = RobustnessScorer()
        self.last_scenarios = []
        self.last_results = {}
        logger.info("ImaginationIntegration initialized")

    async def initialize(self):
        # Get current market data from sync_bus
        market_data = await self.sync_bus.get_state('market_data') or []
        self.scenario_generator = ScenarioGenerator(self.oracle_engine, market_data)
        self.simulator = CounterfactualSimulator(self.strategy_trainer, self.execution_daemon)
        logger.info("ImaginationEngine components initialized")

    async def run_imagination_cycle(self, num_scenarios=20, scenario_length=50):
        """
        Generate scenarios, simulate strategies, and score robustness.
        """
        if self.scenario_generator is None or self.simulator is None:
            await self.initialize()
        if self.scenario_generator is None:
            raise RuntimeError("ScenarioGenerator is not initialized.")
        if self.simulator is None:
            raise RuntimeError("CounterfactualSimulator is not initialized.")
        scenarios = self.scenario_generator.generate_scenarios(num_scenarios, scenario_length)
        self.last_scenarios = scenarios
        strategy_names = list(self.strategy_trainer.strategies.keys())
        results = {}
        for strategy_name in strategy_names:
            results[strategy_name] = []
            for scenario in scenarios:
                perf = await self.simulator.simulate_strategy_in_scenario(strategy_name, scenario)
                self.robustness_scorer.add_performance(perf)
                results[strategy_name].append(perf)
        self.last_results = results
        logger.info(f"Imagination cycle complete: {len(scenarios)} scenarios, {len(strategy_names)} strategies")
        return results

    def get_robustness_scores(self):
        """
        Return robustness scores for all strategies.
        """
        return self.robustness_scorer.calculate_robustness_scores()

    def get_last_scenarios(self):
        return self.last_scenarios

    def get_last_results(self):
        return self.last_results

    def summary(self):
        scores = self.get_robustness_scores()
        logger.info(f"Imagination robustness summary: {scores}")
        return scores

# Integration helper for mirrax.py
async def add_imagination_to_mirrorcore(sync_bus, strategy_trainer, execution_daemon, oracle_engine, enable=True):
    """
    Integrate ImaginationEngine into the main system.
    Returns an ImaginationIntegration instance if enabled, else None.
    """
    if not enable:
        return None
    imagination = ImaginationIntegration(sync_bus, strategy_trainer, execution_daemon, oracle_engine)
    await imagination.initialize()
    logger.info("ImaginationEngine integrated into MirrorCore-X")
    return imagination
