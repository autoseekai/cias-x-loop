"""
AI Scientist Main Loop

Orchestrates Planner, Executor, and Analyzer for automated scientific exploration.
Supports async parallel execution of experiments.
"""

import asyncio
from typing import List, Dict, Any, Tuple
from loguru import logger

from .data_structures import SCIConfiguration
from ..models.world_model import WorldModel
from ..agents.planner import PlannerAgent
from ..agents.executor import ExecutorAgent
from ..agents.analysis import AnalysisAgent


class AIScientist:
    """AI Scientist main loop orchestrator with async support"""

    def __init__(
        self,
        world_model: WorldModel,
        planner: PlannerAgent,
        executor: ExecutorAgent,
        analyzer: AnalysisAgent,
        design_space: Dict[str, Any],
        budget_max: int
    ):
        """
        Initialize AI Scientist

        Args:
            world_model: World model for storing experiment history
            planner: Planner agent for generating experiment configs
            executor: Executor agent for running experiments
            analyzer: Analysis agent for LLM analysis
            design_space: Design space defining parameter ranges
            budget_max: Maximum experiment budget
        """
        self.world_model = world_model
        self.planner = planner
        self.executor = executor
        self.analyzer = analyzer
        self.design_space = design_space
        self.budget_max = budget_max

        logger.info(f"AI Scientist initialized: budget={budget_max}")

    async def run_async(
        self,
        initial_configs: List[SCIConfiguration],
        max_cycles: int = 5
    ) -> Tuple[List[str], Dict[str, Any]]:
        """
        Run AI Scientist main loop with async parallel execution

        Args:
            initial_configs: List of initial experiment configurations
            max_cycles: Maximum number of cycles

        Returns:
            Tuple[List[str], Dict]: Pareto front experiment IDs and insights
        """
        logger.info(f"AI Scientist starting (async): budget={self.budget_max}, cycles={max_cycles}")

        budget_used = 0
        pareto_set = []
        final_insights = {}

        # Initialization phase: run initial experiments in parallel
        logger.info(f"Running {len(initial_configs)} initial experiments in parallel...")
        results = await self.executor.run_experiments_async(initial_configs)

        for result in results:
            self.world_model.add_experiment(result)
            budget_used += 1

        logger.info(f"Initialization complete: {budget_used} experiments")

        # Main loop
        cycle = 0

        while budget_used < self.budget_max and cycle < max_cycles:
            cycle += 1
            logger.info(f"\n{'='*60}")
            logger.info(f"Cycle {cycle}/{max_cycles}")
            logger.info(f"{'='*60}")

            # 1. Summarize - Get current state summary
            summary = self.world_model.summarize()
            logger.info(f"Completed: {summary['total_experiments']} experiments")
            if summary['psnr_stats']['max'] > 0:
                logger.info(f"Best PSNR: {summary['psnr_stats']['max']:.2f} dB")

            # 2. Plan - Plan new experiments (with deduplication and rich context)
            budget_remaining = self.budget_max - budget_used
            existing_experiments = self.world_model.get_all_experiments()
            new_configs = self.planner.plan_experiments(
                summary, self.design_space, min(3, budget_remaining),
                existing_experiments=existing_experiments,
                world_model=self.world_model  # Pass world_model for Pareto/insights context
            )
            logger.info(f"Planner: {len(new_configs)} new configs")

            # Limit configs to remaining budget
            configs_to_run = new_configs[:budget_remaining]

            if configs_to_run:
                # 3. Execute - Run experiments in parallel
                logger.info(f"Running {len(configs_to_run)} experiments in parallel...")
                results = await self.executor.run_experiments_async(configs_to_run)

                for result in results:
                    self.world_model.add_experiment(result)
                    budget_used += 1

            # 4. Analyze - LLM analysis
            pareto_set, insights = self.analyzer.analyze(self.world_model, cycle)
            logger.info(f"Pareto: {len(pareto_set)} experiments")

            # Print LLM insights
            if 'trends' in insights and 'key_findings' in insights['trends']:
                logger.info("LLM Trends:")
                for finding in insights['trends']['key_findings'][:3]:
                    logger.info(f"  - {finding}")

            final_insights = insights

        logger.info(f"\n{'='*60}")
        logger.info(f"Run Complete! Total: {budget_used} experiments")
        logger.info(f"{'='*60}")

        return pareto_set, final_insights

    def run(
        self,
        initial_configs: List[SCIConfiguration],
        max_cycles: int = 5
    ) -> Tuple[List[str], Dict[str, Any]]:
        """
        Run AI Scientist main loop (sync wrapper for async execution)

        Args:
            initial_configs: List of initial experiment configurations
            max_cycles: Maximum number of cycles

        Returns:
            Tuple[List[str], Dict]: Pareto front experiment IDs and insights
        """
        return asyncio.run(self.run_async(initial_configs, max_cycles))

    def run_sync(
        self,
        initial_configs: List[SCIConfiguration],
        max_cycles: int = 5
    ) -> Tuple[List[str], Dict[str, Any]]:
        """
        Run AI Scientist main loop (sequential execution, no async)

        Args:
            initial_configs: List of initial experiment configurations
            max_cycles: Maximum number of cycles

        Returns:
            Tuple[List[str], Dict]: Pareto front experiment IDs and insights
        """
        logger.info(f"AI Scientist starting (sync): budget={self.budget_max}, cycles={max_cycles}")

        budget_used = 0
        pareto_set = []
        final_insights = {}

        # Initialization phase: run initial experiments sequentially
        for config in initial_configs:
            result = self.executor.run_experiment(config)
            self.world_model.add_experiment(result)
            budget_used += 1

        logger.info(f"Initialization complete: {budget_used} experiments")

        # Main loop
        cycle = 0

        while budget_used < self.budget_max and cycle < max_cycles:
            cycle += 1
            logger.info(f"\n{'='*60}")
            logger.info(f"Cycle {cycle}/{max_cycles}")
            logger.info(f"{'='*60}")

            # 1. Summarize
            summary = self.world_model.summarize()
            logger.info(f"Completed: {summary['total_experiments']} experiments")
            if summary['psnr_stats']['max'] > 0:
                logger.info(f"Best PSNR: {summary['psnr_stats']['max']:.2f} dB")

            # 2. Plan
            budget_remaining = self.budget_max - budget_used
            existing_experiments = self.world_model.get_all_experiments()
            new_configs = self.planner.plan_experiments(
                summary, self.design_space, min(3, budget_remaining),
                existing_experiments=existing_experiments,
                world_model=self.world_model
            )
            logger.info(f"Planner: {len(new_configs)} new configs")

            # 3. Execute sequentially
            for config in new_configs:
                if budget_used >= self.budget_max:
                    break
                result = self.executor.run_experiment(config)
                self.world_model.add_experiment(result)
                budget_used += 1

            # 4. Analyze
            pareto_set, insights = self.analyzer.analyze(self.world_model, cycle)
            logger.info(f"Pareto: {len(pareto_set)} experiments")

            if 'trends' in insights and 'key_findings' in insights['trends']:
                logger.info("LLM Trends:")
                for finding in insights['trends']['key_findings'][:3]:
                    logger.info(f"  - {finding}")

            final_insights = insights

        logger.info(f"\n{'='*60}")
        logger.info(f"Run Complete! Total: {budget_used} experiments")
        logger.info(f"{'='*60}")

        return pareto_set, final_insights
