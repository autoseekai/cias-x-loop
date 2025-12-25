"""
Core Abstract Agents
Defines the abstract base classes for the Planner, Reviewer, and Analysis agents.
This allows for implementing domain-specific versions (e.g. SCI, NLP, Bio) while sharing the same workflow structure.
"""

from abc import abstractmethod
from typing import List, Dict, Any, TypeVar, Generic, Tuple, Optional
from ..base import BaseAgent
from pydantic import BaseModel

# Generic Types for Configuration and Result
T_Config = TypeVar("T_Config", bound=BaseModel)
T_Result = TypeVar("T_Result", bound=BaseModel)
T_Review = TypeVar("T_Review", bound=BaseModel)


class AbstractLearningAgent(BaseAgent):
    """
    Abstract Learning Agent: Responsible for acquiring knowledge from documents or other sources.
    This agent builds the initial 'LearnedKnowledge' that guides the rest of the process.
    """

    @abstractmethod
    def learn(self, doc_paths: List[str], goal: str) -> Any:
        """
        Learn from the provided documents based on the verification goal.

        Args:
            doc_paths: List of paths to documents (PDF, MD, etc.)
            goal: The specific goal to focus learning on (e.g., "optimize PSNR for SCI")

        Returns:
            A LearnedKnowledge object containing rules, constraints, and suggestions.
        """
        pass


class AbstractPlannerAgent(BaseAgent, Generic[T_Config]):
    """
    Abstract Planner Agent: Responsible for generating experiment configurations.
    """

    @abstractmethod
    def plan_experiments(
        self,
        world_summary: Dict[str, Any],
        design_space: Dict[str, Any],
        budget: int,
        feedback: str = "",
        strategy: str = "",
        knowledge_context: str = ""
    ) -> List[T_Config]:
        """
        Generate a list of experiment configurations.

        Args:
            world_summary: Summary of current world state (metrics, progress)
            design_space: Definition of the search space (hyperparameters)
            budget: Number of experiments to plan
            feedback: Feedback from previous iterations (e.g. from Reviewer)
            strategy: High-level strategy to follow (e.g. "Exploit best models")
            knowledge_context: Relevant knowledge distilled by the Learning Agent

        Returns:
            List of configuration objects (T_Config)
        """
        pass

    @abstractmethod
    def _generate_dynamic_strategy(
        self,
        world_summary: Dict[str, Any],
        context: Dict[str, Any]
    ) -> str:
        """
        Generate a high-level research strategy (Meta-Prompting).
        """
        pass


class AbstractReviewerAgent(BaseAgent, Generic[T_Config, T_Review]):
    """
    Abstract Reviewer Agent: Responsible for validating and critiquing plans.
    """

    @abstractmethod
    async def review_plan(
        self,
        proposed_configs: List[T_Config],
        context: Dict[str, Any],
        validation_rules: Optional[List[str]] = None
    ) -> T_Review:
        """
        Review the proposed experiment plan.

        Args:
            proposed_configs: List of configs to review
            context: Context including cycle, strategy (from State)
            validation_rules: List of constraints/rules from the Learning Agent (optional)

        Returns:
            ReviewResult object (T_Review) containing approval status, feedback, and critiques.
        """
        pass


class AbstractExecutorAgent(BaseAgent, Generic[T_Config, T_Result]):
    """
    Abstract Executor Agent: Responsible for executing experiments.
    In many cases, this is a wrapper around a simulator or hardware interface.
    """

    @abstractmethod
    async def execute_experiments(
        self,
        configs: List[T_Config]
    ) -> List[T_Result]:
        """
        Execute a batch of experiments.

        Args:
            configs: List of approved configurations to run

        Returns:
            List of experiment results (T_Result)
        """
        pass


class AbstractAnalysisAgent(BaseAgent, Generic[T_Result]):
    """
    Abstract Analysis Agent: Responsible for analyzing results and generating insights.
    """

    @abstractmethod
    def analyze(
        self,
        current_results: List[T_Result],
        history_summary: Dict[str, Any],
        cycle: int
    ) -> Tuple[List[str], Dict[str, Any]]:
        """
        Analyze current experiment results against history.

        Args:
            current_results: Results from the current cycle's execution
            history_summary: Summary of all previous cycles
            cycle: The current cycle number

        Returns:
            Tuple containing:
            - List of significant experiment IDs (e.g., new Pareto points)
            - Dictionary of insights (trends, patterns, theory validation)
        """
        pass
