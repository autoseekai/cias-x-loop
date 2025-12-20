"""
AI Scientist for SCI - Complete Implementation v3.0
An AI Scientist system for SCI domain based on Kosmos and CIAS-X algorithms

Key Features:
- AnalysisAgent uses LLM for intelligent analysis
- Supports all OpenAI-compatible LLM services
- Complete Pareto verification, trend analysis and experiment recommendations
"""

__version__ = "3.0.0"
__author__ = "AI Scientist Team"

from .core.scientist import AIScientist
from .core.data_structures import (
    SCIConfiguration,
    ForwardConfig,
    ReconParams,
    TrainConfig,
    Metrics,
    Artifacts,
    ExperimentResult,
    ReconFamily,
    UQScheme,
)
from .models.world_model import WorldModel
from .agents.planner import PlannerAgent
from .agents.executor import ExecutorAgent
from .agents.analysis import AnalysisAgent
from .llm.client import LLMClient

__all__ = [
    # Core
    "AIScientist",
    # Data Structures
    "SCIConfiguration",
    "ForwardConfig",
    "ReconParams",
    "TrainConfig",
    "Metrics",
    "Artifacts",
    "ExperimentResult",
    "ReconFamily",
    "UQScheme",
    # Models
    "WorldModel",
    # Agents
    "PlannerAgent",
    "ExecutorAgent",
    "AnalysisAgent",
    # LLM
    "LLMClient",
]
