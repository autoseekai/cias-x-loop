"""
Learning Agent Implementation
Uses LlamaIndex to process documents and extract structured knowledge.
"""

import logging
import os
from typing import List, Any
import json

try:
    from llama_index.core import (
        VectorStoreIndex,
        SimpleDirectoryReader,
        StorageContext,
        load_index_from_storage,
        Settings
    )
    from llama_index.llms.openai import OpenAI
    LLAMA_INDEX_AVAILABLE = True
except ImportError:
    LLAMA_INDEX_AVAILABLE = False

from src.agents.core.abstract_agents import AbstractLearningAgent
from src.agents.sci.structures import LearnedKnowledge, Rule

logger = logging.getLogger(__name__)

class LearningAgent(AbstractLearningAgent):
    """
    RAG-based Learning Agent using LlamaIndex.
    """

    def __init__(self, storage_dir: str = "./storage/doc_index", model: str = "gpt-4o"):
        super().__init__("LearningAgent", None, None) # BaseAgent init, might need adjustment
        self.storage_dir = storage_dir
        self.model_name = model
        self.index = None

        if not LLAMA_INDEX_AVAILABLE:
            logger.warning("LlamaIndex not installed. Learning functionality will be limited.")
            return

        # Initialize LlamaIndex Settings
        # Since we use the project's LLMClient for other agents, we might want to unify.
        # But LlamaIndex works best with its own LLM wrappers.
        # We assume OPENAI_API_KEY is set in env.
        try:
            Settings.llm = OpenAI(model=model, temperature=0.1)
        except Exception as e:
            logger.error(f"Failed to initialize LlamaIndex OpenAI LLM: {e}")

    def learn(self, doc_paths: List[str], goal: str) -> LearnedKnowledge:
        """
        Learn from documents to support the goal.
        """
        if not LLAMA_INDEX_AVAILABLE:
            logger.error("Cannot learn: LlamaIndex not available.")
            return LearnedKnowledge()

        if not doc_paths:
            logger.warning("No documents provided for learning.")
            return LearnedKnowledge()

        try:
            # 1. Load or Build Index
            self._build_or_load_index(doc_paths)

            # 2. Extract Knowledge
            return self._extract_knowledge(goal)

        except Exception as e:
            logger.error(f"Learning failed: {e}")
            return LearnedKnowledge()

    def _build_or_load_index(self, doc_paths: List[str]):
        """
        Builds a new index from documents or loads existing if available
        (Logic can be enhanced to check if docs changed).
        For now, we rebuild if docs are provided to ensure freshness.
        """
        if not os.path.exists(self.storage_dir):
            os.makedirs(self.storage_dir)

        logger.info(f"Loading documents from {len(doc_paths)} paths...")
        # Verify files exist
        valid_paths = [p for p in doc_paths if os.path.exists(p)]
        if not valid_paths:
            raise ValueError("No valid document paths found.")

        reader = SimpleDirectoryReader(input_files=valid_paths)
        documents = reader.load_data()

        logger.info("Indexing documents...")
        self.index = VectorStoreIndex.from_documents(documents)

        # Persist
        self.index.storage_context.persist(persist_dir=self.storage_dir)
        logger.info(f"Index persisted to {self.storage_dir}")

    def _extract_knowledge(self, goal: str) -> LearnedKnowledge:
        """
        Query the index to extract structured rules.
        """
        query_engine = self.index.as_query_engine()

        # We use a two-step approach or a complex prompt.
        # Here we use a direct prompt expecting JSON.
        # Note: LlamaIndex has structured output capabilities (PydanticProgram),
        # but for simplicity/robustness without heavy deps we use prompt engineering here
        # or we could use the LLMClient structure if we wanted.
        # Let's try standard query with specific formatting instructions.

        prompt = f"""
        You are an expert scientific researcher.
        Goal: {goal}

        Based on the provided context, extract key knowledge to guide experiments.
        Output MUST be valid JSON matching this structure:
        {{
            "rules": [
                {{"description": "...", "source": "...", "priority": "critical/recommended"}}
            ],
            "suggestions": [
                 {{"description": "...", "source": "...", "priority": "optional"}}
            ],
            "insights": [
                {{"title": "...", "description": "...", "evidence_ids": []}}
            ]
        }}

        Focus on:
        1. Parameter constraints (min/max values, dependencies).
        2. Best practices for configurations.
        3. Performance trade-offs.
        """

        response = query_engine.query(prompt)
        response_text = str(response)

        # Parse JSON
        try:
            # Clean potential markdown code blocks
            clean_text = response_text.replace("```json", "").replace("```", "").strip()
            data = json.loads(clean_text)
            return LearnedKnowledge.model_validate(data)
        except Exception as e:
            logger.error(f"Failed to parse knowledge JSON: {e}. Raw: {response_text}")
            return LearnedKnowledge()
