"""
Analysis Agent

Uses LLM for intelligent Pareto verification, trend analysis, and experiment recommendations
"""

import json
from typing import Dict, List, Any, Tuple, Optional

import numpy as np
from loguru import logger

from ...llm.client import LLMClient
from .world_model import WorldModel

from ...agents.utils import Utils


from ...core.bus import MessageBus, Event
from ..base import BaseAgent


class AnalysisAgent(BaseAgent):
    """Analysis Agent - Uses LLM for intelligent analysis (Event-Driven)"""

    def __init__(self, llm_config: Dict[str, Any], bus: MessageBus, world_model: Optional[Any] = None):
        """
        Initialize analysis agent

        Args:
            llm_config: LLM configuration dictionary
            bus: Message Bus
            world_model: World Model instance
        """
        super().__init__("AnalysisAgent", bus)

        self.llm_client = LLMClient(llm_config)
        self.world_model = world_model
        self.objectives = ['psnr', 'ssim', 'latency']
        logger.info("Analysis Agent initialized with LLM")

    def setup_subscriptions(self):
        self.bus.subscribe("ANALYSIS_REQUESTED", self._on_analysis_requested)

    async def _on_analysis_requested(self, event: Event):
        """Handle Analysis Requested event"""
        logger.debug(f"Analysis agent received ANALYSIS_REQUESTED: {event.payload}")
        if event.payload.get('trigger_analysis', False):
            cycle = event.payload.get('cycle', 1)
            logger.info(f"Triggering analysis for cycle {cycle}...")
            # Run analysis
            await self.run_analysis(cycle)

    async def run_analysis(self, cycle: int):
        if not self.world_model:
            return [], {}

        logger.info(f"Running analysis for cycle {cycle}")
        pareto_ids, insights = self.analyze(self.world_model, cycle)

        # Publish completion event with full insights payload
        await self.publish("INSIGHT_GENERATED", {
            "insights": insights,
            "pareto_ids": pareto_ids,
            "cycle": cycle
        })

        return pareto_ids, insights

    def analyze(
        self,
        world_model: WorldModel,
        cycle_number: int
    ) -> Tuple[List[str], Dict[str, Any]]:
        """
        Complete analysis workflow

        Args:
            world_model: World model
            cycle_number: Current cycle number

        Returns:
            Tuple[List[str], Dict]: Pareto front IDs and insights payload
        """
        experiments = world_model.get_all_experiments()

        if len(experiments) < 3:
            logger.warning(f"Too few experiments: {len(experiments)}")
            return [], {"message": "Insufficient data"}

        logger.info(f"Analyzing {len(experiments)} experiments")

        # Step 1: Compute Pareto front
        pareto_ids = self._compute_pareto_front(experiments)
        logger.info(f"Pareto front: {len(pareto_ids)} experiments")

        analysis_records = []

        # Step 2: LLM verification
        verification, meta_ver = self._llm_verify_pareto(
            experiments, pareto_ids, cycle_number
        )
        if meta_ver:
            analysis_records.append(meta_ver)

        # Step 3: LLM trend analysis
        trends, meta_trends = self._llm_analyze_trends(
            experiments, pareto_ids, cycle_number
        )
        if meta_trends:
            analysis_records.append(meta_trends)

        # Step 4: LLM generate recommendations
        recommendations, meta_recs = self._llm_generate_recommendations(
            experiments, trends, cycle_number
        )
        if meta_recs:
            analysis_records.append(meta_recs)

        # Construct complete insights payload object to be passed to Director
        insights = {
            'pareto_front_ids': pareto_ids,
            'llm_analyses': analysis_records,
            # Keeping these for easy access if needed by other components
            'verification': verification,
            'trends': trends,
            'recommendations': recommendations,
            'cycle': cycle_number,
            'total_experiments_analyzed': len(experiments)
        }

        return pareto_ids, insights

    def _compute_pareto_front(self, experiments: List[Any]) -> List[str]:
        """
        Compute Pareto front

        Args:
            experiments: List of experiments

        Returns:
            List of Pareto front experiment IDs
        """
        if not experiments:
            return []

        obj_matrix = []
        exp_ids = []

        for exp in experiments:
            values = [
                exp.metrics.psnr,
                exp.metrics.ssim,
                -exp.metrics.latency  # Minimize latency
            ]
            obj_matrix.append(values)
            exp_ids.append(exp.experiment_id)

        obj_matrix = np.array(obj_matrix)
        is_pareto = np.ones(len(obj_matrix), dtype=bool)

        for i, obj_i in enumerate(obj_matrix):
            if is_pareto[i]:
                dominated = np.all(obj_matrix >= obj_i, axis=1) & \
                           np.any(obj_matrix > obj_i, axis=1)
                is_pareto[dominated] = False

        return [exp_ids[i] for i in range(len(exp_ids)) if is_pareto[i]]

    def _llm_verify_pareto(
        self,
        all_exps: List[Any],
        pareto_ids: List[str],
        cycle: int
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        """
        LLM verification of Pareto front
        Returns: (verification_result, metadata_record)
        """
        pareto_exps = [e for e in all_exps if e.experiment_id in pareto_ids]

        # Prepare data summary
        pareto_lines = []
        for exp in pareto_exps[:10]:  # Max 10
            pareto_lines.append(
                f"{exp.experiment_id}: PSNR={exp.metrics.psnr:.2f}, "
                f"SSIM={exp.metrics.ssim:.4f}, Latency={exp.metrics.latency:.1f}ms"
            )

        psnrs = [e.metrics.psnr for e in all_exps]
        ssims = [e.metrics.ssim for e in all_exps]
        latencies = [e.metrics.latency for e in all_exps]

        prompt = f"""You are an SCI domain expert. Please verify the reasonableness of the Pareto front.

Total experiments: {len(all_exps)}
Pareto front: {len(pareto_ids)} points

Pareto points:
{chr(10).join(pareto_lines)}

Statistics:
- PSNR: {min(psnrs):.2f} - {max(psnrs):.2f} dB
- SSIM: {min(ssims):.4f} - {max(ssims):.4f}
- Latency: {min(latencies):.1f} - {max(latencies):.1f} ms

Please analyze:
1. Is the Pareto front reasonable
2. Are there any anomalies
3. Trade-off quality
4. Improvement suggestions

Return JSON format:
{{
    "is_reasonable": bool,
    "anomalies": [],
    "suggestions": []
}}"""

        messages = [
            {"role": "system", "content": "You are an SCI domain expert"},
            {"role": "user", "content": prompt}
        ]

        try:
            response = self.llm_client.chat(messages, "json")
            json_content = Utils.extract_json_from_response(response['content'])
            verification = json.loads(json_content)

            metadata = {
                'type': 'pareto_verification',
                'prompt': prompt,
                'response': response['content'],
                'parsed_result': verification,
                'model': response['model'],
                'tokens': response['tokens'],
                'related_ids': pareto_ids,
                'roles': {exp_id: 'pareto' for exp_id in pareto_ids}
            }

            return verification, metadata
        except Exception as e:
            logger.error(f"Pareto verification failed: {e}")
            return {'is_reasonable': True, 'error': str(e)}, None

    def _llm_analyze_trends(
        self,
        all_exps: List[Any],
        pareto_ids: List[str],
        cycle: int
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        """
        LLM trend analysis
        Returns: (trends_result, metadata_record)
        """
        psnrs = [e.metrics.psnr for e in all_exps]
        best_exp = max(all_exps, key=lambda e: e.metrics.psnr)

        prompt = f"""You are a data analysis expert. Please analyze experiment trends.

Total experiments: {len(all_exps)}
PSNR range: {min(psnrs):.2f} - {max(psnrs):.2f} dB
Best experiment: PSNR={best_exp.metrics.psnr:.2f}, SSIM={best_exp.metrics.ssim:.4f}

Please analyze:
1. Key findings (main factors affecting performance)
2. Best configuration patterns
3. Performance bottlenecks
4. Unexpected insights

Return JSON: {{"key_findings": [], "best_patterns": {{}}, "bottlenecks": []}}"""

        messages = [
            {"role": "system", "content": "You are a data analysis expert"},
            {"role": "user", "content": prompt}
        ]

        try:
            response = self.llm_client.chat(messages, "json")
            trends = json.loads(Utils.extract_json_from_response(response['content']))

            all_exp_ids = [e.experiment_id for e in all_exps]
            metadata = {
                'type': 'trend_analysis',
                'prompt': prompt,
                'response': response['content'],
                'parsed_result': trends,
                'model': response['model'],
                'tokens': response['tokens'],
                'related_ids': all_exp_ids,
                'roles': {exp_id: 'analyzed' for exp_id in all_exp_ids}
            }

            return trends, metadata
        except Exception as e:
            logger.error(f"Trend analysis failed: {e}")
            return {'error': str(e)}, None

    def _llm_generate_recommendations(
        self,
        all_exps: List[Any],
        trends: Dict[str, Any],
        cycle: int
    ) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        """
        LLM generate experiment recommendations
        Returns: (recommendations_result, metadata_record)
        """
        best_psnr = max([e.metrics.psnr for e in all_exps])

        prompt = f"""Based on the analysis, provide experiment recommendations.

Current best PSNR: {best_psnr:.2f} dB
Completed: {len(all_exps)} experiments

Please provide:
1. 3 specific configuration suggestions
2. Exploration strategy (explore or exploit)
3. Expected improvements

Return JSON: {{"config_suggestions": [], "strategy": "", "expected_improvements": {{}}}}"""

        messages = [
            {"role": "system", "content": "You are an experiment design expert"},
            {"role": "user", "content": prompt}
        ]

        try:
            response = self.llm_client.chat(messages, "json")
            recommendations = json.loads(Utils.extract_json_from_response(response['content']))

            all_exp_ids = [e.experiment_id for e in all_exps]
            metadata = {
                'type': 'recommendation',
                'prompt': prompt,
                'response': response['content'],
                'parsed_result': recommendations,
                'model': response['model'],
                'tokens': response['tokens'],
                'related_ids': all_exp_ids,
                'roles': {exp_id: 'reference' for exp_id in all_exp_ids}
            }

            return recommendations, metadata
        except Exception as e:
            logger.error(f"Recommendation failed: {e}")
            return {'error': str(e)}, None
