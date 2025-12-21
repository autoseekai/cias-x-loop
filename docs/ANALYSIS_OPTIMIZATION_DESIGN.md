# Analysis Agent 优化设计方案

## 目标
优化 AnalysisAgent，使其分析结果能够更有效地指导 PlannerAgent 生成下一轮实验配置。

## 当前问题分析

### 现有流程
1. **Pareto Front 验证** - 验证合理性，但缺少定量分析
2. **趋势分析** - 发现关键因素，但与具体参数关联度不足
3. **配置推荐** - 给出建议，但过于宽泛，缺乏可执行的具体参数值

### 主要不足
1. **缺乏参数级别的洞察**: 没有分析具体参数（如 num_stages, learning_rate）对性能的影响
2. **探索-利用策略模糊**: 没有明确量化探索空间的覆盖度
3. **参数组合分析缺失**: 没有分析参数之间的交互效应
4. **历史经验利用不足**: 没有充分利用已测试配置的成功/失败模式
5. **推荐不够具体**: 给出的建议缺乏可直接使用的参数值

## 优化设计方案

### 1. 增强的分析流程

```
┌─────────────────────────────────────────────────────────────┐
│              Enhanced Analysis Workflow                      │
└─────────────────────────────────────────────────────────────┘

Step 1: Pareto Front Analysis (保留 + 增强)
├── Compute Pareto front
├── LLM verification
└── **NEW**: Extract Pareto configs' parameter distributions

Step 2: Parameter Sensitivity Analysis (新增)
├── Analyze impact of each parameter on objectives
├── Identify high-impact vs low-impact parameters
├── Detect parameter interaction effects
└── Generate parameter importance ranking

Step 3: Exploration Coverage Analysis (新增)
├── Map tested parameter combinations
├── Identify unexplored/under-explored regions
├── Calculate design space coverage percentage
└── Suggest high-potential unexplored regions

Step 4: Configuration Pattern Mining (新增)
├── Cluster successful configurations
├── Extract common patterns from top performers
├── Identify failure patterns to avoid
└── Generate actionable config templates

Step 5: Trend Analysis (增强)
├── Performance trends over cycles
├── Convergence analysis
├── Diminishing returns detection
└── **NEW**: Parameter-specific trend insights

Step 6: Strategic Recommendations (重构)
├── **NEW**: Specific parameter value recommendations
├── Exploration vs Exploitation ratio suggestion
├── Multi-objective trade-off guidance
└── Prioritized action items with expected gains
```

### 2. 新增分析方法设计

#### 2.1 Parameter Sensitivity Analysis

```python
def _analyze_parameter_sensitivity(
    self,
    experiments: List[Any],
    design_space: Dict[str, List[Any]]
) -> Dict[str, Any]:
    """
    分析每个参数对性能的影响

    Returns:
        {
            "parameter_impact": {
                "num_stages": {
                    "importance_score": 0.85,  # 0-1
                    "correlation_with_psnr": 0.72,
                    "correlation_with_ssim": 0.68,
                    "best_values": [7, 9],  # 性能最好的值
                    "avoid_values": [5],     # 性能较差的值
                    "recommendations": "Prefer higher stages (7-9)"
                },
                "learning_rate": {...},
                ...
            },
            "interaction_effects": [
                {
                    "params": ["num_stages", "num_features"],
                    "effect": "high_stages + high_features = best PSNR",
                    "confidence": 0.78
                }
            ]
        }
    """
```

**分析方法**:
- 单变量分析: 固定其他参数，观察单个参数变化的影响
- 相关性分析: 计算参数与目标指标的相关系数
- 方差分析: 确定哪些参数贡献了最多的性能差异
- LLM 辅助解释: 让 LLM 解释统计发现的物理/算法意义

#### 2.2 Exploration Coverage Analysis

```python
def _analyze_exploration_coverage(
    self,
    experiments: List[Any],
    design_space: Dict[str, List[Any]]
) -> Dict[str, Any]:
    """
    评估设计空间的探索覆盖度

    Returns:
        {
            "overall_coverage": 0.35,  # 35% of design space explored
            "parameter_coverage": {
                "compression_ratio": {
                    "tested_values": [8, 16],
                    "untested_values": [24, 32],
                    "coverage": 0.5
                },
                ...
            },
            "combination_coverage": {
                "total_combinations": 1728,
                "tested_combinations": 45,
                "coverage": 0.026
            },
            "high_potential_regions": [
                {
                    "description": "High stages (9) + Medium features (64)",
                    "reason": "Adjacent configs show promise",
                    "priority": "high"
                }
            ],
            "saturated_regions": [
                {
                    "description": "Low stages (5) + Low features (32)",
                    "tested_count": 8,
                    "avg_performance": "poor",
                    "recommendation": "Stop exploring this region"
                }
            ]
        }
    """
```

**分析方法**:
- 网格映射: 将设计空间可视化为网格，标记已测试点
- 密度分析: 识别过度测试和未充分测试的区域
- 基于梯度的推荐: 从高性能区域推断可能的改进方向
- LLM 策略规划: 让 LLM 基于覆盖度制定探索策略

#### 2.3 Configuration Pattern Mining

```python
def _mine_configuration_patterns(
    self,
    experiments: List[Any],
    top_k: int = 10
) -> Dict[str, Any]:
    """
    从实验中挖掘配置模式

    Returns:
        {
            "success_patterns": [
                {
                    "pattern": {
                        "num_stages": [7, 9],
                        "num_features": [64, 128],
                        "learning_rate": [1e-4]
                    },
                    "performance": {
                        "avg_psnr": 35.2,
                        "avg_ssim": 0.92
                    },
                    "sample_count": 6,
                    "insight": "Higher stages with medium-high features consistently perform well"
                }
            ],
            "failure_patterns": [
                {
                    "pattern": {
                        "num_stages": [5],
                        "num_features": [32],
                        "learning_rate": [5e-5]
                    },
                    "performance": {
                        "avg_psnr": 28.5,
                        "avg_ssim": 0.78
                    },
                    "sample_count": 4,
                    "insight": "Low complexity configs underfit the problem"
                }
            ],
            "config_templates": [
                {
                    "name": "High Quality Template",
                    "config": {
                        "compression_ratio": "16",  # specific value
                        "num_stages": "[7, 9]",     # range
                        "num_features": "[64, 128]",
                        "learning_rate": "1e-4"
                    },
                    "expected_psnr": "34-36 dB",
                    "use_case": "exploitation"
                },
                {
                    "name": "High Speed Template",
                    "config": {...},
                    "expected_latency": "50-80 ms",
                    "use_case": "trade-off"
                }
            ]
        }
    """
```

**分析方法**:
- 聚类分析: K-means 或 DBSCAN 聚类配置
- 模式提取: 从每个簇中提取共同特征
- 性能标注: 标记每个模式的平均性能
- LLM 模板生成: 让 LLM 将模式转化为可执行的配置模板

### 3. 增强的推荐系统

#### 3.1 Structured Recommendations

```python
def _generate_structured_recommendations(
    self,
    param_analysis: Dict,
    coverage_analysis: Dict,
    pattern_analysis: Dict,
    current_cycle: int,
    budget_remaining: int
) -> Dict[str, Any]:
    """
    生成结构化的、可执行的推荐

    Returns:
        {
            "strategy": {
                "phase": "exploration|exploitation|balanced",
                "reasoning": "...",
                "exploration_ratio": 0.4,  # 40% explore, 60% exploit
                "focus_areas": ["parameter_x", "region_y"]
            },
            "concrete_configs": [
                {
                    "id": "rec_1",
                    "type": "exploitation",
                    "config": {
                        "compression_ratio": 16,
                        "mask_type": "optimized",
                        "num_stages": 9,
                        "num_features": 128,
                        "num_blocks": 4,
                        "learning_rate": 1e-4,
                        "activation": "LeakyReLU"
                    },
                    "rationale": "Refine best Pareto config with higher features",
                    "expected_improvement": "+1.5 dB PSNR",
                    "confidence": "high",
                    "priority": 1
                },
                {
                    "id": "rec_2",
                    "type": "exploration",
                    "config": {
                        "compression_ratio": 24,
                        "mask_type": "random",
                        "num_stages": 7,
                        "num_features": 64,
                        "num_blocks": 3,
                        "learning_rate": 5e-5,
                        "activation": "ReLU"
                    },
                    "rationale": "Test unexplored high compression region",
                    "expected_improvement": "Unknown, exploratory",
                    "confidence": "medium",
                    "priority": 2
                }
            ],
            "parameter_guidelines": {
                "num_stages": {
                    "recommendation": "Use 7 or 9",
                    "confidence": "high",
                    "supporting_evidence": "85% of top configs use these values"
                },
                "learning_rate": {
                    "recommendation": "Prefer 1e-4 for quality, 5e-5 for stability",
                    "confidence": "medium",
                    "supporting_evidence": "Correlation analysis"
                }
            },
            "avoid_configs": [
                {
                    "pattern": {"num_stages": 5, "num_features": 32},
                    "reason": "Consistently poor performance (<30 dB)",
                    "tested_count": 5
                }
            ]
        }
    """
```

### 4. LLM Prompt 优化

#### 4.1 Parameter Analysis Prompt

```python
prompt = f"""You are an expert in SCI (Snapshot Compressive Imaging) and machine learning.

**Task**: Analyze how each parameter affects reconstruction quality.

**Data Summary**:
- Total experiments: {len(experiments)}
- Parameters analyzed: {list(design_space.keys())}
- Performance range: PSNR {min_psnr:.2f} - {max_psnr:.2f} dB

**Statistical Analysis**:
{parameter_statistics}

**Top Performing Configs** (Top 10):
{top_configs_summary}

**Bottom Performing Configs** (Bottom 10):
{bottom_configs_summary}

**Questions to Answer**:
1. Which parameters have the strongest impact on PSNR/SSIM?
2. What are the optimal value ranges for each critical parameter?
3. Are there any parameter interaction effects? (e.g., param A works well only when param B is set to X)
4. Which parameters can be safely fixed to reduce search space?
5. What are the physical/algorithmic reasons for the observed patterns?

**Output Format** (JSON):
{{
    "high_impact_params": [
        {{
            "name": "num_stages",
            "impact_score": 0.85,
            "optimal_range": [7, 9],
            "reasoning": "More stages allow better reconstruction but plateau after 9"
        }}
    ],
    "low_impact_params": [...],
    "interaction_effects": [
        {{
            "params": ["num_stages", "num_features"],
            "effect": "Synergistic: both should be high together",
            "confidence": "high"
        }}
    ],
    "fixable_params": [
        {{
            "name": "activation",
            "recommended_value": "LeakyReLU",
            "reasoning": "No significant difference observed, use default"
        }}
    ],
    "key_insights": [...]
}}
"""
```

#### 4.2 Strategic Planning Prompt

```python
prompt = f"""You are an AI scientist designing the next round of experiments.

**Current Status**:
- Cycle: {cycle_number}/{max_cycles}
- Budget used: {experiments_done}/{total_budget}
- Budget remaining: {budget_remaining}
- Best PSNR so far: {best_psnr:.2f} dB
- Pareto front size: {len(pareto_ids)}

**Analysis Results**:

**Parameter Insights**:
{json.dumps(param_analysis['high_impact_params'], indent=2)}

**Exploration Coverage**:
- Overall coverage: {coverage_analysis['overall_coverage']:.1%}
- Unexplored high-potential regions: {len(coverage_analysis['high_potential_regions'])}
- Saturated regions: {len(coverage_analysis['saturated_regions'])}

**Success Patterns**:
{json.dumps(pattern_analysis['success_patterns'][:3], indent=2)}

**Failure Patterns to Avoid**:
{json.dumps(pattern_analysis['failure_patterns'][:3], indent=2)}

**Your Task**:
Design {n_configs} concrete experiment configurations for the next cycle.

**Strategy Considerations**:
1. **Exploration vs Exploitation**:
   - Early cycles (1-2): 70% exploration, 30% exploitation
   - Mid cycles (3-4): 50/50 balance
   - Late cycles (5+): 30% exploration, 70% exploitation

2. **Multi-objective Balance**:
   - Don't just optimize PSNR; consider SSIM and latency trade-offs
   - Target different regions of Pareto front

3. **Avoid Redundancy**:
   - Don't test configs too similar to existing ones
   - Don't waste budget on known poor regions

4. **Leverage Insights**:
   - Use high-impact params from analysis
   - Apply successful patterns
   - Avoid failure patterns

**Output Format** (JSON):
{{
    "strategy_summary": {{
        "phase": "exploration|exploitation|balanced",
        "exploration_ratio": 0.4,
        "focus": "Refine Pareto front while testing unexplored regions",
        "expected_outcomes": "Improve best PSNR by 1-2 dB, discover efficient low-latency configs"
    }},
    "recommended_configs": [
        {{
            "id": 1,
            "type": "exploitation",
            "config": {{
                "compression_ratio": 16,
                "mask_type": "optimized",
                "num_stages": 9,
                "num_features": 128,
                "num_blocks": 4,
                "learning_rate": 0.0001,
                "activation": "LeakyReLU"
            }},
            "rationale": "Refine best Pareto config (exp_123) by increasing features",
            "based_on": "exp_123",
            "expected_psnr": "35.5 dB",
            "expected_improvement": "+1.2 dB vs exp_123",
            "confidence": "high",
            "priority": 1
        }},
        {{
            "id": 2,
            "type": "exploration",
            "config": {{...}},
            "rationale": "Test high compression (24) with medium complexity",
            "based_on": "coverage_gap",
            "expected_psnr": "32-34 dB",
            "expected_improvement": "Unknown, exploratory",
            "confidence": "medium",
            "priority": 2
        }}
    ],
    "parameter_guidelines": {{
        "num_stages": "Prefer 7 or 9 (high impact)",
        "learning_rate": "Use 1e-4 for quality-focused, 5e-5 for stability"
    }}
}}
"""
```

### 5. PlannerAgent 集成

#### 5.1 使用增强分析结果

```python
# In PlannerAgent._gather_planning_context()

context = {
    'pareto_configs': [...],  # 现有

    # 新增: 参数级别指导
    'parameter_insights': {
        'high_impact': analysis['param_analysis']['high_impact_params'],
        'optimal_ranges': analysis['param_analysis']['optimal_ranges'],
        'avoid_combinations': analysis['param_analysis']['failure_patterns']
    },

    # 新增: 探索覆盖信息
    'exploration_status': {
        'coverage': analysis['coverage']['overall_coverage'],
        'unexplored_regions': analysis['coverage']['high_potential_regions'],
        'saturated_regions': analysis['coverage']['saturated_regions']
    },

    # 新增: 配置模板
    'templates': analysis['pattern_analysis']['config_templates'],

    # 新增: 具体推荐
    'concrete_recommendations': analysis['recommendations']['concrete_configs'],

    'historical_insights': [...],  # 现有
    'recommendations': [...]  # 现有
}
```

#### 5.2 优化 Prompt 使用分析结果

```python
# In PlannerAgent._build_planning_prompt()

# 添加参数指导部分
if context.get('parameter_insights'):
    insights = context['parameter_insights']
    param_guidance = f"""
## Parameter-Level Guidance
**High-Impact Parameters** (prioritize these):
{format_high_impact_params(insights['high_impact'])}

**Optimal Ranges** (from statistical analysis):
{format_optimal_ranges(insights['optimal_ranges'])}

**Combinations to Avoid**:
{format_avoid_patterns(insights['avoid_combinations'])}
"""

# 添加探索状态
if context.get('exploration_status'):
    status = context['exploration_status']
    exploration_guidance = f"""
## Exploration Status
- Design space coverage: {status['coverage']:.1%}
- Recommend: {'Explore more' if status['coverage'] < 0.3 else 'Focus on exploitation'}

**High-Potential Unexplored Regions**:
{format_unexplored_regions(status['unexplored_regions'])}

**Saturated Regions** (avoid further testing):
{format_saturated_regions(status['saturated_regions'])}
"""

# 添加配置模板
if context.get('templates'):
    template_guidance = f"""
## Proven Configuration Templates
You can use these templates as starting points:
{format_templates(context['templates'])}
"""

# 添加具体推荐
if context.get('concrete_recommendations'):
    recs = context['concrete_recommendations']
    recommendations_section = f"""
## Previous LLM Recommendations (from last cycle)
Consider but don't duplicate:
{format_concrete_recommendations(recs)}
"""
```

### 6. 实现优先级

#### Phase 1: 核心增强 (最高优先级)
- [ ] Parameter Sensitivity Analysis (基础统计版本)
- [ ] Configuration Pattern Mining (Top-K 聚类)
- [ ] Enhanced Recommendations (具体参数值)
- [ ] Updated LLM Prompts (参数指导)

#### Phase 2: 高级分析 (中等优先级)
- [ ] Exploration Coverage Analysis (完整版)
- [ ] Parameter Interaction Detection
- [ ] Multi-objective Trade-off Analysis
- [ ] Convergence Detection

#### Phase 3: 智能优化 (低优先级)
- [ ] Adaptive Strategy Selection (基于cycle自动调整)
- [ ] Meta-learning from Historical Cycles
- [ ] Uncertainty-aware Recommendations
- [ ] Active Learning Integration

### 7. 预期效果

#### 定量指标
- **参数效率**: 减少 30% 的无效实验（避免已知差的区域）
- **收敛速度**: 提前 1-2 个 cycle 达到性能目标
- **Pareto 质量**: Pareto front 覆盖度提升 40%
- **探索效率**: 设计空间覆盖度提升 50%

#### 定性改进
- **更具体的建议**: 从"增加stages"到"设置 num_stages=9, num_features=128"
- **更科学的策略**: 基于数据的 exploration/exploitation 平衡
- **更少的冗余**: 避免重复测试相似配置
- **更好的可解释性**: 每个推荐都有清晰的依据和预期效果

### 8. 测试验证

#### 单元测试
```python
def test_parameter_sensitivity():
    # 测试参数影响分析
    assert param_impact['num_stages']['importance_score'] > 0.5
    assert 'optimal_range' in param_impact['num_stages']

def test_pattern_mining():
    # 测试模式挖掘
    patterns = mine_patterns(experiments)
    assert len(patterns['success_patterns']) > 0
    assert all('config' in p for p in patterns['success_patterns'])

def test_concrete_recommendations():
    # 测试推荐具体性
    recs = generate_recommendations(...)
    assert all('config' in r for r in recs['concrete_configs'])
    assert all(isinstance(r['config']['num_stages'], int)
               for r in recs['concrete_configs'])
```

#### 集成测试
```python
def test_end_to_end_workflow():
    # 模拟多轮循环
    for cycle in range(1, 6):
        analysis = analyzer.analyze(world_model, cycle)
        configs = planner.plan_experiments(
            world_summary, design_space, budget,
            world_model=world_model
        )

        # 验证推荐质量
        assert len(configs) > 0
        assert no_duplicates(configs)
        assert configs_follow_guidelines(configs, analysis)
```

## 总结

这个优化设计通过以下方式增强 Analysis 对 Config 生成的指导能力:

1. **更细粒度**: 从实验级别分析深入到参数级别分析
2. **更具体**: 给出具体参数值而非宽泛建议
3. **更智能**: 基于统计 + LLM 的混合分析方法
4. **更系统**: 覆盖度分析确保充分探索设计空间
5. **更高效**: 避免无效实验，加速收敛

下一步可以根据优先级逐步实现这些功能，建议先从 Phase 1 开始，验证效果后再扩展到 Phase 2 和 3。
