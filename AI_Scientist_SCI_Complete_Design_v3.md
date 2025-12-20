# AI Scientist 在 SCI 领域的应用 - 完整设计文档 v3.0

> **最新版本**: AnalysisAgent 使用 LLM 进行智能帕累托前沿验证和趋势分析

---

## 系统简介

基于 Kosmos AI Scientist 和 CIAS-X 循环算法，实现全自动 AI 科学家系统，用于 SCI 领域实验探索。

### 核心升级 v3.0 ⭐

- ✅ **LLM 驱动的帕累托验证**: 使用大语言模型验证帕累托前沿合理性
- ✅ **智能趋势分析**: LLM 自动发现隐藏模式和规律
- ✅ **实验建议生成**: LLM 提供下一步实验方案
- ✅ **OpenAI 兼容**: 支持所有 OpenAI API 格式的 LLM
- ✅ **完整可追溯**: 记录所有 LLM 分析过程和推理

### 系统架构

```
┌───────────────────────────────────────────────────┐
│            AI Scientist 主循环                     │
│                                                    │
│  ┌─────────┐  ┌─────────┐  ┌──────────────────┐ │
│  │Planner  │─▶│Executor │─▶│Analysis + LLM ⭐ │ │
│  └─────────┘  └─────────┘  └──────────────────┘ │
│       │            │                │            │
│       └────────────┼────────────────┘            │
│                    ▼                             │
│             ┌─────────────┐                      │
│             │World Model  │                      │
│             └─────────────┘                      │
└───────────────────────────────────────────────────┘
                    │                    │
            RESTful API          OpenAI Compatible API
                    │                    │
        ┌───────────────────┐   ┌──────────────┐
        │ SCI Service       │   │ LLM Service  │
        │ - CIAS-Core       │   │ - GPT-4      │
        │ - Training        │   │ - DeepSeek   │
        └───────────────────┘   │ - Qwen       │
                                 │ - Ollama     │
                                 └──────────────┘
```

---

## 核心模块

### 1. World Model

**数据库表（新增 LLM 相关）**:

```sql
-- LLM 分析记录表
CREATE TABLE llm_analyses (
    id INTEGER PRIMARY KEY,
    cycle_number INTEGER,
    analysis_type TEXT,  -- pareto_verification, trend_analysis, recommendation
    input_summary TEXT,
    llm_response TEXT,
    conclusions_json TEXT,
    model_name TEXT,
    tokens_used INTEGER,
    timestamp TIMESTAMP
);

-- 帕累托前沿表（增加 LLM 验证）
CREATE TABLE pareto_front (
    id INTEGER PRIMARY KEY,
    experiment_id TEXT,
    cycle_number INTEGER,
    objectives_json TEXT,
    llm_verification TEXT,  -- LLM验证结果
    timestamp TIMESTAMP
);
```

**新增方法**:
- `save_llm_analysis()`: 保存 LLM 分析记录
- `get_historical_analyses()`: 获取历史分析

---

### 2. Analysis Agent + LLM ⭐

**核心功能**:
1. 计算帕累托前沿
2. LLM 验证前沿合理性
3. LLM 深度趋势分析
4. LLM 生成实验建议

**LLM 客户端**:

```python
from openai import OpenAI

class LLMClient:
    def __init__(self, config):
        self.client = OpenAI(
            api_key=config['api_key'],
            base_url=config.get('base_url', 'https://api.openai.com/v1')
        )
        self.model = config.get('model', 'gpt-4-turbo-preview')

    def chat(self, messages, response_format="text"):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.3,
            max_tokens=4096
        )
        return {
            'content': response.choices[0].message.content,
            'tokens': response.usage.total_tokens
        }
```

**分析流程**:

```python
class AnalysisAgent:
    def analyze(self, world_model, cycle):
        # 1. 计算帕累托前沿
        pareto_ids = self._compute_pareto_front(experiments)

        # 2. LLM 验证
        verification = self._llm_verify_pareto(experiments, pareto_ids)

        # 3. LLM 趋势分析
        trends = self._llm_analyze_trends(experiments, pareto_ids)

        # 4. LLM 生成建议
        recommendations = self._llm_generate_recommendations(trends)

        return pareto_ids, {
            'verification': verification,
            'trends': trends,
            'recommendations': recommendations
        }
```

---

### 3. LLM 提示词设计

#### 帕累托验证提示词

```
你是 SCI 领域专家。请验证以下帕累托前沿是否合理。

实验数据:
[实验列表，包含配置和性能指标]

请分析:
1. 是否覆盖关键性能区间
2. 是否存在异常点
3. Trade-off 是否合理
4. 改进建议

返回 JSON 格式:
{
  "is_reasonable": bool,
  "anomalies": [...],
  "suggestions": [...]
}
```

#### 趋势分析提示词

```
你是数据分析专家。请从实验数据中提取深层洞察。

数据统计:
[参数统计、性能统计、相关性]

请分析:
1. 哪些参数影响最大
2. 最佳配置模式
3. 性能瓶颈
4. 意外发现

返回 JSON 格式:
{
  "key_findings": [...],
  "best_patterns": {...},
  "bottlenecks": [...]
}
```

#### 实验建议提示词

```
基于当前进展，请提供实验建议。

当前状态:
[已完成实验数、最佳性能、趋势洞察]

请提供:
1. 优先探索方向
2. 3-5个具体配置建议
3. 探索策略
4. 预期收益

返回 JSON 格式:
{
  "priority_directions": [...],
  "config_suggestions": [...],
  "strategy": "...",
  "expected_improvements": {...}
}
```

---

## 支持的 LLM 服务

| 服务 | Base URL | 模型 |
|------|----------|------|
| OpenAI | https://api.openai.com/v1 | gpt-4-turbo-preview |
| DeepSeek | https://api.deepseek.com/v1 | deepseek-chat |
| Qwen | https://dashscope.aliyuncs.com/compatible-mode/v1 | qwen-turbo |
| Moonshot | https://api.moonshot.cn/v1 | moonshot-v1-8k |
| Ollama (本地) | http://localhost:11434/v1 | llama2, mistral |

---

## 配置文件

```yaml
analysis:
  llm:
    # OpenAI 配置
    base_url: "https://api.openai.com/v1"
    api_key: "${OPENAI_API_KEY}"
    model: "gpt-4-turbo-preview"
    temperature: 0.3
    max_tokens: 4096

    # 或使用 DeepSeek
    # base_url: "https://api.deepseek.com/v1"
    # api_key: "${DEEPSEEK_API_KEY}"
    # model: "deepseek-chat"

    # 或使用本地 Ollama
    # base_url: "http://localhost:11434/v1"
    # api_key: "ollama"
    # model: "llama2"

design_space:
  compression_ratios: [8, 16, 24]
  mask_types: ["random", "optimized"]
  recon_families: ["CIAS-Core"]
  num_stages: [5, 7, 9]
  num_features: [32, 64, 128]
  learning_rates: [1e-4, 5e-5]
```

---

## 使用指南

### 安装依赖

```bash
pip install numpy loguru requests tenacity openai scipy pyyaml
```

### 运行

```bash
# 1. 设置 API Key
export OPENAI_API_KEY="sk-..."

# 2. Mock 模式运行（测试用）
python main_v3.py --mock --config config.yaml

# 3. 查看 LLM 分析结果
sqlite3 world_model.db "SELECT * FROM llm_analyses ORDER BY timestamp DESC LIMIT 5"
```

### 输出示例

```json
{
  "pareto_front": {
    "experiment_ids": ["exp_001", "exp_045", "exp_087"],
    "count": 3,
    "verification": {
      "is_reasonable": true,
      "anomalies": [],
      "tradeoff_quality": "合理的PSNR-延迟权衡"
    }
  },
  "trends": {
    "key_findings": [
      "增加unrolling stages显著提升PSNR",
      "特征通道数与SSIM强相关",
      "压缩比16时性价比最高"
    ],
    "best_patterns": {
      "compression_ratio": 16,
      "num_stages": 7,
      "num_features": 64
    }
  },
  "recommendations": {
    "config_suggestions": [
      {"cr": 16, "stages": 9, "features": 64},
      {"cr": 24, "stages": 7, "features": 128}
    ],
    "strategy": "在CR=16附近做精细探索"
  }
}
```

---

## 扩展开发

### 使用其他 LLM

```python
# 只需修改配置
llm_config = {
    'base_url': 'http://your-llm-service/v1',
    'api_key': 'your-key',
    'model': 'your-model'
}

analyzer = AnalysisAgent(llm_config)
```

### 自定义分析逻辑

```python
class CustomAnalysisAgent(AnalysisAgent):
    def _llm_custom_analysis(self, experiments):
        prompt = "自定义提示词..."
        response = self.llm_client.chat([
            {"role": "user", "content": prompt}
        ])
        return json.loads(response['content'])
```

---

## 总结

### v3.0 核心特性

✅ LLM 驱动的帕累托验证  
✅ 智能趋势分析和模式发现  
✅ 自动生成实验建议  
✅ 支持所有 OpenAI 兼容 LLM  
✅ 完整的分析过程追溯  
✅ 生产就绪的错误处理  

### 下一步

- 🔄 Planner 升级为 LLM 驱动
- 📊 实时可视化 Dashboard
- 🌐 Web UI 界面

---

**版本**: 3.0.0  
**日期**: 2025-12-20  
**核心升级**: AnalysisAgent 使用 LLM 智能分析  
**作者**: AI Scientist Team
