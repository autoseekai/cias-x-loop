# AI Scientist for SCI v3.0

An AI Scientist system for SCI domain based on Kosmos and CIAS-X algorithms.

## Key Features

- ✅ **LLM-Driven Pareto Verification**: Uses LLM to verify Pareto front reasonableness
- ✅ **Intelligent Trend Analysis**: LLM automatically discovers hidden patterns
- ✅ **Experiment Recommendations**: LLM provides next experiment suggestions
- ✅ **OpenAI Compatible**: Supports all OpenAI API format LLMs
- ✅ **Full Traceability**: Records all LLM analysis processes

## Project Structure

```
sci-ai-scientist/
├── src/
│   └── sci_scientist/
│       ├── __init__.py           # Package entry
│       ├── core/
│       │   ├── data_structures.py # Data structures
│       │   └── scientist.py       # AI Scientist main loop
│       ├── agents/
│       │   ├── planner.py         # Planner agent
│       │   ├── executor.py        # Executor agent
│       │   └── analysis.py        # Analysis agent (LLM)
│       ├── models/
│       │   └── world_model.py     # World model
│       └── llm/
│           └── client.py          # LLM client
├── config/
│   └── default.yaml               # Default config
├── main.py                        # Entry point
├── pyproject.toml                 # Project config
└── README.md
```

## Installation

```bash
pip install -e .
```

Or install dependencies:

```bash
pip install numpy loguru requests tenacity openai scipy pyyaml
```

## Usage

### Mock Mode (Testing)

```bash
python main.py --mock
```

### Real Mode

```bash
export OPENAI_API_KEY="sk-..."
python main.py --config config/default.yaml
```

### Command Line Arguments

| Argument | Description |
|----------|-------------|
| `--config` | Config file path |
| `--mock` | Enable mock mode |
| `--budget` | Experiment budget |
| `--cycles` | Maximum cycles |

## Supported LLM Services

| Service | Base URL | Model |
|---------|----------|-------|
| OpenAI | https://api.openai.com/v1 | gpt-4-turbo-preview |
| DeepSeek | https://api.deepseek.com/v1 | deepseek-chat |
| Qwen | https://dashscope.aliyuncs.com/compatible-mode/v1 | qwen-turbo |
| Ollama | http://localhost:11434/v1 | llama2, mistral |

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black src/
isort src/

# Start Mock Service
.venv/bin/uvicorn mock_service:app --port 8000

# In another terminal, run AI Scientist (using remote service)
# Set mock_mode: false in config/default.yaml
.venv/bin/python main.py --budget 10 --cycles 3
```

## License

MIT
