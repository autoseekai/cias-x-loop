# AI Scientist CI (Computation Imaging Scientist)

**AI Scientist CI** represents a significant leap towards autonomous scientific discovery in the field of **Computational Imaging (CI)**. It leverages a sophisticated **multi-agent architecture** orchestrated by **LangGraph** to autonomously plan, execute, analyze, and refine scientific experiments.

Unlike traditional automated machine learning (AutoML) or simple optimization scripts, AI Scientist CI possesses a **closed-loop learning capability**. It can "read" scientific literature to understand constraints, formulate research strategies, execute experiments potentially involving remote hardware, and perform in-depth analysis to generate new scientific insights—all without human intervention.

---

## 🚀 Key Innovation: The "Scientist" Knowledge Loop

This project introduces a **Simplified World Model** design that acts as the system's brain, enabling true "Context-Aware Research":

1.  **📚 Learn (Literature Review)**: The **Learning Agent** (powered by **LlamaIndex**) ingests reference documents (PDFs/MDs). It distills unstructured text into structured **Knowledge Capsules**—containing critical constraints ("Learning rate must be < 1e-3") and best practices.
2.  **🧠 Plan (Strategy Formulation)**: The **Planner Agent** doesn't just guess hyperparameters. It queries the World Model for a **Planning Context**, combining learned literature rules with the results of the best past experiments to propose scientifically grounded experimental designs.
3.  **🛡️ Review (Safety & Validity)**: The **Reviewer Agent** acts as a strict "Reviewer #2". It validates proposed plans against the explicit rules learned in step 1, ensuring no dangerous or theoretically invalid experiments are run on expensive hardware.
4.  **🔬 Execute (Experimentation)**: The **Executor Agent** manages the training of SCI reconstruction networks (supporting local execution or remote API for real optical hardware).
5.  **📈 Analyze (Insight Generation)**: The **Analysis Agent** performs Pareto front analysis and stratified trend analysis. Crucially, it **writes new insights back to the World Model**, allowing the system to "learn from experience" and improve subsequent cycles.

---

## 🛠️ System Architecture

The core of the system is the **Simplified World Model (SQLite)**, which serves as a shared "Context Generator" for all agents.

```mermaid
graph TD
    User[User] --> |Upload Docs & Goal| NodeLearner[Learning Node]

    subgraph "Knowledge Loop"
    NodeLearner --> |Ingest & Index| Learner[Learning Agent]
    Learner --> |Save Rules & Insights| WM[(World Model\nSQLite)]
    end

    subgraph "Experiment Loop (LangGraph)"
    NodeLearner --> NodePlanner[Planner Node]

    WM --> |Get Context\n(Rules + Best Exps)| NodePlanner
    NodePlanner --> |Propose Configs| Planner[Planner Agent]

    Planner --> NodeReviewer[Reviewer Node]
    WM --> |Get Validation Rules| NodeReviewer
    NodeReviewer --> |Approve/Reject| Reviewer[Reviewer Agent]

    Reviewer --> |If Approved| NodeExecutor[Executor Node]
    Reviewer --> |If Rejected| NodePlanner

    NodeExecutor --> |Run Experiments| Executor[Executor Agent]
    Executor --> |Save Results| WM

    NodeExecutor --> NodeAnalyzer[Analyzer Node]
    NodeAnalyzer --> |Fetch Data| Analyzer[Analysis Agent]
    Analyzer --> |Save New Insights| WM

    NodeAnalyzer --> |Next Cycle| NodePlanner
    end
```

### Core Components

1.  **World Model (`src/agents/sci/world_model.py`)**:
    *   **Technology**: SQLite (Pure Python, no external DB required).
    *   **Function**: Stores `knowledge_capsules` (Rules, Insights), `experiments` (Configs, Metrics), and `llm_analyses`.
    *   **Key Capability**: `get_planning_context()` generates a token-optimized prompt summary for the Planner.

2.  **Learning Agent (`src/agents/sci/learner.py`)**:
    *   **Technology**: **LlamaIndex RAG**.
    *   **Function**: Reads scientific papers/documentation and extracts structured rules (e.g., "Use Adam optimizer for non-convex problems").

3.  **Planner Agent (`src/agents/sci/planner.py`)**:
    *   **Function**: Context-aware experiment design. Uses a **Meta-Prompting** strategy (propose strategy first, then configs) to improve quality.

4.  **Reviewer Agent (`src/agents/sci/reviewer.py`)**:
    *   **Function**: Rules-based and LLM-based safety checks. prevents "hallucinated" parameters.

5.  **Analysis Agent (`src/agents/sci/analysis.py`)**:
    *   **Function**: Computes Pareto frontiers (e.g., PSNR vs. Latency) and identifies trends (e.g., "Higher compression requires deeper networks").

---

## 📂 Project Structure

```
├── config/              # Configuration files
├── docs/                # Documentation
├── sci_loop.py          # Main entry point
├── src/
│   ├── agents/
│   │   ├── core/        # Abstract Base Classes (Generic for any Domain)
│   │   │   └── abstract_agents.py
│   │   ├── sci/         # SCI Domain Implementation
│   │   │   ├── learner.py     # LlamaIndex Document Learner
│   │   │   ├── planner.py     # Context-Aware Planner
│   │   │   ├── reviewer.py    # Rule-Based Reviewer
│   │   │   ├── executor.py    # Experiment Executor
│   │   │   ├── analysis.py    # Insight Generator
│   │   │   ├── world_model.py # SQLite Database Manager
│   │   │   └── structures.py  # Pydantic Data Models
│   │   └── base.py
│   ├── core/            # Framework Core
│   │   ├── workflow_graph.py  # LangGraph Topology
│   │   └── state.py           # Global State Schema
│   └── llm/             # LLM Client (OpenAI/LiteLLM)
└── data/                # Default storage for DB and vectors
```

---

## 🚦 Getting Started

### Prerequisites

*   **Python 3.10+** (Tested on 3.12)
*   **OpenAI API Key** (or compatible endpoint)
*   Optional: `uv` for fast package management

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/ai-scientist-ci.git
cd ai-scientist-ci

# Install dependencies
# Using pip:
pip install -r requirements.txt
# OR using uv (recommended):
uv sync
```

### 2. Configuration

Set up your environment variables (create a `.env` file):

```bash
OPENAI_API_KEY=sk-your-key-here
# Optional: Model selection
OPENAI_MODEL_NAME=gpt-4-turbo
```

Check `config/default.yaml` for system settings (budgets, cycle counts).

### 3. Usage

**Basic Run (No learning files, pure exploration):**
```bash
python sci_loop.py --budget 10 --cycles 3
```

**Full "Scientist" Run (With Document Learning):**
Place your reference papers (PDF/MD) in a folder (e.g., `papers/`) and run:
```python
# (In python script or extended CLI - see sci_loop.py)
python sci_loop.py --docs ./papers/ --goal "Maximize reconstruction PSNR for compression ratio > 16"
```

---

## 🌐 Extending to Other Domains

This framework is designed for **General Purpose Automated Science**. To adapt it to a new domain (e.g., **MRI**, **Biology**, **Materials Science**):

1.  **Define Configuration**: Create a new `Pydantic` model for your experiment parameters in a new `structures.py`.
2.  **Implement Agents**: Inherit from the abstract classes in `src/agents/core/abstract_agents.py`.
    *   `class MRIPlanner(AbstractPlannerAgent[MRIConfig]): ...`
    *   `class MRIExecutor(AbstractExecutorAgent[MRIConfig, MRIResult]): ...`
3.  **Reuse Core**: The `LearningAgent`, `WorldModel` (with minor schema tweaks), and `LangGraph` workflow remain largely unchanged!

---

## 🤝 Contributing

We welcome contributions! Please see `CONTRIBUTING.md` for details on how to submit Pull Requests.

## 📄 License

MIT License
