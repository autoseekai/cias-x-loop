# Orchestration System Design: The "Research Director" Architecture

## 1. Vision
Transform the current linear execution loop (`Summarize -> Plan -> Execute -> Analyze`) into a **Dynamic, Multi-Agent Collaboration System**.

The goal is to move from a "Scripted Procedure" to an **"Autonomous Research Laboratory"** where agents coordinate to solve problems, negotiate on plans, and react to results in real-time.

## 2. Core Concepts

### 2.1 The "Research Director" (Orchestrator)
Instead of a simple loop, we introduce a central brain called the **Research Director**.
*   **Role**: Like a Principal Investigator (PI) managing a lab.
*   **Responsibility**: manages high-level state, allocates budget, resolves deadlocks, and decides when to pivot research direction.
*   **Implementation**: A State Machine driven manager.

### 2.2 The "Message Bus" (Communication Layer)
Agents should not call each other's methods directly (tight coupling). Instead, they utilize a structured message passing system.
*   **Message Types**: `TASK_REQUEST`, `TASK_RESULT`, `CRITIQUE`, `ALERT`, `STATUS_UPDATE`.
*   **Channel**: A central event bus where agents publish events and subscribe to relevant ones.

## 3. Architecture Design

```mermaid
graph TD
    Director[Research Director] -- Manages --> Bus[Message Bus]

    subgraph "Planning Cluster"
        Planner[Planner Agent]
        Reviewer[Reviewer Agent]
    end

    subgraph "Action Cluster"
        Executor[Executor Agent]
    end

    subgraph "Cognition Cluster"
        Analyzer[Analysis Agent]
        WorldModel[World Model (Shared Memory)]
    end

    Planner <-->|Negotiate| Reviewer
    Reviewer -->|Approved Plan| Bus
    Bus -->|Dispatch Task| Executor
    Executor -->|Results| WorldModel
    Executor -->|Completion Event| Bus
    Bus -->|Trigger| Analyzer
    Analyzer -->|Insight| Director
```

## 4. Coordination Protocols (Self-Coordination)

We design specific **"Conversational Protocols"** for agent interaction, allowing them to coordinate locally without micro-management from the Director.

### 4.1 The "Peer Review" Protocol (Planner <-> Reviewer)
This replaces the single-pass planning with a negotiation loop.
1.  **Planner** posts `PROPOSAL` to the bus (or directly to Reviewer).
2.  **Reviewer** analyzes and replies with `CRITIQUE` or `APPROVAL`.
    *   *If Critique*: **Planner** reads feedback -> modifies plan -> reposts `PROPOSAL`.
    *   *If Approval*: **Director** stamps it as `AUTHORIZED`.
3.  **Fail-safe**: If negotiation loops > 3 times, **Director** intervenes (e.g., forces the safest plan or lowers strictness).

### 4.2 The "Early Stopping" Protocol (Executor <-> Analyzer)
Agents can talk during execution.
1.  **Executor** publishes `INTERIM_METRIC` (e.g., Epoch 10/50, PSNR=25dB).
2.  **Analyzer** (monitoring bus) detects `INTERIM_METRIC`.
3.  **Analyzer** evaluates: "This is already worse than baseline, no hope of recovery."
4.  **Analyzer** sends `STOP_REQUEST` to **Executor**.
5.  **Executor** acknowledges, aborts job, and frees up GPU resources.

## 5. System Components

### 5.1 The `Agent` Base Class
Standardize interfaces so the Director can treat them uniformly.

```python
class Agent(ABC):
    def __init__(self, name: str, message_bus: MessageBus):
        self.inbox = Queue()

    async def process_message(self, message: Message):
        """Handle incoming messages"""

    async def publish(self, topic: str, payload: Any):
        """Send message to bus"""
```

### 5.2 The `ResearchDirector` State Machine
The Director maintains the "Research Phase":
*   **INIT**: Loading context.
*   **STRATEGIZING**: High-level goal setting (Broad exploration vs. Deep optimization).
*   **PLANNING**: Waiting for Planner/Reviewer consensus.
*   **EXECUTING**: Monitoring Executor resource usage.
*   **REFLECTING**: Waiting for Analysis.
*   **CONCLUDED**: Budget exhausted or goal met.

## 6. Implementation Stages

### Phase 1: The "Boardroom" (Centralized)
*   Keep the `AIScientist` loop but refactor it into a `Director` class.
*   The `Director` explicitly calls `Planner`, then `Reviewer`, handles the loop logic internally.
*   **Benefit**: Easy to implement, immediate gain in robustness.

### Phase 2: The "Chat Room" (Event Driven)
*   Introduce the `MessageBus`.
*   Agents listen for events.
*   Example: `Executor` doesn't run because it was "called", it runs because it saw a `NEW_APPROVED_PLAN` event.
*   **Benefit**: Highly decoupled. You can add a `LoggerAgent` or `DashboardAgent` without changing core logic.

## 7. Example Workflow Scenario

1.  **Director** starts Cycle 1. Changes state to `STRATEGIZING`.
2.  **Director** asks **Analyzer**: "What is our best direction?"
3.  **Analyzer** replies: "Focus on increasing `num_stages`."
4.  **Director** sends directive to **Planner**: "Generate ideas. Constraint: `num_stages` > 5."
5.  **Planner** creates 5 drafts. Sends to **Reviewer**.
6.  **Reviewer** rejects 2 (unsafe). Approves 3.
7.  **Planner** acknowledges, sends 3 `Final Plans` to Director.
8.  **Director** forwards to **Executor**.
9.  **Executor** runs them. One fails early; Executor reports `FAILURE`. Two succeed; Executor reports `SUCCESS`.
10. **WorldModel** updates automatically upon `SUCCESS` events.
11. **Analyzer** sees `CYCLE_COMPLETE` and generates report.

## 8. Summary
This design moves `sci-ai-scientist` from a linear script to a **responsive, resilient multi-agent system**. The key innovation is the **Review Loop** and the **Director's ability to intervene**, making the system truly autonomous.
