# EcoAgent: Resource-Constrained Forecasting Benchmark

**EcoAgent** is a benchmark for evaluating AI agents on resource-constrained time-series forecasting tasks using the A2A (Agent-to-Agent) protocol. It consists of two main components:

1. **Green Agent (The Environment)**: Holds the data, defines the task, and evaluates performance based on accuracy and resource usage (budget).
2. **White Agent (The Solver)**: A modular agent that requests data from the Green Agent to solve the forecasting task under budget constraints.

## Repository Information

- **Branch**: `main`

## Project Structure

```
EcoAgent/
├── ecoagent/
│   ├── green_agent/           # Green Agent (Task Provider & Evaluator)
│   │   ├── green_agent.py     # Core evaluation logic
│   │   └── main.py            # FastAPI server with A2A support
│   ├── white_agents/          # White Agent Strategies
│   │   ├── main.py            # FastAPI server with A2A support
│   │   ├── naive_last_value.py    # Simple baseline (no ML)
│   │   ├── two_feature_regression.py  # Linear regression (no LLM)
│   │   ├── plan_execute_agent.py  # Plan-Execute agent (requires LLM)
│   │   └── react_agent.py     # ReAct-based agent (requires LLM)
│   └── contexts/housing/      # Data contexts
│       ├── train.csv          # Training data
│       ├── test.csv           # Test data
│       ├── catalog.yaml       # Dataset catalog
│       └── meta.yaml          # Task configuration
├── configs/                   # Configuration (API keys, model settings)
├── deploy_green_full.sh       # Deploy green agent with AgentBeats controller
├── deploy_white_full.sh       # Deploy white agents with AgentBeats controller
├── run.sh                     # Script to start Green Agent server
├── run_white.sh               # Script to start White Agent server
├── launcher.py                # Script to trigger the assessment
└── requirements.txt           # Dependencies
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/chuk1123/EcoAgent
   cd EcoAgent
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure API Keys (Optional)**:
   Set your API keys in `configs/config.yaml` if using LLM-based agents.

## Task Configuration

### Pick a Target Category

Select a target category from the following:
- Total Housing Units
- Occupied Units
- Vacant Units
- Median Listing Price (US dollars)
- Total Population
- Median Age (years)
- Homeless Total
- Median Household Income (inflation-adjusted US dollars)
- Percentage of Population at or below the Poverty Level

For your chosen target (e.g., `Vacant Units`), find the corresponding ID in `catalog.yaml`:
- `total_housing_units`, `occupied_units`, `vacant_units`, `median_listing_price`, `population`, `median_age`, `homeless_total`, `median_income`, `poverty_rate`

Then update `catalog.yaml`:
```yaml
datasets:
  - id: target
    columns: ["Year", "Vacant Units"]
...
```

And update `meta.yaml`:
```yaml
context_id: housing-trends
target: Vacant Units
horizon: 1
budget:
  max_datasets: 2
scoring:
  metric: rmse
  mode: bonus
  alpha_accuracy: 0.7
  beta_efficiency: 0.3 
leakage_guard_test_block:
  - target
  - vacant_units
```

## Running Locally (Without AgentBeats)

### 1. Start the Green Agent (Evaluator)
```bash
./run.sh
# Or manually:
python -m uvicorn ecoagent.green_agent.main:app --host 0.0.0.0 --port 8000
```
*Runs on port 8000 by default.*

### 2. Start the White Agent (Participant)
In a new terminal:
```bash
./run_white.sh
# Or manually:
export GREEN_AGENT_URL=http://localhost:8000
python -m uvicorn ecoagent.white_agents.main:app --host 0.0.0.0 --port 8001
```
*Runs on port 8001 by default.*

### 3. Trigger the Assessment
In a third terminal:
```bash
python launcher.py
```
This causes the White Agent to contact the Green Agent, solve the task, and print the final report.

## Running on AgentBeats Platform

### Prerequisites

- Install `cloudflared` for Cloudflare tunnels: `brew install cloudflared` (macOS)
- Install AgentBeats controller: `pip install agentbeats`
- Have an AgentBeats account at https://agentbeats.org

### Deploy Green Agent (Controller Mode)

```bash
./deploy_green_full.sh
```

This will:
1. Start a Cloudflare tunnel
2. Launch the AgentBeats controller on port 8010
3. Output a public URL to register in AgentBeats

### Deploy White Agent (Controller Mode)

```bash
./deploy_white_full.sh
```

This will:
1. Start a Cloudflare tunnel
2. Launch the AgentBeats controller on port 8011
3. Deploy all white agents (Linear Regression, Naive Last Value, Plan-Execute, ReAct)
4. Output a public URL to register in AgentBeats

### Register in AgentBeats

1. Go to AgentBeats platform
2. Add your controller URLs as agents
3. Create an assessment linking green and white agents
4. Run the assessment

## Testing Green Agent Evaluation

### Test with Different Agents

```bash
# Test naive last value agent
python -c "
from ecoagent.green_agent.green_agent import GreenAgent
from ecoagent.white_agents.naive_last_value import run

ga = GreenAgent('ecoagent/contexts/housing')
result = run(ga)
print(f'Naive Last Value Result: {result}')
"

# Test linear regression agent
python -c "
from ecoagent.green_agent.green_agent import GreenAgent
from ecoagent.white_agents.two_feature_regression import run

ga = GreenAgent('ecoagent/contexts/housing')
result = run(ga)
print(f'Linear Regression Result: {result}')
"
```

### Run Full Demo Evaluation

```bash
python -m ecoagent.demo
```

## Available White Agents

| Agent ID | Description | Requires LLM |
|----------|-------------|--------------|
| `linear-regression` | Uses sklearn LinearRegression with feature selection | No |
| `naive-last-value` | Predicts the last observed training value | No |
| `react-agent` | ReAct-based reasoning agent | Yes (OpenAI) |
| `plan-execute-agent` | Two-phase agent: plan then execute | Yes (OpenAI) |

## API Reference

### Green Agent Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | POST | A2A JSON-RPC message handler |
| `/v1/chat/completions` | POST | OpenAI-compatible chat endpoint |
| `/.well-known/agent.json` | GET | A2A agent card |
| `/health` | GET | Health check |

### Green Agent Commands (via chat)

```json
// Describe the task
{"action": "describe"}

// Request a dataset (costs budget)
{"action": "request_dataset", "args": {"ds_id": "target", "split": "train"}}

// Submit predictions for evaluation
{"action": "evaluate_predictions", "args": {"y_pred": [100, 101, 102]}}
```

### White Agent Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | POST | A2A JSON-RPC message handler |
| `/v1/chat/completions` | POST | Runs default agent |
| `/agents/{id}/v1/chat/completions` | POST | Runs specific agent |
| `/.well-known/agent.json` | GET | A2A agent card |

## Troubleshooting

### Port Already in Use
```bash
pkill -9 -f uvicorn
pkill -9 cloudflared
```

### AgentBeats Controller Issues
```bash
# Clean agent state
rm -rf .state/
```

### Cloudflare Tunnel Drops
Tunnels are temporary. Restart deployment scripts to get new URLs.

## License

MIT License
