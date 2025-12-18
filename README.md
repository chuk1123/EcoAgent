# EcoAgent: Resource-Constrained Forecasting

**EcoAgent** is a benchmark and agent system for resource-constrained forecasting tasks. It consists of two main components:
1.  **Green Agent (The Environment)**: Holds the data, defines the task, and evaluates performance based on accuracy and resource usage (budget).
2.  **White Agent (The Solver)**: A modular agent that requests data from the Green Agent to solve the forecasting task under budget constraints.

## 🔗 Repository Information

*   **Branch**: `main`

## 📂 Project Structure

```
EcoAgent/
├── ecoagent/
│   ├── green_agent/        # Green Agent (Task Provider & Evaluator)
│   ├── white_agents/       # White Agent Strategies
│   └── contexts/           # Data contexts (e.g., housing)
├── configs/                # Configuration (API keys, model settings)
├── run.sh       # Script to start Green Agent server
├── run_white.sh            # Script to start White Agent server
├── launcher.py             # Script to trigger the assessment
└── requirements.txt        # Dependencies
```

## 🚀 Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/chuk1123/EcoAgent
    cd EcoAgent
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure API Keys (Optional)**:
    Set your API keys in `configs/config.yaml` and pick which model you want to use.

### Pick task

Pick a target category from the following: Total Housing Units,Occupied Units,Vacant Units,Median Listing Price (US dollars),Total Population,Median Age (years),Homeless Total,Median Household Income (inflation-adjusted US dollars),Percentage of Population at or below the Poverty Level

For the target category you picked (which we'll denote as `TARGET`), also note the corresponding id for the target in `catalog.yaml` (ids are total_housing_units, occupied_units, vacant_units, median_listing_price, population, median_age, homeless_total, median_income, poverty_rate); we'll denoate this target id as `target_id`. For example, if your `TARGET` is Vacant Units, `target_id` will be vacant_units.

Then, in `catalog.yaml`, change the value for `columns` under the id `target` to `["Year", "<TARGET>"]`. For instance, if `TARGET` was Vacant Units, `catalog.yaml` would be:
```yaml
datasets:
  - id: target
    columns: ["Year", "Vacant Units"]
...
```

Then, change `meta.yaml` so that `target` is set to `TARGET` and change `leakage_guard_test_block` to list `target` and `target_id`. For example, if `TARGET` is Vacant Units (and `target_id` is vacant_units), meta.yaml would be the following:
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

### Full Client-Server Deployment (AgentBeats Style)
To run the agents as separate services (mimicking the AgentBeats deployment):

#### 1. Start the Green Agent (Server)
The Green Agent serves the task and evaluates submissions.
```bash
./run_green_ctrl.sh
```
*Runs on port 8000 by default.*

#### 2. Start the White Agent (Client)
In a new terminal, start the White Agent. You can select the strategy (what agent to use) by changing AGENT_ID in `white_agents/main.py`.
```bash
./run_white.sh
```
*Runs on port 8001 by default.*

#### 3. Trigger the Assessment
In a third terminal, send the "Start" signal to the White Agent:
```bash
python launcher.py
```
This will cause the White Agent to contact the Green Agent, solve the task, and print the final report.
