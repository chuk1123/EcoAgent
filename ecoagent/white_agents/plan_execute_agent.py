import json
import os
from pathlib import Path
from typing import List

import yaml
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

# Global proxy reference (set during run())
_proxy = None


@tool
def describe_task() -> str:
    """Get information about the forecasting task including target variable,
    forecast horizon, dataset budget, and available datasets in the catalog."""
    info = _proxy.describe()
    return json.dumps(info, indent=2)


@tool
def get_dataset(dataset_id: str, split: str = "train") -> str:
    """Request a dataset from the Green Agent. Each unique dataset counts toward your budget.

    Args:
        dataset_id: The ID of the dataset to request (e.g., 'target', 'median_listing_price', 'population')
        split: Either 'train' or 'test'. Note: some datasets are blocked on test split to prevent leakage.

    Returns:
        The dataset as a JSON string with columns and data.
    """
    try:
        df = _proxy.request_dataset(dataset_id, split)
        return df.to_json(orient="split")
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def submit_predictions(predictions_json: str) -> str:
    """Submit your predictions to be evaluated by the Green Agent.

    Args:
        predictions_json: A JSON string containing a list of predicted values for the forecast horizon.
                          Example: "[1234.5]" or "[100.0, 200.0]"

    Returns:
        Evaluation results including RMSE, accuracy score, efficiency bonus, and final score.
    """
    try:
        predictions = json.loads(predictions_json)
        if not isinstance(predictions, list):
            raise ValueError("predictions_json must be a JSON array")
        predictions = [float(p) for p in predictions]
        result = _proxy.evaluate_predictions(predictions)
        return json.dumps(result, indent=2)
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Invalid JSON format: {str(e)}"})
    except Exception as e:
        return json.dumps({"error": str(e)})


def _load_config():
    """Load configuration from configs/config.yaml"""
    config_path = Path(__file__).parent.parent.parent / "configs" / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def _get_llm(config):
    """Initialize the LLM based on config settings."""
    agent_config = config.get("agents", {}).get("white_agent", {})
    provider = agent_config.get("provider", "google")
    model_name = agent_config.get("model_name", "gemini-2.0-flash-exp")
    api_keys = config.get("api_keys", {})

    if provider == "google":
        api_key = api_keys.get("google") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "Google API key not found. Set it in configs/config.yaml or GOOGLE_API_KEY env var"
            )
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0,
        )
    elif provider == "openai":
        api_key = api_keys.get("openai") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key not found. Set it in configs/config.yaml or OPENAI_API_KEY env var"
            )
        os.environ["OPENAI_API_KEY"] = api_key
        return ChatOpenAI(model=model_name, temperature=0)
    else:
        raise ValueError(f"Unsupported provider: {provider}. Use 'google' or 'openai'")


def _create_plan(llm, task_description: str) -> List[str]:
    """Use the LLM to create a plan for solving the forecasting task."""
    planner_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a planning agent for a forecasting task. Your job is to create a step-by-step plan.

The task has these constraints:
- Limited dataset budget (usually 2 datasets max)
- Efficiency bonus for using fewer datasets (30% of score)
- Accuracy accounts for 70% of score
- Some datasets are blocked on test split to prevent leakage

Available tools:
1. describe_task - Get task info, budget, and available datasets
2. get_dataset - Request a dataset (counts against budget)
3. submit_predictions - Submit final predictions

Create a numbered plan with 4-6 concrete steps. Each step should be a single action.
Output ONLY the numbered list, no other text.""",
            ),
            ("user", "Create a plan to complete this forecasting task: {task}"),
        ]
    )

    chain = planner_prompt | llm
    response = chain.invoke({"task": task_description})

    # Parse the plan from the response
    plan_text = response.content if hasattr(response, "content") else str(response)

    steps = []
    for line in plan_text.strip().split("\n"):
        line = line.strip()
        if line and (line[0].isdigit() or line.startswith("-")):
            step = line.lstrip("0123456789.-) ").strip()
            if step:
                steps.append(step)

    # Fallback plan if parsing fails
    if not steps:
        steps = [
            "Call describe_task to understand the forecasting requirements",
            "Request the target training dataset to analyze historical values",
            "Analyze the data to identify trends and patterns",
            "Generate predictions based on the analysis",
            "Submit predictions using submit_predictions",
        ]

    return steps


def _execute_plan(llm, tools, plan: List[str]) -> dict:
    """Execute the plan using a ReAct-style executor."""
    executor = create_react_agent(llm, tools)

    plan_text = "\n".join([f"{i+1}. {step}" for i, step in enumerate(plan)])

    executor_prompt = f"""You are executing a pre-defined plan for a forecasting task.

YOUR PLAN:
{plan_text}

INSTRUCTIONS:
- Execute each step in order
- Use the available tools to complete each step
- After completing all steps, make sure to submit your predictions
- Be efficient with dataset requests to maximize your efficiency bonus

Begin executing the plan now. Start with step 1."""

    result = executor.invoke(
        {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a forecasting agent executing a pre-defined plan. Follow the plan step by step.",
                },
                {"role": "user", "content": executor_prompt},
            ]
        }
    )

    return result


def run(proxy):
    """Main entry point - runs the Plan-and-Execute agent to solve the forecasting task.

    This agent operates in two phases:
    1. Planning: Creates a complete plan of steps
    2. Execution: Executes each step using tool calls

    Loads API keys and model settings from configs/config.yaml
    """
    global _proxy
    _proxy = proxy

    # Load config and initialize LLM
    config = _load_config()
    llm = _get_llm(config)
    tools = [describe_task, get_dataset, submit_predictions]

    # Phase 1: Planning
    task_description = """Predict future housing market values. You have a limited budget of datasets 
    you can request. You are scored on both accuracy (70%) and efficiency (30% bonus for using fewer datasets).
    The target variable's test data is blocked to prevent leakage."""

    plan = _create_plan(llm, task_description)

    # Phase 2: Execution
    result = _execute_plan(llm, tools, plan)

    # Extract the final result from the messages
    messages = result.get("messages", [])

    # Find the evaluation result in tool outputs
    for msg in reversed(messages):
        if hasattr(msg, "content") and isinstance(msg.content, str):
            try:
                parsed = json.loads(msg.content)
                if "final_score" in parsed:
                    parsed["agent_type"] = "plan_and_execute"
                    parsed["plan"] = plan
                    return parsed
            except:
                pass

    # If no structured result found, return the last message content
    if messages:
        last_msg = messages[-1]
        return {
            "status": "completed",
            "agent_type": "plan_and_execute",
            "plan": plan,
            "last_message": str(last_msg.content),
        }

    return {"status": "error", "reason": "No result from agent"}
