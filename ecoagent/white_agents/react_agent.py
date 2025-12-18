import json
import os
from pathlib import Path
from typing import Optional

import yaml
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
        # Parse the JSON string to get the list of predictions
        predictions = json.loads(predictions_json)
        if not isinstance(predictions, list):
            raise ValueError("predictions_json must be a JSON array")
        # Convert to list of floats
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


def run(proxy):
    """Main entry point - runs the ReAct agent to solve the forecasting task.

    Loads API keys and model settings from configs/config.yaml
    """
    global _proxy
    _proxy = proxy

    # Load config
    config = _load_config()
    agent_config = config.get("agents", {}).get("white_agent", {})
    provider = agent_config.get("provider", "google")
    model_name = agent_config.get("model_name", "gemini-2.0-flash-exp")
    api_keys = config.get("api_keys", {})

    # Initialize the LLM based on provider
    if provider == "google":
        api_key = api_keys.get("google") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "Google API key not found. Set it in configs/config.yaml or GOOGLE_API_KEY env var"
            )
        llm = ChatGoogleGenerativeAI(
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
        llm = ChatOpenAI(
            model=model_name,
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}. Use 'google' or 'openai'")

    # Create the ReAct agent with our tools
    tools = [describe_task, get_dataset, submit_predictions]
    agent = create_react_agent(llm, tools)

    # System prompt for the agent
    system_message = """You are a forecasting agent. Your task is to predict future values of a target variable.

IMPORTANT CONSTRAINTS:
1. You have a LIMITED BUDGET of datasets you can request. Check the budget with describe_task first.
2. You get an EFFICIENCY BONUS for using fewer datasets, so be strategic.
3. Some datasets are BLOCKED on the test split to prevent data leakage.

STRATEGY:
1. First, call describe_task to understand what you need to predict and your constraints.
2. Request the training data for the target variable to understand the historical values.
3. Optionally request ONE additional predictor dataset if you think it will help.
4. Analyze the data and make your prediction(s).
5. Submit your predictions using submit_predictions with a JSON string like "[1234.5]" for a single prediction or "[100.0, 200.0]" for multiple.

Be efficient - you're scored on both accuracy AND efficiency (using fewer datasets)!"""

    # Run the agent
    result = agent.invoke(
        {
            "messages": [
                {"role": "system", "content": system_message},
                {
                    "role": "user",
                    "content": "Complete the forecasting task. Start by describing the task, then gather data, make predictions, and submit them.",
                },
            ]
        }
    )

    # Extract the final result from the last message
    messages = result.get("messages", [])

    # Find the evaluation result in tool outputs
    for msg in reversed(messages):
        if hasattr(msg, "content") and isinstance(msg.content, str):
            try:
                parsed = json.loads(msg.content)
                if "final_score" in parsed:
                    return parsed
            except:
                pass

    # If no structured result found, return the last message content
    if messages:
        last_msg = messages[-1]
        return {"status": "completed", "last_message": str(last_msg.content)}

    return {"status": "error", "reason": "No result from agent"}
