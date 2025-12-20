"""White Agent - A2A Protocol compliant implementation using official SDK."""

import importlib
import json
import logging
import os
from importlib import import_module

import numpy as np
import pandas as pd
import uvicorn
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from a2a.utils import new_agent_text_message

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [WHITE AGENT] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class GreenAgentProxy:
    """Proxy for communicating with the green agent via A2A protocol."""

    def __init__(self, green_agent_url: str):
        self.url = green_agent_url
        self._client = None
        self._card = None

    async def _get_client(self):
        """Lazily initialize the A2A client."""
        if self._client is None:
            import httpx
            from a2a.client import A2ACardResolver, A2AClient

            httpx_client = httpx.AsyncClient(timeout=120.0)
            resolver = A2ACardResolver(httpx_client=httpx_client, base_url=self.url)
            self._card = await resolver.get_agent_card()
            self._client = A2AClient(httpx_client=httpx_client, agent_card=self._card)
        return self._client

    async def _send_command_async(self, action: str, args: dict = None) -> dict:
        """Send a command to the green agent via A2A protocol."""
        import uuid

        from a2a.types import (
            Message,
            MessageSendParams,
            Part,
            Role,
            SendMessageRequest,
            SendMessageSuccessResponse,
            TextPart,
        )

        command_json = json.dumps(
            {"action": action, "args": args or {}}, cls=NumpyEncoder
        )

        logger.info(f">>> SENDING TO GREEN AGENT: action={action}, args={args or {}}")

        client = await self._get_client()

        message_id = uuid.uuid4().hex
        params = MessageSendParams(
            message=Message(
                role=Role.user,
                parts=[Part(root=TextPart(text=command_json))],
                message_id=message_id,
            )
        )
        request_id = uuid.uuid4().hex
        req = SendMessageRequest(id=request_id, params=params)

        response = await client.send_message(request=req)

        # Extract the response text
        res_root = response.root
        if isinstance(res_root, SendMessageSuccessResponse):
            res_result = res_root.result
            # Handle both Message and Task responses
            if hasattr(res_result, 'parts'):
                # It's a Message
                from a2a.utils import get_text_parts
                text_parts = get_text_parts(res_result.parts)
                if text_parts:
                    content = text_parts[0]
                    try:
                        data = json.loads(content)
                        logger.info(f"<<< RECEIVED FROM GREEN AGENT: {json.dumps(data, cls=NumpyEncoder)[:200]}...")
                        return data
                    except json.JSONDecodeError:
                        return {"raw": content}
        
        raise RuntimeError(f"Unexpected response from green agent: {response}")

    def _send_command(self, action: str, args: dict = None) -> dict:
        """Synchronous wrapper for sending commands."""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(self._send_command_async(action, args))

    def describe(self) -> dict:
        return self._send_command("describe")

    def request_dataset(self, ds_id: str, split: str = "train") -> pd.DataFrame:
        json_data = self._send_command("request_dataset", {"ds_id": ds_id, "split": split})
        return pd.read_json(json.dumps(json_data), orient="split")

    def evaluate_predictions(self, y_pred) -> dict:
        return self._send_command("evaluate_predictions", {"y_pred": y_pred})


# Available agent implementations
AGENTS = {
    "react-agent": {
        "module": "ecoagent.white_agents.react_agent",
        "name": "React Agent",
        "description": "Uses React-style reasoning to solve tasks.",
    },
    "plan-execute-agent": {
        "module": "ecoagent.white_agents.plan_execute_agent",
        "name": "Plan-Execute Agent",
        "description": "Two-phase agent that first creates a plan, then executes it step by step.",
    },
    "oracle-agent": {
        "module": "ecoagent.white_agents.oracle_agent",
        "name": "Oracle Agent",
        "description": "Returns the ground truth value directly.",
    },
    "linear-regression": {
        "module": "ecoagent.white_agents.two_feature_regression",
        "name": "Linear Regression Agent",
        "description": "Uses sklearn LinearRegression with median listing price as feature.",
    },
    "naive-last-value": {
        "module": "ecoagent.white_agents.naive_last_value",
        "name": "Naive Last Value Agent",
        "description": "Simple baseline that predicts the last observed training value.",
    },
}

# Default agent to use
DEFAULT_AGENT_ID = os.getenv("AGENT_ID", "oracle-agent")


def prepare_agent_card(url: str, agent_id: str = None) -> AgentCard:
    """Create the agent card for the white agent."""
    if agent_id is None:
        agent_id = DEFAULT_AGENT_ID
    
    agent_info = AGENTS.get(agent_id, AGENTS[DEFAULT_AGENT_ID])
    
    skill = AgentSkill(
        id="housing_prediction",
        name="Housing Price Prediction",
        description=agent_info["description"],
        tags=["white agent", "prediction", "housing", "forecasting"],
        examples=[],
    )
    return AgentCard(
        name=agent_info["name"],
        description=f"White Agent for EcoAgent (A2A). {agent_info['description']}",
        url=url,
        version="1.0.0",
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        capabilities=AgentCapabilities(),
        skills=[skill],
    )


class WhiteAgentExecutor(AgentExecutor):
    """Executor that handles incoming A2A messages for the white agent."""

    def __init__(self, agent_id: str = None):
        self.agent_id = agent_id or DEFAULT_AGENT_ID
        self.green_agent_url = os.getenv("GREEN_AGENT_URL")

    def run_agent(self) -> dict:
        """Run the configured agent and return results."""
        if self.agent_id not in AGENTS:
            raise ValueError(f"Unknown agent: {self.agent_id}")

        if not self.green_agent_url:
            raise RuntimeError("GREEN_AGENT_URL environment variable not set")

        proxy = GreenAgentProxy(self.green_agent_url)
        agent_module = import_module(AGENTS[self.agent_id]["module"])
        importlib.reload(agent_module)
        return agent_module.run(proxy)

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Handle incoming A2A message."""
        user_input = context.get_user_input()
        logger.info(f"Received message: {user_input[:200] if user_input else 'empty'}...")

        try:
            result = self.run_agent()
            result_text = json.dumps(result, cls=NumpyEncoder)
            logger.info(f"Agent completed successfully: {result_text[:200]}...")
        except Exception as e:
            logger.error(f"Agent error: {e}")
            result_text = json.dumps({"error": str(e)})

        # Send response using proper A2A format
        await event_queue.enqueue_event(
            new_agent_text_message(result_text, context_id=context.context_id)
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Handle cancellation request."""
        raise NotImplementedError("Cancel not implemented")


def create_app(agent_id: str = None):
    """Create and configure the A2A application."""
    if agent_id is None:
        agent_id = DEFAULT_AGENT_ID

    # Get the agent URL from environment
    agent_url = os.getenv("AGENT_URL", os.getenv("PUBLIC_URL", "http://localhost:9001"))
    
    # Ensure URL ends with /
    if not agent_url.endswith("/"):
        agent_url = agent_url + "/"

    logger.info(f"Creating white agent '{agent_id}' with URL: {agent_url}")

    # Create agent card
    agent_card = prepare_agent_card(agent_url, agent_id)

    # Create request handler with our executor
    request_handler = DefaultRequestHandler(
        agent_executor=WhiteAgentExecutor(agent_id),
        task_store=InMemoryTaskStore(),
    )

    # Create A2A application
    a2a_app = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )

    return a2a_app.build()


# Create the app instance for uvicorn
app = create_app()


def start_white_agent(agent_id: str = None, host: str = "0.0.0.0", port: int = 9001):
    """Start the white agent server."""
    application = create_app(agent_id)
    uvicorn.run(application, host=host, port=port)


if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("AGENT_PORT", os.getenv("PORT", "9001")))
    agent_id = os.getenv("AGENT_ID", DEFAULT_AGENT_ID)
    start_white_agent(agent_id=agent_id, host=host, port=port)
