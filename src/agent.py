"""Multi-nodal agent implementation integrated with LiveKit."""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    cli,
    inference,
    metrics,
)
from livekit.plugins import noise_cancellation, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from agents.config_store import ConfigStore
from agents.multi_node_router import MultiNodeRouter
from agents.node_executor import NodeExecutor
from agents.config_models import MultiNodeAgentConfig
from utils.logger import configure_logging, get_logger

logger = get_logger("multi-node-agent")

load_dotenv(".env.local")

# Global instances
_router: Optional[MultiNodeRouter] = None
_executor: Optional[NodeExecutor] = None


class MultiNodeAgent(Agent):
    """Multi-nodal conversational agent with dynamic configuration."""

    def __init__(
        self,
        router: MultiNodeRouter,
        executor: NodeExecutor,
        config: MultiNodeAgentConfig,
    ) -> None:
        """Initialize multi-nodal agent.

        Args:
            router: Multi-node router for managing conversation flow
            executor: Node executor for running node logic
            config: Agent configuration with domain-specific settings
        """
        instructions = (
            config.agentInstructions
            or """You are a professional assistant.
            You guide users through their needs with empathy and professionalism.
            Keep responses concise and helpful."""
        )

        super().__init__(instructions=instructions)
        self.router = router
        self.executor = executor
        self.config = config


def prewarm(proc: JobProcess):
    """Pre-warm VAD model for better performance."""
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    """Main entrypoint for multi-nodal agent."""
    configure_logging(level="ERROR", log_format="json")

    ctx.log_context_fields = {"room": ctx.room.name}

    # Load multi-nodal configuration (dynamic, can be any domain)
    config_file = os.getenv("AGENT_CONFIG_FILE", "realestate_multinode.json")
    config_path = Path(__file__).parent.parent / "configs" / config_file

    try:
        config_store = ConfigStore()
        multi_node_config = config_store.load_multinode_config(str(config_path))
    except Exception as e:
        logger.error("config_load_failed", error=str(e))
        raise

    # Initialize LLM provider
    if os.getenv("OPENAI_API_KEY"):
        from agents.llm_provider import OpenAIProvider

        llm_provider = OpenAIProvider(model="gpt-4-turbo")
    else:
        from agents.llm_provider import MockLLMProvider

        mock_response = f"I'm {multi_node_config.name}. How can I assist you?"
        llm_provider = MockLLMProvider(mock_response)

    # Initialize router and executor
    global _router, _executor
    _router = MultiNodeRouter(multi_node_config, llm_provider=llm_provider)
    _executor = NodeExecutor(llm_provider)

    # Set up voice AI pipeline
    session = AgentSession(
        stt=inference.STT(model="assemblyai/universal-streaming", language="en"),
        llm=inference.LLM(model="openai/gpt-4.1-mini"),
        tts=inference.TTS(
            model="cartesia/sonic-3", voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    # Setup metrics collection
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    ctx.add_shutdown_callback(lambda: usage_collector.get_summary())

    # Start the session
    await session.start(
        agent=MultiNodeAgent(
            router=_router, executor=_executor, config=multi_node_config
        ),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Connect to room
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
