"""LangGraph-based voice agent for LiveKit."""

import os
from pathlib import Path

from dotenv import load_dotenv
from livekit.agents import (
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    WorkerOptions,
    cli,
    inference,
    metrics,
    RoomInputOptions,
)
from livekit.plugins import noise_cancellation, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from langgraph_flows import GraphExecutor
from utils.logger import configure_logging, get_logger

logger = get_logger("langgraph-agent")

load_dotenv(".env.local")


def prewarm(proc: JobProcess):
    """Pre-warms VAD model for better performance."""
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    """Main entrypoint for LangGraph-based voice agent."""
    configure_logging(level="INFO", log_format="json")

    ctx.log_context_fields = {"room": ctx.room.name}

    graph_file = os.getenv("LANGGRAPH_CONFIG", "ecommerce.json")
    graph_path = Path(__file__).parent.parent / "configs" / "langgraph" / graph_file

    logger.info("loading_langgraph_config", path=str(graph_path))

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

    try:
        executor = GraphExecutor(str(graph_path), agent_session=session)
        logger.info(
            "langgraph_initialized",
            graph_id=executor.config.id,
            graph_name=executor.config.name,
        )
    except Exception as e:
        logger.error("langgraph_load_failed", error=str(e), path=str(graph_path))
        raise

    from livekit.agents import Agent

    class LangGraphAgent(Agent):
        """Agent wrapper that integrates LangGraph state machine."""

        def __init__(self, graph_executor: GraphExecutor):
            super().__init__(
                instructions=f"You are orchestrating a conversation using LangGraph: {graph_executor.config.name}"
            )
            self.executor = graph_executor
            self.current_state = None

        async def on_enter(self) -> None:
            """Initializes LangGraph execution on agent entry."""
            logger.info("langgraph_agent_entered")

            self.current_state = self.executor.create_initial_state(
                session_id=ctx.room.name,
            )

            self.current_state = await self.executor.execute_step(self.current_state)

            if self.current_state.get("agent_response"):
                await self.session.generate_reply(
                    instructions=self.current_state["agent_response"]
                )
            else:
                await self.session.generate_reply()

    langgraph_agent = LangGraphAgent(executor)

    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def _log_usage_summary():
        summary = usage_collector.get_summary()
        logger.info("usage_summary", summary=summary)

    ctx.add_shutdown_callback(_log_usage_summary)

    await session.start(
        agent=langgraph_agent,
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()

    logger.info("langgraph_agent_started", graph_id=executor.config.id)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
