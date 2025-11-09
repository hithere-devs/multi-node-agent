"""Node-based workflow agent implementation for LiveKit."""

import os
from pathlib import Path

from dotenv import load_dotenv
from livekit.agents import (
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

from workflows import initialize_orchestrator, load_workflow_from_json
from utils.logger import configure_logging, get_logger

logger = get_logger("workflow-agent")

load_dotenv(".env.local")


def prewarm(proc: JobProcess):
    """Pre-warm VAD model for better performance."""
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    """Main entrypoint for node-based workflow agent."""
    configure_logging(level="INFO", log_format="json")

    ctx.log_context_fields = {"room": ctx.room.name}

    # Load workflow configuration
    workflow_file = os.getenv("WORKFLOW_CONFIG", "restaurant_ordering.json")
    workflow_path = (
        Path(__file__).parent.parent / "configs" / "workflows" / workflow_file
    )

    try:
        workflow_config = load_workflow_from_json(str(workflow_path))
        logger.info(
            "workflow_loaded",
            workflow_id=workflow_config.id,
            workflow_name=workflow_config.name,
        )
    except Exception as e:
        logger.error("workflow_load_failed", error=str(e), path=str(workflow_path))
        raise

    # Initialize workflow orchestrator
    orchestrator = initialize_orchestrator(workflow_config)

    # Get the entry node agent
    entry_agent = orchestrator.get_entry_agent()

    logger.info(
        "starting_workflow",
        entry_node=workflow_config.entry_node_id,
    )

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

    def _log_usage_summary():
        summary = usage_collector.get_summary()
        logger.info("usage_summary", summary=summary)

    ctx.add_shutdown_callback(_log_usage_summary)

    # Start the session with the entry agent
    await session.start(
        agent=entry_agent,
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Connect to room
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
