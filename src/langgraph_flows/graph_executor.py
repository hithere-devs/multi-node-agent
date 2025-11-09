"""Graph executor for running LangGraph-based voice agents."""

from typing import Any, Dict, Optional
from uuid import uuid4

from .graph_schema import AgentState
from .graph_builder import VoiceGraphBuilder, load_graph_config_from_json
from utils.logger import get_logger

logger = get_logger("langgraph-executor")


class GraphExecutor:
    """Runs LangGraph state machines with LiveKit integration."""

    def __init__(self, config_path: str, agent_session: Any = None):
        self.config = load_graph_config_from_json(config_path)
        self.agent_session = agent_session

        builder = VoiceGraphBuilder(self.config, agent_session)
        self.graph = builder.build()

        logger.info(
            "graph_executor_initialized",
            graph_id=self.config.id,
            entry_node=self.config.entry_node,
        )

    def create_initial_state(
        self, session_id: Optional[str] = None, user_id: Optional[str] = None, **kwargs
    ) -> AgentState:
        """Creates initial state for graph execution."""
        state: AgentState = {
            "current_node": self.config.entry_node,
            "previous_node": None,
            "node_history": [],
            "messages": [],
            "user_input": None,
            "agent_response": None,
            "session_id": session_id or str(uuid4()),
            "user_id": user_id,
            "session_data": {},
            "next_action": None,
            "should_continue": True,
            "is_complete": False,
            "error": None,
            "tool_results": {},
            "pending_tools": [],
            "chat_context": [],
            "preserve_context": False,
        }

        state.update(kwargs)

        return state

    async def execute_step(
        self,
        state: AgentState,
        thread_id: Optional[str] = None,
    ) -> AgentState:
        """Executes a single step in the graph."""
        config = {"configurable": {"thread_id": thread_id or state["session_id"]}}

        try:
            result = self.graph.invoke(state, config)

            logger.info(
                "graph_step_executed",
                current_node=result.get("current_node"),
                should_continue=result.get("should_continue"),
            )

            return result

        except Exception as e:
            logger.error("graph_execution_error", error=str(e))
            state["error"] = str(e)
            state["is_complete"] = True
            return state

    async def execute_full(
        self,
        initial_state: AgentState,
        max_steps: int = 100,
    ) -> AgentState:
        """Execute the graph until completion or max steps.

        Args:
            initial_state: Starting state
            max_steps: Maximum number of steps to execute

        Returns:
            Final agent state
        """
        state = initial_state
        steps = 0

        logger.info("starting_graph_execution", session_id=state["session_id"])

        while state.get("should_continue", True) and steps < max_steps:
            state = await self.execute_step(state)
            steps += 1

            if state.get("is_complete"):
                logger.info(
                    "graph_execution_complete",
                    session_id=state["session_id"],
                    steps=steps,
                )
                break

            if state.get("error"):
                logger.error(
                    "graph_execution_failed",
                    error=state["error"],
                    steps=steps,
                )
                break

        if steps >= max_steps:
            logger.warning(
                "graph_max_steps_reached",
                session_id=state["session_id"],
                max_steps=max_steps,
            )

        return state

    def stream_execution(
        self,
        initial_state: AgentState,
        thread_id: Optional[str] = None,
    ):
        """Stream graph execution step by step.

        Args:
            initial_state: Starting state
            thread_id: Optional thread ID for checkpointing

        Yields:
            State after each step
        """
        config = {
            "configurable": {"thread_id": thread_id or initial_state["session_id"]}
        }

        logger.info("streaming_graph_execution", session_id=initial_state["session_id"])

        try:
            for state in self.graph.stream(initial_state, config):
                logger.info(
                    "graph_step_streamed",
                    current_node=state.get("current_node"),
                )
                yield state

                if state.get("is_complete") or state.get("error"):
                    break

        except Exception as e:
            logger.error("graph_streaming_error", error=str(e))
            yield {
                **initial_state,
                "error": str(e),
                "is_complete": True,
            }

    def get_state_history(self, thread_id: str) -> list:
        """Get execution history for a thread.

        Args:
            thread_id: Thread/session ID

        Returns:
            List of state snapshots
        """
        try:
            config = {"configurable": {"thread_id": thread_id}}
            history = list(self.graph.get_state_history(config))
            logger.info(
                "retrieved_state_history", thread_id=thread_id, count=len(history)
            )
            return history
        except Exception as e:
            logger.error("history_retrieval_error", error=str(e))
            return []

    def visualize(self, output_path: Optional[str] = None) -> str:
        """Generate visualization of the graph.

        Args:
            output_path: Optional path to save visualization

        Returns:
            Mermaid diagram string
        """
        builder = VoiceGraphBuilder(self.config, self.agent_session)
        return builder.visualize(output_path)
