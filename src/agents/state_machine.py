"""State machine engine for multi-prompt agent orchestration."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agents.config_models import CustomerConfig, StateConfig, TransitionConfig
from agents.llm_provider import LLMProvider
from agents.prompt_renderer import PromptRenderer
from agents.tool_invoker import ToolInvoker
from utils.expr_evaluator import ExpressionEvaluator
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SessionContext:
    """Tracks session state including variables, history, and tool results."""

    session_id: str
    customer_id: str
    agent_id: str
    current_state: str
    variables: Dict[str, Any] = field(default_factory=dict)
    history: List[Dict[str, Any]] = field(default_factory=list)
    tool_results: Dict[str, Any] = field(default_factory=dict)

    def add_to_history(self, entry: Dict[str, Any]) -> None:
        self.history.append(entry)

    def set_variable(self, key: str, value: Any) -> None:
        self.variables[key] = value

    def get_variable(self, key: str, default: Any = None) -> Any:
        return self.variables.get(key, default)


class StateMachineEngine:
    """Orchestrates multi-prompt agent workflows with state transitions."""

    def __init__(
        self,
        config: CustomerConfig,
        renderer: PromptRenderer,
        llm: LLMProvider,
        tools: ToolInvoker,
    ) -> None:
        self.config = config
        self.renderer = renderer
        self.llm = llm
        self.tools = tools
        self.evaluator = ExpressionEvaluator()

    async def process_user_input(
        self,
        session: SessionContext,
        user_text: str,
    ) -> tuple[str, str]:
        """
        Process user input and generate response.

        Args:
            session: Session context
            user_text: User input text

        Returns:
            Tuple of (response_text, next_state_id)
        """
        # Record user input in history
        session.set_variable("last_user_input", user_text)
        session.add_to_history(
            {
                "type": "user_input",
                "text": user_text,
                "state": session.current_state,
            }
        )

        logger.info(
            "processing_user_input",
            session_id=session.session_id,
            state=session.current_state,
            text_length=len(user_text),
        )

        # Get current state
        state = self._get_state(session.current_state)
        if not state:
            raise ValueError(f"State not found: {session.current_state}")

        # Render prompt
        try:
            prompt = await self.renderer.render(
                state.prompt.template,
                session.variables,
            )
        except Exception as e:
            logger.error("prompt_render_failed", error=str(e))
            raise

        # Call LLM
        system_prompt = state.prompt.systemPrompt
        llm_response = await self.llm.call(
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=state.prompt.maxTokens,
            temperature=state.prompt.temperature,
        )

        response_text = llm_response.get("text", "")

        # Check for tool calls
        tool_calls = llm_response.get("tool_calls", [])
        if tool_calls:
            for tool_call in tool_calls:
                result = await self._invoke_tool(session, tool_call)
                session.tool_results[tool_call.get("name", "unknown")] = result

        # Record LLM response
        session.add_to_history(
            {
                "type": "llm_response",
                "text": response_text,
                "state": session.current_state,
                "tool_calls": tool_calls,
            }
        )

        next_state = await self._evaluate_transitions(
            session,
            state,
            user_text,
        )

        if next_state != session.current_state:
            session.add_to_history(
                {
                    "type": "state_transition",
                    "from_state": session.current_state,
                    "to_state": next_state,
                }
            )
            logger.info(
                "state_transition",
                session_id=session.session_id,
                from_state=session.current_state,
                to_state=next_state,
            )
            session.current_state = next_state

        return response_text, next_state

    async def _evaluate_transitions(
        self,
        session: SessionContext,
        state: StateConfig,
        user_text: str,
    ) -> str:
        """
        Evaluate state transitions and return next state.

        Args:
            session: Session context
            state: Current state
            user_text: User input text

        Returns:
            Next state ID
        """
        # Sort transitions by priority
        sorted_transitions = sorted(
            state.transitions,
            key=lambda t: t.priority,
            reverse=True,
        )

        for transition in sorted_transitions:
            if self.evaluator.evaluate(
                transition.condition,
                user_input=user_text,
                variables=session.variables,
                tool_results=session.tool_results,
            ):
                return transition.targetState

        if state.fallbackState:
            return state.fallbackState

        return session.current_state

    async def _invoke_tool(
        self,
        session: SessionContext,
        tool_call: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Invoke a tool based on LLM tool call.

        Args:
            session: Session context
            tool_call: Tool call from LLM

        Returns:
            Tool result
        """
        tool_name = tool_call.get("name")
        tool_config = self.config.tools.get(tool_name)

        if not tool_config:
            logger.warning("tool_not_found", tool_name=tool_name)
            return {"error": f"Tool not found: {tool_name}"}

        payload = {}
        if tool_config.payloadTemplate:
            try:
                payload = await self.renderer.render_dict(
                    tool_config.payloadTemplate,
                    session.variables,
                )
            except Exception as e:
                logger.error("payload_template_render_failed", error=str(e))
                return {"error": f"Failed to render payload: {e}"}

        if tool_call.get("arguments"):
            try:
                import json

                arguments = json.loads(tool_call["arguments"])
                payload.update(arguments)
            except json.JSONDecodeError:
                logger.error("tool_arguments_parse_failed")

        result = await self.tools.invoke(tool_config, payload)

        logger.info(
            "tool_invoked",
            session_id=session.session_id,
            tool_name=tool_name,
            success=result.get("success", False),
        )

        return result

    def _get_state(self, state_id: str) -> Optional[StateConfig]:
        for state in self.config.states:
            if state.id == state_id:
                return state
        return None

    def get_entry_state(self) -> StateConfig:
        """Returns the entry state configuration."""
        state = self._get_state(self.config.entryState)
        if not state:
            raise ValueError(f"Entry state not found: {self.config.entryState}")
        return state
