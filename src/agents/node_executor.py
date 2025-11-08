"""Node executor for multi-nodal agent workflows."""

import json
from typing import Any, Dict, Optional

from agents.config_models import MultiNodeConfig
from agents.prompt_renderer import PromptRenderer
from agents.llm_provider import LLMProvider
from agents.builtin_tools import get_builtin_tool, is_builtin_tool
from utils.logger import get_logger

logger = get_logger("node-executor")


class NodeExecutor:
    """Executes individual node logic in multi-nodal flows."""

    def __init__(self, llm_provider: LLMProvider):
        """Initialize executor with LLM provider.

        Args:
            llm_provider: LLM provider for generating responses
        """
        self.llm_provider = llm_provider
        self.prompt_renderer = PromptRenderer()
        self.execution_history: list = []

    async def execute_node(
        self,
        node: MultiNodeConfig,
        context: Dict[str, Any],
        full_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute a node's logic and generate response.

        Args:
            node: Node configuration to execute
            context: Execution context with variables and user input
            full_context: Full LLM context including chat history

        Returns:
            Execution result with generated response and metadata
        """
        try:
            # Validate context requirements
            self._validate_context(node, context)

            # Render prompt with context (async)
            rendered_prompt = await self.prompt_renderer.render(
                node.prompt.template, context
            )

            logger.info(
                "node_execution_started",
                node_id=node.id,
                node_name=node.name,
                context_keys=list(context.keys()),
            )

            # Generate response using LLM with full context
            response = await self._generate_response(
                rendered_prompt,
                node.prompt.systemPrompt,
                node.prompt.maxTokens,
                node.prompt.temperature,
                full_context=full_context,
            )

            # Check if response triggers a tool call
            tool_result = await self._check_and_execute_tools(response, context)

            # Build execution result
            result = {
                "success": True,
                "node_id": node.id,
                "node_name": node.name,
                "response": response,
                "tools_executed": (
                    tool_result.get("tools_executed", []) if tool_result else []
                ),
                "context_used": context,
                "metadata": node.metadata or {},
            }

            # Add tool result if any tool was executed
            if tool_result:
                result["tool_result"] = tool_result

            # Record in history
            self.execution_history.append(result)

            logger.info(
                "node_execution_completed",
                node_id=node.id,
                response_length=len(response),
                tools_executed=len(result.get("tools_executed", [])),
            )

            return result

        except Exception as e:
            logger.error("node_execution_failed", node_id=node.id, error=str(e))
            return {"success": False, "node_id": node.id, "error": str(e)}

    async def _generate_response(
        self,
        prompt: str,
        system_prompt: Optional[str],
        max_tokens: int,
        temperature: float,
        full_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate response using LLM.

        Args:
            prompt: User/rendered prompt
            system_prompt: System prompt for the model
            max_tokens: Maximum tokens in response
            temperature: Temperature for generation
            full_context: Full context including chat history

        Returns:
            Generated response text
        """
        try:
            # Call LLM provider using standard interface
            result = await self.llm_provider.call(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            return result.get("text", "")

        except Exception as e:
            logger.error("llm_generation_failed", error=str(e))
            raise

    async def _check_and_execute_tools(
        self, response: str, context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Check if response contains tool calls and execute them.

        Args:
            response: LLM response text
            context: Execution context

        Returns:
            Tool execution result or None
        """
        try:
            # Look for tool invocations in the response
            # Format: [TOOL: tool_name] or similar patterns
            tools_executed = []

            for tool_name in ["hangup", "transfer", "get_health_id"]:
                if (
                    f"[{tool_name.upper()}]" in response
                    or tool_name in response.lower()
                ):
                    tool = get_builtin_tool(tool_name)
                    if tool:
                        result = await tool.execute(**context)
                        tools_executed.append(
                            {
                                "tool": tool_name,
                                "result": result,
                            }
                        )
                        logger.info("tool_executed", tool=tool_name, result=result)

            if tools_executed:
                return {
                    "tools_executed": tools_executed,
                    "execution_count": len(tools_executed),
                }

            return None

        except Exception as e:
            logger.error("tool_execution_error", error=str(e))
            return None

    def _validate_context(self, node: MultiNodeConfig, context: Dict[str, Any]) -> None:
        """Validate that required context is present.

        Args:
            node: Node to validate
            context: Execution context

        Raises:
            ValueError: If required context is missing
        """
        if not node.context:
            return

        for field_name, requirement in node.context.items():
            if requirement == "required" and field_name not in context:
                raise ValueError(f"Required context field missing: {field_name}")

    def get_execution_history(self) -> list:
        """Get execution history for current session."""
        return self.execution_history

    def clear_execution_history(self) -> None:
        """Clear execution history."""
        self.execution_history = []

    def get_node_context_requirements(self, node: MultiNodeConfig) -> Dict[str, str]:
        """Get context requirements for a node.

        Args:
            node: Node configuration

        Returns:
            Dict of required context fields
        """
        return node.context or {}
