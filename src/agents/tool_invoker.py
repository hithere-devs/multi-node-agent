"""Tool invocation system for agents."""

import asyncio
from typing import Any, Callable, Dict, Optional

import httpx

from agents.config_models import ToolConfig
from utils.logger import get_logger

logger = get_logger(__name__)


class ToolInvoker:
    """Invokes external tools (HTTP endpoints, Lambda functions, local handlers)."""

    def __init__(self, function_registry: Optional[Dict[str, Callable]] = None) -> None:
        self.function_registry = function_registry or {}
        self.http_client = httpx.AsyncClient(timeout=30.0)

    async def invoke(
        self,
        tool: ToolConfig,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Invoke a tool and return the result.

        Args:
            tool: Tool configuration
            payload: Input payload for the tool

        Returns:
            Tool result as a dictionary

        Raises:
            ValueError: If tool type is unsupported or invocation fails
        """
        payload = payload or {}
        logger.info(
            "invoking_tool",
            tool_id=tool.id,
            tool_type=tool.type,
        )

        try:
            if tool.type == "http":
                return await self._invoke_http(tool, payload)
            elif tool.type in ("lambda", "function"):
                return await self._invoke_function(tool, payload)
            else:
                raise ValueError(f"Unsupported tool type: {tool.type}")
        except Exception as e:
            logger.error(
                "tool_invocation_failed",
                tool_id=tool.id,
                error=str(e),
            )
            return {
                "error": str(e),
                "tool_id": tool.id,
                "success": False,
            }

    async def _invoke_http(
        self, tool: ToolConfig, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Invoke an HTTP endpoint.

        Args:
            tool: Tool configuration with HTTP details
            payload: Request payload

        Returns:
            Response from the HTTP endpoint
        """
        if not tool.endpoint:
            raise ValueError("HTTP tool missing endpoint")

        headers = tool.headers or {}
        if tool.credentials:
            if "Authorization" in tool.credentials:
                headers["Authorization"] = tool.credentials["Authorization"]

        method = (tool.method or "POST").upper()

        try:
            response = await asyncio.wait_for(
                self.http_client.request(
                    method,
                    tool.endpoint,
                    json=payload,
                    headers=headers,
                ),
                timeout=tool.timeout,
            )
            response.raise_for_status()
            return {
                "success": True,
                "status": response.status_code,
                "data": response.json(),
            }
        except httpx.HTTPError as e:
            raise ValueError(f"HTTP request failed: {e}") from e
        except asyncio.TimeoutError as e:
            raise ValueError(f"HTTP request timed out after {tool.timeout}s") from e

    async def _invoke_function(
        self, tool: ToolConfig, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Invoke a local function from the registry.

        Args:
            tool: Tool configuration
            payload: Input payload

        Returns:
            Function result
        """
        if tool.id not in self.function_registry:
            raise ValueError(f"Function not found in registry: {tool.id}")

        handler = self.function_registry[tool.id]

        try:
            if asyncio.iscoroutinefunction(handler):
                result = await asyncio.wait_for(
                    handler(payload),
                    timeout=tool.timeout,
                )
            else:
                result = await asyncio.wait_for(
                    asyncio.to_thread(handler, payload),
                    timeout=tool.timeout,
                )

            return {
                "success": True,
                "data": result,
            }
        except asyncio.TimeoutError as e:
            raise ValueError(f"Function timed out after {tool.timeout}s") from e
        except Exception as e:
            raise ValueError(f"Function invocation failed: {e}") from e

    def register_function(
        self,
        function_id: str,
        handler: Callable,
    ) -> None:
        """
        Register a local function handler.

        Args:
            function_id: Unique identifier for the function
            handler: Async or sync callable
        """
        self.function_registry[function_id] = handler
        logger.info("function_registered", function_id=function_id)

    async def close(self) -> None:
        """Closes HTTP client and cleans up resources."""
        await self.http_client.aclose()
