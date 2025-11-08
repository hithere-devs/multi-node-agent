"""Built-in tools for multi-nodal agents."""

from typing import Any, Dict, Optional
from utils.logger import get_logger

logger = get_logger("builtin-tools")


class BuiltinTool:
    """Base class for built-in tools."""

    def __init__(self, name: str, description: str):
        """Initialize built-in tool.

        Args:
            name: Tool name
            description: Tool description
        """
        self.name = name
        self.description = description

    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the tool.

        Returns:
            Execution result
        """
        raise NotImplementedError


class HangupTool(BuiltinTool):
    """Tool to gracefully end a call."""

    def __init__(self):
        """Initialize hangup tool."""
        super().__init__(
            name="hangup",
            description="Gracefully end the call with a closing message and say goodbye",
        )

    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute hangup.

        Returns:
            Result indicating hangup was executed
        """
        closing_message = kwargs.get(
            "message",
            "Thank you for reaching out to our healthcare management system. We hope we could assist you. Take care and have a great day!",
        )

        logger.info("hangup_executed", message=closing_message)

        return {
            "success": True,
            "tool": "hangup",
            "message": closing_message,
            "action": "end_call",
        }


class TransferTool(BuiltinTool):
    """Tool to transfer to another service or node."""

    def __init__(self):
        """Initialize transfer tool."""
        super().__init__(
            name="transfer",
            description="Transfer the conversation to a different service or department",
        )

    async def execute(
        self, target_service: Optional[str] = None, **kwargs
    ) -> Dict[str, Any]:
        """Execute transfer.

        Args:
            target_service: Target service to transfer to

        Returns:
            Result indicating transfer was executed
        """
        logger.info("transfer_executed", target_service=target_service)

        return {
            "success": True,
            "tool": "transfer",
            "target_service": target_service,
            "action": "transfer_call",
        }


class GetHealthIDTool(BuiltinTool):
    """Tool to retrieve or validate health ID."""

    def __init__(self):
        """Initialize health ID tool."""
        super().__init__(
            name="get_health_id",
            description="Retrieve or validate user's health ID",
        )

    async def execute(
        self, user_identifier: Optional[str] = None, **kwargs
    ) -> Dict[str, Any]:
        """Execute health ID lookup.

        Args:
            user_identifier: User's phone number or email

        Returns:
            Health ID or validation result
        """
        logger.info("health_id_lookup", user_identifier=user_identifier)

        # Mock implementation - in production, this would query a database
        health_id = f"HID-{user_identifier[-4:]}" if user_identifier else "HID-0000"

        return {
            "success": True,
            "tool": "get_health_id",
            "health_id": health_id,
            "user_identifier": user_identifier,
        }


# Registry of built-in tools
BUILTIN_TOOLS = {
    "hangup": HangupTool(),
    "transfer": TransferTool(),
    "get_health_id": GetHealthIDTool(),
}


def get_builtin_tool(tool_name: str) -> Optional[BuiltinTool]:
    """Get a built-in tool by name.

    Args:
        tool_name: Name of the tool

    Returns:
        Tool instance or None if not found
    """
    return BUILTIN_TOOLS.get(tool_name)


def is_builtin_tool(tool_name: str) -> bool:
    """Check if a tool is a built-in tool.

    Args:
        tool_name: Name of the tool

    Returns:
        True if tool is built-in
    """
    return tool_name in BUILTIN_TOOLS
