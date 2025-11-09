"""Node-based workflow schema definitions."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class NodeType(str, Enum):
    """Types of workflow nodes."""

    WELCOME = "welcome"
    CONVERSATIONAL = "conversational"
    TRANSITION = "transition"
    TOOL = "tool"
    HANGUP = "hangup"


class TransitionConditionType(str, Enum):
    """Types of conditions for node transitions."""

    INTENT = "intent"
    REGEX = "regex"
    EXPRESSION = "expression"
    TOOL_RESULT = "tool_result"
    ALWAYS = "always"


@dataclass
class TransitionCondition:
    """Defines when a transition should occur."""

    type: TransitionConditionType
    description: Optional[str] = None
    expression: Optional[str] = None  # regex pattern or Python expression
    intent_description: Optional[str] = None  # for LLM-based intent matching
    tool_name: Optional[str] = None  # for tool-based transitions
    expected_value: Optional[Any] = None  # expected tool result


@dataclass
class NodeTransition:
    """Defines a transition from one node to another."""

    target_node_id: str
    condition: TransitionCondition
    priority: int = 5  # higher = evaluated first
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PromptConfig:
    """Configuration for node prompts."""

    system_prompt: str
    instructions: str
    max_tokens: int = 150
    temperature: float = 0.7


@dataclass
class NodeConfig:
    """Configuration for a single workflow node."""

    id: str
    type: NodeType
    name: str
    description: str
    prompt: PromptConfig
    tools: List[str] = field(default_factory=list)
    transitions: List[NodeTransition] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Optional TTS/STT overrides
    tts_voice: Optional[str] = None
    preserve_chat_context: bool = False  # whether to pass chat_ctx on transitions


@dataclass
class WorkflowConfig:
    """Complete workflow configuration."""

    id: str
    name: str
    version: str
    description: str
    entry_node_id: str
    nodes: List[NodeConfig]
    global_tools: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowState:
    """Runtime state for workflow execution."""

    current_node_id: str
    session_data: Dict[str, Any] = field(default_factory=dict)
    visited_nodes: List[str] = field(default_factory=list)
    transition_count: int = 0
    is_complete: bool = False
