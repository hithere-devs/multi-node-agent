"""Pydantic models for multi-prompt agent configuration."""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ConditionType(str, Enum):
    """Supported condition types for state transitions."""

    REGEX = "regex"
    VARIABLE = "variable"
    TOOL_RESULT = "toolResult"
    ALWAYS = "always"
    AND = "and"
    OR = "or"
    NOT = "not"
    INTENT = "intent"  # LLM-based intent classification


class NodeType(str, Enum):
    """Supported node types in multi-nodal workflows."""

    ENTRY = "entry"  # Entry point to the conversation
    CONVERSATION = "conversation"  # Standard conversation node
    SERVICE = "service"  # Service-specific node (pharmacy, appointments, etc.)
    TOOL = "tool"  # Tool execution node
    DECISION = "decision"  # Decision/branching node
    TRANSITION = "transition"  # State transition node
    EXIT = "exit"  # Exit/end call node


class ConditionConfig(BaseModel):
    """Configuration for a transition condition."""

    type: ConditionType
    expression: Optional[str] = None
    variable: Optional[str] = None
    value: Optional[Any] = None
    conditions: Optional[List["ConditionConfig"]] = None


ConditionConfig.model_rebuild()


class TransitionConfig(BaseModel):
    """Configuration for a state/node transition."""

    condition: ConditionConfig
    targetState: Optional[str] = None  # For legacy state machine
    targetNode: Optional[str] = None  # For multi-nodal system
    priority: int = 0


class NodeTransitionConfig(BaseModel):
    """Configuration for multi-nodal transitions."""

    condition: ConditionConfig
    targetNode: str
    priority: int = 0
    description: Optional[str] = None


class PromptConfig(BaseModel):
    """Configuration for a prompt template."""

    template: str
    maxTokens: int = 100
    temperature: float = 0.7
    systemPrompt: Optional[str] = None


class ToolCallConfig(BaseModel):
    """Configuration for a tool that can be called by the agent."""

    id: str
    type: str  # "hangup", "transfer", "custom"
    name: str
    description: str
    enabled: bool = True
    parameters: Optional[Dict[str, Any]] = None


class ToolConfig(BaseModel):
    """Configuration for a tool."""

    id: str
    type: str  # "http", "lambda", "function"
    name: str
    description: str
    endpoint: Optional[str] = None
    method: Optional[str] = "POST"
    timeout: int = 30
    headers: Optional[Dict[str, str]] = None
    payloadTemplate: Optional[Dict[str, Any]] = None
    credentials: Optional[Dict[str, str]] = None


class StateConfig(BaseModel):
    """Configuration for a state in the state machine."""

    id: str
    name: str
    prompt: PromptConfig
    tools: List[str] = Field(default_factory=list)  # Tool IDs
    transitions: List[TransitionConfig] = Field(default_factory=list)
    fallbackState: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class SessionDefaultsConfig(BaseModel):
    """Default provider configurations for a session."""

    sttModel: str = "assemblyai/universal-streaming"
    llmModel: str = "openai/gpt-4.1-mini"
    ttsModel: str = "cartesia/sonic-3"
    ttsVoice: Optional[str] = None
    language: str = "en"


class CustomerConfig(BaseModel):
    """Complete configuration for a customer."""

    id: str
    name: str
    version: str = "1.0"
    entryState: str
    states: List[StateConfig]
    tools: Dict[str, ToolConfig] = Field(default_factory=dict)
    sessionDefaults: SessionDefaultsConfig = Field(
        default_factory=SessionDefaultsConfig
    )
    metadata: Optional[Dict[str, Any]] = None


class SessionConfig(BaseModel):
    """Session-level configuration and context."""

    sessionId: str
    customerId: str
    customerConfig: CustomerConfig
    agentId: str
    participantId: str
    roomName: str
    ttlSeconds: int = 3600  # 1 hour default
    metadata: Optional[Dict[str, Any]] = None


# Multi-nodal agent models
class NodeType(str, Enum):
    """Types of nodes in multi-nodal flow."""

    ENTRY = "entry"
    SERVICE = "service"
    DECISION = "decision"
    EXIT = "exit"


class ToolDefinition(BaseModel):
    """Definition of a tool available in a node."""

    name: str
    description: str
    required: bool = False


class NodeContextRequirement(BaseModel):
    """Context requirements for a node."""

    field_name: str
    requirement_level: str  # "required", "optional", "conditional"
    description: Optional[str] = None


class MultiNodeConfig(BaseModel):
    """Configuration for a single node in multi-nodal flow."""

    id: str
    type: NodeType
    name: str
    description: Optional[str] = None
    prompt: PromptConfig
    tools: List[str] = Field(default_factory=list)  # Tool names
    context: Optional[Dict[str, str]] = None  # Context requirements
    transitions: List[NodeTransitionConfig] = Field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = None


class MultiNodeAgentConfig(BaseModel):
    """Complete configuration for multi-nodal agent."""

    id: str
    name: str
    version: str = "2.0"
    description: Optional[str] = None
    agentInstructions: Optional[str] = None  # Dynamic agent instructions
    domainContext: Optional[str] = None  # Domain-specific context
    entryNode: str
    nodes: List[MultiNodeConfig]
    globalTools: List[str] = Field(default_factory=list)  # Global tool names
    validationRules: Optional[Dict[str, Any]] = None
    sessionDefaults: SessionDefaultsConfig = Field(
        default_factory=SessionDefaultsConfig
    )
    metadata: Optional[Dict[str, Any]] = None
