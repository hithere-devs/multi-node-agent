"""Tests for node-based workflow system."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from workflows import (
    WorkflowConfig,
    NodeConfig,
    NodeType,
    PromptConfig,
    NodeTransition,
    TransitionCondition,
    TransitionConditionType,
    WorkflowOrchestrator,
    load_workflow_from_json,
)
from workflows.node_agents import (
    WelcomeNodeAgent,
    ConversationalNodeAgent,
    TransitionNodeAgent,
    HangupNodeAgent,
)


@pytest.fixture
def simple_workflow_config():
    """Create a simple workflow configuration for testing."""
    return WorkflowConfig(
        id="test-workflow",
        name="Test Workflow",
        version="1.0",
        description="Test workflow",
        entry_node_id="welcome",
        nodes=[
            NodeConfig(
                id="welcome",
                type=NodeType.WELCOME,
                name="Welcome",
                description="Welcome node",
                prompt=PromptConfig(
                    system_prompt="You are a test assistant",
                    instructions="Greet the user",
                    max_tokens=100,
                    temperature=0.7,
                ),
                transitions=[
                    NodeTransition(
                        target_node_id="conversation",
                        condition=TransitionCondition(
                            type=TransitionConditionType.INTENT,
                            intent_description="User wants to chat",
                        ),
                        priority=10,
                    )
                ],
            ),
            NodeConfig(
                id="conversation",
                type=NodeType.CONVERSATIONAL,
                name="Conversation",
                description="Main conversation",
                prompt=PromptConfig(
                    system_prompt="You are helpful",
                    instructions="Have a conversation",
                ),
                preserve_chat_context=True,
                transitions=[
                    NodeTransition(
                        target_node_id="hangup",
                        condition=TransitionCondition(
                            type=TransitionConditionType.INTENT,
                            intent_description="User wants to end",
                        ),
                        priority=10,
                    )
                ],
            ),
            NodeConfig(
                id="hangup",
                type=NodeType.HANGUP,
                name="Hangup",
                description="End call",
                prompt=PromptConfig(
                    system_prompt="Say goodbye",
                    instructions="End the call",
                ),
            ),
        ],
    )


class TestWorkflowOrchestrator:
    """Test WorkflowOrchestrator functionality."""

    def test_orchestrator_initialization(self, simple_workflow_config):
        """Test orchestrator initializes with correct config."""
        orchestrator = WorkflowOrchestrator(simple_workflow_config)

        assert orchestrator.config.id == "test-workflow"
        assert len(orchestrator.nodes_by_id) == 3
        assert "welcome" in orchestrator.nodes_by_id
        assert orchestrator.workflow_state["current_node_id"] == "welcome"

    def test_get_node_config(self, simple_workflow_config):
        """Test retrieving node configuration."""
        orchestrator = WorkflowOrchestrator(simple_workflow_config)

        welcome_node = orchestrator.get_node_config("welcome")
        assert welcome_node is not None
        assert welcome_node.id == "welcome"
        assert welcome_node.type == NodeType.WELCOME

        invalid_node = orchestrator.get_node_config("nonexistent")
        assert invalid_node is None

    def test_create_node_agent(self, simple_workflow_config):
        """Test creating node agents."""
        orchestrator = WorkflowOrchestrator(simple_workflow_config)

        welcome_agent = orchestrator.create_node_agent("welcome")
        assert isinstance(welcome_agent, WelcomeNodeAgent)
        assert welcome_agent.node_config.id == "welcome"

        conversation_agent = orchestrator.create_node_agent("conversation")
        assert isinstance(conversation_agent, ConversationalNodeAgent)

    def test_create_invalid_node_agent(self, simple_workflow_config):
        """Test error handling for invalid node."""
        orchestrator = WorkflowOrchestrator(simple_workflow_config)

        with pytest.raises(ValueError, match="Node not found"):
            orchestrator.create_node_agent("invalid_node")


class TestTransitionConditions:
    """Test transition condition evaluation."""

    def test_regex_condition(self, simple_workflow_config):
        """Test regex-based transitions."""
        orchestrator = WorkflowOrchestrator(simple_workflow_config)

        # Add a node with regex transition
        regex_node = NodeConfig(
            id="regex_test",
            type=NodeType.CONVERSATIONAL,
            name="Regex Test",
            description="Test regex",
            prompt=PromptConfig(
                system_prompt="Test",
                instructions="Test",
            ),
            transitions=[
                NodeTransition(
                    target_node_id="hangup",
                    condition=TransitionCondition(
                        type=TransitionConditionType.REGEX,
                        expression=r"(?i)(goodbye|bye|exit)",
                    ),
                    priority=10,
                )
            ],
        )

        agent = ConversationalNodeAgent(
            regex_node,
            orchestrator.workflow_state,
        )

        # Test matching
        context = {"user_input": "goodbye"}
        assert agent._evaluate_transition(regex_node.transitions[0].condition, context)

        # Test non-matching
        context = {"user_input": "hello"}
        assert not agent._evaluate_transition(
            regex_node.transitions[0].condition, context
        )

    def test_always_condition(self, simple_workflow_config):
        """Test always transitions."""
        orchestrator = WorkflowOrchestrator(simple_workflow_config)

        transition_node = NodeConfig(
            id="always_test",
            type=NodeType.TRANSITION,
            name="Always Test",
            description="Test always",
            prompt=PromptConfig(
                system_prompt="Test",
                instructions="Test",
            ),
            transitions=[
                NodeTransition(
                    target_node_id="hangup",
                    condition=TransitionCondition(type=TransitionConditionType.ALWAYS),
                    priority=10,
                )
            ],
        )

        agent = TransitionNodeAgent(
            transition_node,
            orchestrator.workflow_state,
        )

        context = {}
        assert agent._evaluate_transition(
            transition_node.transitions[0].condition, context
        )

    def test_expression_condition(self, simple_workflow_config):
        """Test expression-based transitions."""
        orchestrator = WorkflowOrchestrator(simple_workflow_config)

        expr_node = NodeConfig(
            id="expr_test",
            type=NodeType.CONVERSATIONAL,
            name="Expression Test",
            description="Test expression",
            prompt=PromptConfig(
                system_prompt="Test",
                instructions="Test",
            ),
            transitions=[
                NodeTransition(
                    target_node_id="hangup",
                    condition=TransitionCondition(
                        type=TransitionConditionType.EXPRESSION,
                        expression="count > 5",
                    ),
                    priority=10,
                )
            ],
        )

        agent = ConversationalNodeAgent(
            expr_node,
            orchestrator.workflow_state,
        )

        # Test expression evaluation
        context = {"count": 10}
        assert agent._evaluate_transition(expr_node.transitions[0].condition, context)

        context = {"count": 3}
        assert not agent._evaluate_transition(
            expr_node.transitions[0].condition, context
        )


class TestWorkflowLoading:
    """Test loading workflows from JSON."""

    def test_load_customer_support_workflow(self):
        """Test loading the customer support workflow."""
        from pathlib import Path

        workflow_path = (
            Path(__file__).parent.parent
            / "configs"
            / "workflows"
            / "customer_support.json"
        )

        if workflow_path.exists():
            workflow = load_workflow_from_json(str(workflow_path))

            assert workflow.id == "customer-support-workflow"
            assert workflow.entry_node_id == "welcome"
            assert len(workflow.nodes) > 0

            # Check node types
            node_types = {node.type for node in workflow.nodes}
            assert NodeType.WELCOME in node_types
            assert NodeType.CONVERSATIONAL in node_types
            assert NodeType.HANGUP in node_types


class TestWorkflowState:
    """Test workflow state management."""

    def test_state_initialization(self, simple_workflow_config):
        """Test initial workflow state."""
        orchestrator = WorkflowOrchestrator(simple_workflow_config)

        state = orchestrator.workflow_state
        assert state["current_node_id"] == "welcome"
        assert state["visited_nodes"] == []
        assert state["transition_count"] == 0
        assert state["is_complete"] is False

    def test_state_sharing_across_nodes(self, simple_workflow_config):
        """Test that state is shared across node agents."""
        orchestrator = WorkflowOrchestrator(simple_workflow_config)

        agent1 = orchestrator.create_node_agent("welcome")
        agent2 = orchestrator.create_node_agent("conversation")

        # Modify state in first agent
        agent1.workflow_state["custom_data"] = "test_value"

        # Check it's visible in second agent
        assert agent2.workflow_state["custom_data"] == "test_value"


class TestNodeTypes:
    """Test specific node type behaviors."""

    def test_welcome_node_creates_transition_tools(self, simple_workflow_config):
        """Test that welcome nodes create tools for transitions."""
        orchestrator = WorkflowOrchestrator(simple_workflow_config)
        welcome_agent = orchestrator.create_node_agent("welcome")

        # Check that transition tools are created
        assert hasattr(welcome_agent, "go_to_conversation")

    def test_chat_context_preservation(self, simple_workflow_config):
        """Test chat context preservation setting."""
        orchestrator = WorkflowOrchestrator(simple_workflow_config)

        welcome_node = orchestrator.get_node_config("welcome")
        assert welcome_node.preserve_chat_context is False

        conversation_node = orchestrator.get_node_config("conversation")
        assert conversation_node.preserve_chat_context is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
