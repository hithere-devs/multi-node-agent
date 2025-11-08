#!/usr/bin/env python3
"""Verification script for multi-prompt agent system implementation."""

import asyncio
import sys
from pathlib import Path


async def verify_implementation():
    """Verify all components are properly implemented."""
    print("🔍 Multi-Prompt Agent System - Implementation Verification\n")
    print("=" * 70)

    checks_passed = 0
    checks_total = 0

    # Check 1: Module imports
    print("\n✓ Checking module imports...")
    checks_total += 1
    try:
        from agents.config_models import CustomerConfig, StateConfig
        from agents.config_store import ConfigStore
        from agents.prompt_renderer import PromptRenderer
        from agents.tool_invoker import ToolInvoker
        from agents.llm_provider import OpenAIProvider, MockLLMProvider
        from agents.tts_service import TTSService
        from agents.state_machine import StateMachineEngine, SessionContext
        from agents.session_manager import SessionManager, SessionRegistry
        from agents.livekit_connector import LiveKitConnector
        from utils.expr_evaluator import ExpressionEvaluator
        from utils.logger import get_logger

        print("  ✅ All imports successful")
        checks_passed += 1
    except ImportError as e:
        print(f"  ❌ Import failed: {e}")

    # Check 2: Configuration loading
    print("\n✓ Checking configuration loading...")
    checks_total += 1
    try:
        config = await ConfigStore.load_from_file("configs/example_customer.json")
        assert config.id == "customer-001"
        assert len(config.states) > 0
        print(f"  ✅ Config loaded: {config.name} ({len(config.states)} states)")
        checks_passed += 1
    except Exception as e:
        print(f"  ❌ Config loading failed: {e}")

    # Check 3: Expression evaluator
    print("\n✓ Checking expression evaluator...")
    checks_total += 1
    try:
        evaluator = ExpressionEvaluator()
        from agents.config_models import ConditionConfig, ConditionType

        # Test regex matching
        condition = ConditionConfig(
            type=ConditionType.REGEX,
            expression=r"(?i)test",
        )
        result = evaluator.evaluate(condition, user_input="This is a TEST")
        assert result is True

        # Test variable matching
        condition = ConditionConfig(
            type=ConditionType.VARIABLE,
            variable="status",
            value="active",
        )
        result = evaluator.evaluate(condition, variables={"status": "active"})
        assert result is True

        print("  ✅ Expression evaluator working (regex, variables)")
        checks_passed += 1
    except Exception as e:
        print(f"  ❌ Expression evaluator failed: {e}")

    # Check 4: Prompt renderer
    print("\n✓ Checking prompt renderer...")
    checks_total += 1
    try:
        renderer = PromptRenderer()
        result = await renderer.render("Hello {{name}}", {"name": "World"})
        assert result == "Hello World"
        print("  ✅ Prompt renderer working (Jinja2)")
        checks_passed += 1
    except Exception as e:
        print(f"  ❌ Prompt renderer failed: {e}")

    # Check 5: Tool invoker
    print("\n✓ Checking tool invoker...")
    checks_total += 1
    try:
        invoker = ToolInvoker()

        # Register a test function
        async def test_handler(payload):
            return {"result": "success"}

        invoker.register_function("test_func", test_handler)
        print("  ✅ Tool invoker working (function registration)")
        checks_passed += 1
        await invoker.close()
    except Exception as e:
        print(f"  ❌ Tool invoker failed: {e}")

    # Check 6: LLM provider
    print("\n✓ Checking LLM provider...")
    checks_total += 1
    try:
        llm = MockLLMProvider("Test response")
        response = await llm.call("Test prompt")
        assert response["text"] == "Test response"
        print("  ✅ LLM provider working (mock)")
        checks_passed += 1
    except Exception as e:
        print(f"  ❌ LLM provider failed: {e}")

    # Check 7: Session context
    print("\n✓ Checking session context...")
    checks_total += 1
    try:
        session = SessionContext(
            session_id="test",
            customer_id="cust",
            agent_id="agent",
            current_state="state1",
        )
        session.set_variable("key", "value")
        assert session.get_variable("key") == "value"
        print("  ✅ Session context working")
        checks_passed += 1
    except Exception as e:
        print(f"  ❌ Session context failed: {e}")

    # Check 8: State machine
    print("\n✓ Checking state machine...")
    checks_total += 1
    try:
        from agents.config_models import PromptConfig, TransitionConfig
        from agents.config_models import ConditionConfig, ConditionType

        config = CustomerConfig(
            id="test",
            name="Test",
            entryState="s1",
            states=[
                StateConfig(
                    id="s1",
                    name="S1",
                    prompt=PromptConfig(
                        template="Test: {{last_user_input}}", maxTokens=50
                    ),
                    transitions=[
                        TransitionConfig(
                            condition=ConditionConfig(type=ConditionType.ALWAYS),
                            targetState="s2",
                        )
                    ],
                ),
                StateConfig(
                    id="s2",
                    name="S2",
                    prompt=PromptConfig(template="End", maxTokens=50),
                ),
            ],
        )

        renderer = PromptRenderer()
        llm = MockLLMProvider("Response")
        tools = ToolInvoker()
        engine = StateMachineEngine(config, renderer, llm, tools)

        session = SessionContext(
            session_id="test",
            customer_id="test",
            agent_id="test",
            current_state="s1",
        )

        response, next_state = await engine.process_user_input(session, "hello")
        assert next_state == "s2"
        print("  ✅ State machine working (orchestration)")
        checks_passed += 1
        await tools.close()
    except Exception as e:
        print(f"  ❌ State machine failed: {e}")

    # Check 9: Session manager
    print("\n✓ Checking session manager...")
    checks_total += 1
    try:
        config = CustomerConfig(
            id="test",
            name="Test",
            entryState="s1",
            states=[
                StateConfig(
                    id="s1",
                    name="S1",
                    prompt=PromptConfig(template="Hi", maxTokens=50),
                )
            ],
        )

        renderer = PromptRenderer()
        llm = MockLLMProvider("Hi there")
        tools = ToolInvoker()
        engine = StateMachineEngine(config, renderer, llm, tools)

        session_manager = SessionManager(
            customer_config=config,
            state_machine=engine,
            room_name="test-room",
            participant_id="user-1",
        )

        response = await session_manager.process_user_input("Hello")
        assert len(response) > 0

        info = session_manager.get_session_info()
        assert info["is_active"] is True

        await session_manager.close()
        print("  ✅ Session manager working")
        checks_passed += 1
        await tools.close()
    except Exception as e:
        print(f"  ❌ Session manager failed: {e}")

    # Check 10: LiveKit connector
    print("\n✓ Checking LiveKit connector...")
    checks_total += 1
    try:
        config = CustomerConfig(
            id="test",
            name="Test",
            entryState="s1",
            states=[
                StateConfig(
                    id="s1",
                    name="S1",
                    prompt=PromptConfig(template="Hi", maxTokens=50),
                )
            ],
        )

        connector = LiveKitConnector(
            customer_config=config,
            llm_provider=MockLLMProvider("Hello"),
        )

        session = await connector.create_session(
            room_name="test",
            participant_id="user-1",
        )

        assert session is not None
        await connector.close_session(session.session_context.session_id)
        await connector.cleanup()

        print("  ✅ LiveKit connector working")
        checks_passed += 1
    except Exception as e:
        print(f"  ❌ LiveKit connector failed: {e}")

    # Check 11: Documentation
    print("\n✓ Checking documentation...")
    checks_total += 1
    docs = [
        "MULTI_PROMPT_GUIDE.md",
        "QUICKSTART.md",
        "IMPLEMENTATION_SUMMARY.md",
        "FILE_MANIFEST.md",
    ]
    missing = [d for d in docs if not Path(d).exists()]
    if not missing:
        print(f"  ✅ All documentation present ({len(docs)} files)")
        checks_passed += 1
    else:
        print(f"  ❌ Missing docs: {missing}")

    # Check 12: Configuration files
    print("\n✓ Checking configuration files...")
    checks_total += 1
    if Path("configs/example_customer.json").exists():
        print("  ✅ Example configuration present")
        checks_passed += 1
    else:
        print("  ❌ Example configuration missing")

    # Summary
    print("\n" + "=" * 70)
    print(f"\n📊 Verification Results: {checks_passed}/{checks_total} checks passed")

    if checks_passed == checks_total:
        print("\n✅ ✅ ✅ ALL SYSTEMS GO! ✅ ✅ ✅\n")
        print("The multi-prompt agent system is fully implemented and operational!")
        print("\nNext steps:")
        print("  1. Read MULTI_PROMPT_GUIDE.md for comprehensive documentation")
        print("  2. Review QUICKSTART.md for 5-minute setup")
        print("  3. Run: python examples_integration.py")
        print("  4. Deploy with confidence!")
        print()
        return 0
    else:
        print(
            f"\n⚠️  {checks_total - checks_passed} check(s) failed. Please review above.\n"
        )
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(verify_implementation())
    sys.exit(exit_code)
