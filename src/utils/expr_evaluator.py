"""Expression evaluator for state machine transitions."""

import re
from typing import Any, Dict, List, Optional

from agents.config_models import ConditionConfig, ConditionType


class ExpressionEvaluator:
    """Evaluates conditions for state transitions."""

    def __init__(self) -> None:
        pass

    def evaluate(
        self,
        condition: ConditionConfig,
        user_input: Optional[str] = None,
        variables: Optional[Dict[str, Any]] = None,
        tool_results: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Evaluate a condition against provided context.

        Args:
            condition: The condition to evaluate
            user_input: The user's input text
            variables: Session variables
            tool_results: Results from tool invocations

        Returns:
            True if condition is satisfied, False otherwise
        """
        variables = variables or {}
        tool_results = tool_results or {}

        if condition.type == ConditionType.ALWAYS:
            return True

        elif condition.type == ConditionType.REGEX:
            if not condition.expression or not user_input:
                return False
            try:
                return bool(re.search(condition.expression, user_input, re.IGNORECASE))
            except re.error:
                return False

        elif condition.type == ConditionType.VARIABLE:
            if not condition.variable:
                return False
            var_value = variables.get(condition.variable)
            if condition.value is not None:
                return var_value == condition.value
            return var_value is not None

        elif condition.type == ConditionType.TOOL_RESULT:
            if not condition.variable or not condition.value:
                return False
            tool_result = tool_results.get(condition.variable)
            return tool_result == condition.value

        elif condition.type == ConditionType.AND:
            if not condition.conditions:
                return False
            return all(
                self.evaluate(cond, user_input, variables, tool_results)
                for cond in condition.conditions
            )

        elif condition.type == ConditionType.OR:
            if not condition.conditions:
                return False
            return any(
                self.evaluate(cond, user_input, variables, tool_results)
                for cond in condition.conditions
            )

        elif condition.type == ConditionType.NOT:
            if not condition.conditions or len(condition.conditions) < 1:
                return False
            return not self.evaluate(
                condition.conditions[0], user_input, variables, tool_results
            )

        return False

    def evaluate_conditions_with_priority(
        self,
        conditions: List[ConditionConfig],
        user_input: Optional[str] = None,
        variables: Optional[Dict[str, Any]] = None,
        tool_results: Optional[Dict[str, Any]] = None,
    ) -> Optional[ConditionConfig]:
        """
        Evaluate multiple conditions and return the highest priority match.

        Args:
            conditions: List of conditions to evaluate
            user_input: The user's input text
            variables: Session variables
            tool_results: Results from tool invocations

        Returns:
            First matching condition or None if no matches
        """
        for condition in conditions:
            if self.evaluate(condition, user_input, variables, tool_results):
                return condition
        return None
