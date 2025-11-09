"""Prompt templating and rendering engine."""

from typing import Any, Dict

from jinja2 import Environment, Undefined, Template


class SilentUndefined(Undefined):
    """Undefined that returns empty string instead of raising error."""

    def __str__(self):
        return ""

    def __getattr__(self, name):
        return self

    def __getitem__(self, name):
        return self


class PromptRenderer:
    """Renders prompts with variable injection using Jinja2 templates."""

    def __init__(self) -> None:
        self.env = Environment(undefined=SilentUndefined)

    async def render(self, template_str: str, variables: Dict[str, Any]) -> str:
        """
        Render a template string with provided variables.

        Args:
            template_str: The template string with {{ variable }} placeholders
            variables: Dictionary of variables to inject

        Returns:
            Rendered template string

        Raises:
            jinja2.UndefinedError: If a variable is missing
        """
        try:
            template = self.env.from_string(template_str)
            return template.render(variables)
        except Exception as e:
            raise ValueError(f"Failed to render template: {e}") from e

    async def render_dict(
        self, template_dict: Dict[str, Any], variables: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Render all string values in a dictionary with provided variables.

        Useful for rendering JSON payloads for tool invocations.

        Args:
            template_dict: Dictionary with template values
            variables: Dictionary of variables to inject

        Returns:
            Dictionary with rendered values
        """
        rendered = {}
        for key, value in template_dict.items():
            if isinstance(value, str):
                rendered[key] = await self.render(value, variables)
            elif isinstance(value, dict):
                rendered[key] = await self.render_dict(value, variables)
            elif isinstance(value, list):
                rendered[key] = [
                    (
                        await self.render(item, variables)
                        if isinstance(item, str)
                        else item
                    )
                    for item in value
                ]
            else:
                rendered[key] = value
        return rendered

    def validate_template(self, template_str: str) -> bool:
        """
        Validate a template string without rendering.

        Args:
            template_str: The template string to validate

        Returns:
            True if template is valid, False otherwise
        """
        try:
            self.env.from_string(template_str)
            return True
        except Exception:
            return False
