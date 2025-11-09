#!/usr/bin/env python3
"""Interactive CLI for running different LiveKit agent configurations."""

import os
import sys
import subprocess
from pathlib import Path
from typing import List, Optional

try:
    from InquirerPy import inquirer
    from InquirerPy.base.control import Choice
except ImportError:
    print("Error: InquirerPy is required but not installed.")
    print("Install it with: uv pip install inquirerpy")
    sys.exit(1)


class AgentRunner:
    """Interactive agent runner with configuration selection."""

    def __init__(self):
        self.root_dir = Path(__file__).parent
        self.configs_dir = self.root_dir / "configs"
        self.src_dir = self.root_dir / "src"

    def get_config_files(self, config_path: Path) -> List[str]:
        """Get all JSON config files from a directory.

        Args:
            config_path: Path to the config directory

        Returns:
            List of config filenames
        """
        if not config_path.exists():
            return []

        json_files = sorted([f.name for f in config_path.glob("*.json")])
        return json_files

    def select_agent_type(self) -> Optional[str]:
        """Prompt user to select agent type.

        Returns:
            Selected agent type or None if cancelled
        """
        print("\n" + "=" * 60)
        print("AGENT TYPE SELECTION")
        print("=" * 60 + "\n")

        choices = [
            Choice(
                value="workflow",
                name="LiveKit Workflows",
            ),
            Choice(
                value="multinode",
                name="Custom Node",
            ),
            Choice(
                value="langgraph",
                name="LangGraph Approach",
            ),
            Choice(value="exit", name="Exit"),
        ]

        agent_type = inquirer.select(
            message="Choose your agent type:\n",
            choices=choices,
            default="workflow",
        ).execute()

        if agent_type == "exit":
            return None

        return agent_type

    def select_config(self, agent_type: str) -> Optional[str]:
        """Prompt user to select configuration file.

        Args:
            agent_type: Type of agent (workflow, multinode, langgraph)

        Returns:
            Selected config filename or None if cancelled
        """
        # Map agent types to their config directories
        config_map = {
            "workflow": self.configs_dir / "workflows",
            "multinode": self.configs_dir / "custom_nodes",
            "langgraph": self.configs_dir / "langgraph",
        }

        config_path = config_map.get(agent_type)
        if not config_path or not config_path.exists():
            print(f"Config directory not found: {config_path}")
            return None

        config_files = self.get_config_files(config_path)

        if not config_files:
            print(f"No configuration files found in {config_path}")
            return None

        # Priority order: healthcare, realestate, then others alphabetically
        def sort_key(filename):
            name_lower = filename.lower()
            if "healthcare" in name_lower:
                return (0, filename)
            elif "realestate" in name_lower or "real" in name_lower:
                return (1, filename)
            else:
                return (2, filename)

        config_files = sorted(config_files, key=sort_key)

        print("\n" + "=" * 60)
        print("CONFIGURATION SELECTION")
        print("=" * 60 + "\n")

        choices = []
        for config_file in config_files:
            display_name = config_file.replace(".json", "").replace("_", " ").title()
            choices.append(Choice(value=config_file, name=display_name))

        choices.append(Choice(value="back", name="Back to agent selection"))

        config = inquirer.select(
            message=f"Choose configuration:\n ",
            choices=choices,
            default=config_files[0] if config_files else None,
        ).execute()

        if config == "back":
            return None

        return config

    def get_agent_script(self, agent_type: str) -> str:
        """Get the script path for the agent type.

        Args:
            agent_type: Type of agent

        Returns:
            Script path relative to src directory
        """
        script_map = {
            "workflow": "workflow_agent.py",
            "multinode": "agent.py",
            "langgraph": "langgraph_agent.py",
        }

        return str(self.src_dir / script_map[agent_type])

    def run_agent(
        self, agent_type: str, config_file: str
    ) -> subprocess.CompletedProcess:
        """Run the selected agent with configuration.

        Args:
            agent_type: Type of agent to run
            config_file: Configuration filename

        Returns:
            CompletedProcess result
        """
        script_path = self.get_agent_script(agent_type)
        mode = "console"

        env = os.environ.copy()

        if agent_type == "workflow":
            env["WORKFLOW_CONFIG"] = config_file
        elif agent_type == "multinode":
            env["AGENT_CONFIG_FILE"] = config_file
        elif agent_type == "langgraph":
            env["LANGGRAPH_CONFIG"] = config_file

        cmd = ["uv", "run", "python", script_path, mode]

        print("\n" + "=" * 60)
        print(f"STARTING {agent_type.upper()} AGENT")
        print("=" * 60)
        print(f"Config: {config_file}")
        print(f"Mode: {mode}")
        print(f"Command: {' '.join(cmd)}")
        print("=" * 60 + "\n")

        try:
            result = subprocess.run(cmd, env=env, cwd=self.root_dir)
            return result
        except KeyboardInterrupt:
            print("\n\nAgent stopped by user")
            return subprocess.CompletedProcess(cmd, returncode=130)
        except Exception as e:
            print(f"\nError running agent: {e}")
            return subprocess.CompletedProcess(cmd, returncode=1)

    def run(self):
        """Run the interactive agent selection and execution."""
        print("\n" + "=" * 60)
        print("LIVEKIT AGENT RUNNER")
        print("=" * 60 + "\n")

        while True:
            # Step 1: Select agent type
            agent_type = self.select_agent_type()
            if agent_type is None:
                print("\nGoodbye!")
                break

            # Step 2: Select configuration
            while True:
                config_file = self.select_config(agent_type)
                if config_file is None:
                    break

                result = self.run_agent(agent_type, config_file)

                if result.returncode == 0 or result.returncode == 130:
                    run_another = inquirer.confirm(
                        message="Run another agent?", default=True
                    ).execute()

                    if not run_another:
                        print("\nGoodbye!")
                        return

                    break
                else:
                    retry = inquirer.confirm(
                        message="Agent failed. Try again?", default=True
                    ).execute()

                    if not retry:
                        print("\nGoodbye!")
                        return

                    break


def main():
    """Main entry point."""
    try:
        runner = AgentRunner()
        runner.run()
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
