"""Configuration storage and loading."""

import json
from pathlib import Path
from typing import Dict, Optional, Union

from agents.config_models import CustomerConfig, MultiNodeAgentConfig
from utils.logger import get_logger

logger = get_logger(__name__)


class ConfigStore:
    """Loads and manages customer configurations."""

    @staticmethod
    async def load_from_file(file_path: str) -> CustomerConfig:
        """
        Load customer configuration from a JSON file.

        Args:
            file_path: Path to the JSON configuration file

        Returns:
            Parsed CustomerConfig

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If JSON is invalid
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")

        try:
            with open(path) as f:
                config_dict = json.load(f)
            config = CustomerConfig(**config_dict)
            logger.info("config_loaded", file_path=file_path, customer_id=config.id)
            return config
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in configuration file: {e}") from e
        except Exception as e:
            logger.error("config_load_failed", file_path=file_path, error=str(e))
            raise

    @staticmethod
    async def load_from_dict(config_dict: Dict) -> CustomerConfig:
        """
        Load customer configuration from a dictionary.

        Args:
            config_dict: Configuration dictionary

        Returns:
            Parsed CustomerConfig

        Raises:
            ValueError: If configuration is invalid
        """
        try:
            config = CustomerConfig(**config_dict)
            logger.info("config_loaded_from_dict", customer_id=config.id)
            return config
        except Exception as e:
            logger.error("config_dict_load_failed", error=str(e))
            raise

    @staticmethod
    def save_to_file(config: CustomerConfig, file_path: str) -> None:
        """
        Save customer configuration to a JSON file.

        Args:
            config: CustomerConfig instance
            file_path: Path where to save the file
        """
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(path, "w") as f:
                json.dump(config.model_dump(), f, indent=2)
            logger.info("config_saved", file_path=file_path, customer_id=config.id)
        except Exception as e:
            logger.error("config_save_failed", file_path=file_path, error=str(e))
            raise

    @staticmethod
    async def load_multi_node_from_file(file_path: str) -> MultiNodeAgentConfig:
        """Load multi-nodal agent configuration from JSON file.

        Args:
            file_path: Path to the multi-nodal config JSON

        Returns:
            Parsed MultiNodeAgentConfig

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If JSON is invalid
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")

        try:
            with open(path) as f:
                config_dict = json.load(f)
            config = MultiNodeAgentConfig(**config_dict)
            logger.info(
                "multi_node_config_loaded", file_path=file_path, agent_id=config.id
            )
            return config
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in configuration file: {e}") from e
        except Exception as e:
            logger.error(
                "multi_node_config_load_failed", file_path=file_path, error=str(e)
            )
            raise

    @staticmethod
    async def load_multi_node_from_dict(config_dict: Dict) -> MultiNodeAgentConfig:
        """Load multi-nodal config from dictionary.

        Args:
            config_dict: Configuration dictionary

        Returns:
            Parsed MultiNodeAgentConfig
        """
        try:
            config = MultiNodeAgentConfig(**config_dict)
            logger.info("multi_node_config_loaded_from_dict", agent_id=config.id)
            return config
        except Exception as e:
            logger.error("multi_node_config_dict_load_failed", error=str(e))
            raise

    @staticmethod
    async def save_multi_node_config(
        config: MultiNodeAgentConfig, file_path: str
    ) -> None:
        """Save multi-nodal config to JSON file.

        Args:
            config: MultiNodeAgentConfig instance
            file_path: Path where to save the file
        """
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(path, "w") as f:
                json.dump(config.model_dump(), f, indent=2)
            logger.info(
                "multi_node_config_saved", file_path=file_path, agent_id=config.id
            )
        except Exception as e:
            logger.info(
                "multi_node_config_save_failed", file_path=file_path, error=str(e)
            )
            raise

    # Synchronous versions for compatibility
    @staticmethod
    def load_multinode_config(file_path: str) -> MultiNodeAgentConfig:
        """Load multi-nodal agent configuration (synchronous).

        Args:
            file_path: Path to the multi-nodal config JSON

        Returns:
            Parsed MultiNodeAgentConfig
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")

        try:
            with open(path) as f:
                config_dict = json.load(f)
            config = MultiNodeAgentConfig(**config_dict)
            logger.info(
                "multi_node_config_loaded", file_path=file_path, agent_id=config.id
            )
            return config
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in configuration file: {e}") from e
        except Exception as e:
            logger.error(
                "multi_node_config_load_failed", file_path=file_path, error=str(e)
            )
            raise

    @staticmethod
    def load_customer_config(file_path: str) -> CustomerConfig:
        """Load customer configuration (synchronous).

        Args:
            file_path: Path to the customer config JSON

        Returns:
            Parsed CustomerConfig
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")

        try:
            with open(path) as f:
                config_dict = json.load(f)
            config = CustomerConfig(**config_dict)
            logger.info("config_loaded", file_path=file_path, customer_id=config.id)
            return config
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in configuration file: {e}") from e
        except Exception as e:
            logger.error("config_load_failed", file_path=file_path, error=str(e))
            raise
