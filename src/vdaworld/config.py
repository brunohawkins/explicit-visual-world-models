"""
Pydantic configuration models for vdaworld.

All runtime settings are parsed from a YAML file via :func:`load_config`,
which validates the structure and types before any pipeline code runs.
"""

from __future__ import annotations

import os
from typing import Tuple

import yaml
from pydantic import BaseModel, Field


class ModelSpec(BaseModel):
    """Pricing and identity for a model loaded from ``models/<name>.yaml``."""

    model_name: str
    backend: str = "gemini"  # "gemini" or "vllm"
    base_url: str | None = None  # OpenAI-compatible base URL for vllm backend
    input_price: float = 0.0  # $ per 1M input tokens
    output_price: float = 0.0  # $ per 1M output tokens
    cache_price: float = 0.0  # $ per 1M cached tokens


class VLMConfig(BaseModel):
    """Configuration for the Vision-Language Model interface."""

    model: str = "gemini_flash"
    temperature: float = 0.0


class TaskConfig(BaseModel):
    """Configuration for the specific generation task."""

    image_path: str
    caption: str
    output_dir: str = "outputs"
    simulator_class_name: str = "VideoSimulation"
    prompts_path: str = "prompts/generic"
    fps: int = 30
    simulation_steps: int = 150
    frame_size: Tuple[int, int] = (1024, 576)
    # Agentic generation settings
    critic_max_turns: int = 20
    lint_check: bool = False
    # Whether to include the reference image with the initial agent prompt.
    # Set to False for ablation experiments where the model must act without
    # seeing the input image. Defaults to True for normal runs.
    provide_image: bool = True
    # Whether to restrict the agent to only get_api_documentation and write_code_from_scratch.
    # Set to True for ablation experiments. Defaults to False for normal runs.
    restricted_tools: bool = False
    # Whether to disable the WorldAPI and pass None to the simulator.
    # Set to True for ablation experiments where the model cannot use computer vision.
    # Defaults to False for normal runs.
    no_api: bool = False
    # Whether to disable the motion history image tool.
    # Set to True for ablation experiments. Defaults to False for normal runs.
    no_mhi: bool = False
    # API result cache
    cache_dir: str | None = None


class AppConfig(BaseModel):
    """Root configuration object validated by Pydantic."""

    vlm: VLMConfig = Field(default_factory=VLMConfig)
    task: TaskConfig
    model_spec: ModelSpec = Field(
        default_factory=lambda: ModelSpec(model_name="gemini-3-flash-preview")
    )


def _load_model_spec(model_alias: str) -> ModelSpec:
    """Load a :class:`ModelSpec` from ``models/<model_alias>.yaml``.

    Searches relative to the current working directory.

    Raises:
        FileNotFoundError: If no matching model file is found.
    """
    model_path = os.path.join("models", f"{model_alias}.yaml")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model spec not found: {model_path}. "
            f"Create models/{model_alias}.yaml with model_name, input_price, output_price, cache_price."
        )
    with open(model_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return ModelSpec(**data)


def load_config(config_path: str) -> AppConfig:
    """Load a YAML configuration file and parse it into a strongly typed :class:`AppConfig`.

    The ``vlm.model`` field is resolved to a :class:`ModelSpec` by loading
    ``models/<model>.yaml`` from the current working directory.

    Args:
        config_path: Path to the YAML configuration file.

    Raises:
        FileNotFoundError: If the configuration file or model spec does not exist.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    vlm_data = data.get("vlm", {})
    model_alias = vlm_data.get("model", "gemini_flash")
    model_spec = _load_model_spec(model_alias)

    config = AppConfig(
        vlm=VLMConfig(**vlm_data),
        task=TaskConfig(**data["task"]),
        model_spec=model_spec,
    )
    return config
