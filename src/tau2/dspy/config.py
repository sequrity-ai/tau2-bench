"""Configuration module for tau2-bench DSPy integration."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from tau2.utils.utils import defence_params, DATA_DIR


class StrategyType(str, Enum):
    """Available optimization strategy types."""

    BOOTSTRAP = "bootstrap"
    BOOTSTRAP_RS = "bootstrap_rs"
    MIPRO = "mipro"
    GEPA = "gepa"


@dataclass
class OptimizationConfig:
    """Configuration for optimization strategies."""

    # Common settings
    max_iterations: int = 10
    num_candidates: int = 10

    # Bootstrap-specific
    max_bootstrapped_demos: int = 4
    max_labeled_demos: int = 16

    # MIPRO-specific
    num_threads: int = 6
    requires_permission_to_run: bool = False

    # GEPA-specific
    max_metric_calls: int = 50  # Default stopping condition for GEPA
    reflection_lm: str | None = None
    skip_perfect_score: bool = True
    display_progress_bar: bool = True


@dataclass
class DSPyConfig:
    """Main configuration for tau2 DSPy integration.

    This configuration combines optimization settings with tau2-bench's
    existing defence_params configuration.
    """

    # Model settings
    model: str = "gpt-4.1"
    model_args: dict = field(default_factory=lambda: {"temperature": 0.0})

    # PLLM settings
    pllm_prompt: str | None = None
    who_from: str = "BOT"

    # Optimization settings
    strategy: StrategyType = StrategyType.BOOTSTRAP
    optimization_config: OptimizationConfig = field(
        default_factory=OptimizationConfig
    )

    # Defence params integration
    use_dual_llm: bool = True
    strict_mode: bool = False
    max_retry_attempts: int = 20
    max_n_turns: int = 100

    # Output settings
    output_dir: Path | None = None

    @property
    def defence_params_override(self) -> dict[str, Any]:
        """Generate defence_params override dict from config."""
        return {
            "bot_dual_llm_mode": self.use_dual_llm,
            "strict_mode": self.strict_mode,
            "max_retry_attempts": self.max_retry_attempts,
            "max_n_turns": self.max_n_turns,
        }

    @classmethod
    def from_defence_params(
        cls,
        model: str = "gpt-4.1",
        pllm_prompt: str | None = None,
        strategy: StrategyType = StrategyType.BOOTSTRAP,
        **kwargs,
    ) -> "DSPyConfig":
        """Create DSPyConfig using current defence_params values.

        Args:
            model: Model to use.
            pllm_prompt: PLLM system prompt.
            strategy: Optimization strategy.
            **kwargs: Additional config overrides.

        Returns:
            DSPyConfig instance with defence_params values.
        """
        return cls(
            model=model,
            pllm_prompt=pllm_prompt,
            strategy=strategy,
            use_dual_llm=defence_params.get("bot_dual_llm_mode", True),
            strict_mode=defence_params.get("strict_mode", False),
            max_retry_attempts=defence_params.get("max_retry_attempts", 20),
            max_n_turns=defence_params.get("max_n_turns", 100),
            **kwargs,
        )

    def get_optimized_prompt_path(self, domain: str) -> Path:
        """Get the path for storing optimized prompts.

        Args:
            domain: Domain name (e.g., 'airline', 'retail').

        Returns:
            Path to the optimized prompt JSON file.
        """
        if self.output_dir:
            base_dir = self.output_dir
        else:
            base_dir = DATA_DIR / "tau2" / "optimized_prompts"

        base_dir.mkdir(parents=True, exist_ok=True)
        return base_dir / f"{domain}.json"


@dataclass
class OptimizedPromptInfo:
    """Information about an optimized prompt."""

    domain: str
    prompt: str
    score: float
    strategy: str
    model: str
    train_size: int
    val_size: int
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "domain": self.domain,
            "prompt": self.prompt,
            "score": self.score,
            "strategy": self.strategy,
            "model": self.model,
            "train_size": self.train_size,
            "val_size": self.val_size,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "OptimizedPromptInfo":
        """Create from dictionary."""
        return cls(
            domain=data["domain"],
            prompt=data["prompt"],
            score=data["score"],
            strategy=data["strategy"],
            model=data["model"],
            train_size=data["train_size"],
            val_size=data["val_size"],
            metadata=data.get("metadata", {}),
        )

    def save(self, path: Path) -> None:
        """Save to JSON file."""
        import json
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "OptimizedPromptInfo":
        """Load from JSON file."""
        import json
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))
