"""Configuration management utilities."""

from pathlib import Path
from typing import Any, Dict, Optional

from omegaconf import DictConfig, OmegaConf


def load_config(config_path: str, overrides: Optional[list[str]] = None) -> DictConfig:
    """Load configuration from YAML file with optional overrides.
    
    Args:
        config_path: Path to the configuration file.
        overrides: List of configuration overrides in key=value format.
        
    Returns:
        OmegaConf DictConfig object.
    """
    config = OmegaConf.load(config_path)
    
    if overrides:
        overrides_config = OmegaConf.from_dotlist(overrides)
        config = OmegaConf.merge(config, overrides_config)
    
    return config


def save_config(config: DictConfig, save_path: str) -> None:
    """Save configuration to YAML file.
    
    Args:
        config: Configuration object to save.
        save_path: Path where to save the configuration.
    """
    OmegaConf.save(config, save_path)


def get_config_value(config: DictConfig, key: str, default: Any = None) -> Any:
    """Get a configuration value with optional default.
    
    Args:
        config: Configuration object.
        key: Dot-separated key path (e.g., 'model.hidden_size').
        default: Default value if key is not found.
        
    Returns:
        Configuration value or default.
    """
    try:
        return OmegaConf.select(config, key)
    except Exception:
        return default


def validate_config(config: DictConfig) -> bool:
    """Validate configuration structure and values.
    
    Args:
        config: Configuration object to validate.
        
    Returns:
        True if configuration is valid, False otherwise.
    """
    required_keys = [
        "data.synthetic.n_samples",
        "data.synthetic.sequence_length",
        "model.architecture",
        "training.batch_size",
        "training.learning_rate",
        "training.num_epochs",
    ]
    
    for key in required_keys:
        if OmegaConf.select(config, key) is None:
            print(f"Missing required configuration key: {key}")
            return False
    
    return True


def create_experiment_dir(config: DictConfig, base_dir: str = "experiments") -> Path:
    """Create experiment directory based on configuration.
    
    Args:
        config: Configuration object.
        base_dir: Base directory for experiments.
        
    Returns:
        Path to the created experiment directory.
    """
    exp_name = config.experiment.name
    exp_version = config.experiment.version
    
    exp_dir = Path(base_dir) / f"{exp_name}_v{exp_version}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (exp_dir / "checkpoints").mkdir(exist_ok=True)
    (exp_dir / "logs").mkdir(exist_ok=True)
    (exp_dir / "results").mkdir(exist_ok=True)
    (exp_dir / "configs").mkdir(exist_ok=True)
    
    return exp_dir
