from pathlib import Path

import yaml
from pydantic import BaseModel


class PathsConfig(BaseModel):
    raw_data: Path
    processed_data: Path
    models: Path


class ParametersConfig(BaseModel):
    deletion_threshold: float
    r2_threshold: float


class ModelConfig(BaseModel):
    n_estimators: int
    random_state: int


class AppConfig(BaseModel):
    paths: PathsConfig
    parameters: ParametersConfig
    model: ModelConfig


# La fonction de chargement de la config
def load_config(config_path: str | Path) -> AppConfig:
    """Charge le fichier YAML et valide sa structure via Pydantic."""
    with open(config_path, encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)

    return AppConfig(**config_dict)
