from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


BACKEND_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Bone Age Predictor API"
    app_version: str = "1.0.0"
    debug: bool = True

    checkpoint_dir: Path = Path("checkpoints")
    metrics_dir: Path = Path("metrics")
    upload_dir: Path = Path("uploads")
    rf_models_dir: Path = Path("rf_models")
    checkpoint_dir: Path = BACKEND_ROOT / "checkpoints"
    metrics_dir: Path = BACKEND_ROOT / "metrics"
    upload_dir: Path = BACKEND_ROOT / "uploads"
    rf_models_dir: Path = BACKEND_ROOT / "rf_models"

    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    image_size: int = 512
    device: str = "cpu"
    default_model: str = "cnn"

    norm_mean: float = 0.4523
    norm_std: float = 0.2118

    @property
    def cors_origin_list(self) -> List[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
