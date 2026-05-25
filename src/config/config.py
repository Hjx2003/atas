from dataclasses import dataclass
from typing import Optional
import yaml


@dataclass
class DatasetConfig:
    type: str
    root: str
    image_size: int = 224
    collator: str = "atas"


@dataclass
class TrainConfig:
    # Paper: batch size 36 per GPU.
    # In this implementation, one ATAS sample = 4 object-centric images -> one 2x2 mosaic.
    # DataLoader batch size will be batch_size * 4 raw images.
    batch_size: int = 36
    num_workers: int = 4
    epochs: int = 6
    lr: float = 1e-5
    weight_decay: float = 0.1
    save_dir: str = "./outputs"
    save_interval: int = 1
    amp: bool = True


@dataclass
class BackboneConfig:
    type: str = "open_clip"
    model_name: str = "ViT-B-16"
    pretrained_name: Optional[str] = None
    pretrained_path: Optional[str] = None
    patch_grid_size: Optional[int] = 14


@dataclass
class ModelConfig:
    type: str
    backbone: BackboneConfig


@dataclass
class LossConfig:
    # Paper: lambda_1=1, lambda_2=0.01, lambda_3=1, tau=1.
    gld_weight: float = 1.0
    lld_weight: float = 0.01
    ggd_weight: float = 1.0
    tau: float = 1.0


@dataclass
class LogsConfig:
    dir: str = "./logs"
    log_interval: int = 20


@dataclass
class SwanLabConfig:
    enable: bool = False
    project: str = "ATAS"
    name: str = "atas_run"


@dataclass
class Config:
    dataset: DatasetConfig
    train: TrainConfig
    model: ModelConfig
    loss: LossConfig
    logs: LogsConfig
    swanlab: SwanLabConfig


def load_config(path: str) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return Config(
        dataset=DatasetConfig(**raw["dataset"]),
        train=TrainConfig(**raw["train"]),
        model=ModelConfig(
            type=raw["model"]["type"],
            backbone=BackboneConfig(**raw["model"]["backbone"]),
        ),
        loss=LossConfig(**raw["loss"]),
        logs=LogsConfig(**raw.get("logs", {})),
        swanlab=SwanLabConfig(**raw.get("swanlab", {})),
    )
