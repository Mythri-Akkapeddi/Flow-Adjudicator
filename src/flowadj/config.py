"""Config loading and project paths."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2] 
CONFIGS = ROOT / "configs"
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
CICIDS_ORIGINAL_DIR = RAW_DIR / "cicids2017_original"
CICIDS_CORRECTED_DIR = RAW_DIR / "cicids2017_corrected"


def load_yaml(name: str) -> dict:
    with open(CONFIGS / name, "r") as f:
        return yaml.safe_load(f)


def get_default() -> dict:
    return load_yaml("default.yaml")


def get_model_cfg(key: str = "local") -> dict:
    return load_yaml("models.yaml")[key]