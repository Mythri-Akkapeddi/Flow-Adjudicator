"""Load the hand-entered CIC-IDS2017 attack schedule."""
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

from flowadj.config import CONFIGS

SCHEDULE_PATH = CONFIGS / "attack_schedule_cicids2017.yaml"


@dataclass(frozen=True)
class Attack:
    id: str
    date: str
    family: str
    name: str
    start: pd.Timestamp
    end: pd.Timestamp
    attackers: tuple
    victims: tuple


def _load_raw(path: Path = SCHEDULE_PATH) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def tolerance_minutes(path: Path = SCHEDULE_PATH) -> int:
    return int(_load_raw(path).get("tolerance_minutes", 15))


def load_schedule(path: Path = SCHEDULE_PATH) -> list[Attack]:
    out = []
    for a in _load_raw(path)["attacks"]:
        out.append(
            Attack(
                id=a["id"],
                date=str(a["date"]),
                family=a["family"],
                name=a["name"],
                start=pd.Timestamp(f"{a['date']} {a['start']}"),
                end=pd.Timestamp(f"{a['date']} {a['end']}"),
                attackers=tuple(a["attackers"]),
                victims=tuple(a["victims"]),
            )
        )
    return out


def schedule_frame(path: Path = SCHEDULE_PATH) -> pd.DataFrame:
    return pd.DataFrame([a.__dict__ for a in load_schedule(path)])


def active_at(ts: pd.Timestamp, tolerance_min: int = 0) -> list[Attack]:
    """Which scheduled attacks should be active at time ts (± tolerance)?"""
    tol = pd.Timedelta(minutes=tolerance_min)
    return [a for a in load_schedule() if a.start - tol <= ts <= a.end + tol]