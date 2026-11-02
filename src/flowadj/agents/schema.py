"""The report every system must return, plus a tolerant JSON extractor."""
import json
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

FAMILIES = ["benign", "ftp_bruteforce", "ssh_bruteforce", "dos", "ddos",
            "portscan", "botnet", "web_attack", "other"]
Family = Literal["benign", "ftp_bruteforce", "ssh_bruteforce", "dos", "ddos",
                 "portscan", "botnet", "web_attack", "other"]

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


class Claim(BaseModel):
    """One checkable fact, tied to the tool call that produced it."""
    model_config = ConfigDict(extra="ignore")
    type: Literal["ip", "port", "count", "ratio", "time", "other"] = "other"
    value: str | int | float
    text: str = ""
    tool_call_id: str | None = None


class Report(BaseModel):
    model_config = ConfigDict(extra="ignore")
    verdict: Literal["attack", "benign"]
    family: Family
    attackers: list[str] = Field(default_factory=list)
    victims: list[str] = Field(default_factory=list)
    ports: list[int] = Field(default_factory=list)
    window_start: str | None = None
    window_end: str | None = None
    summary: str = ""
    claims: list[Claim] = Field(default_factory=list)

    @field_validator("verdict", "family", mode="before")
    @classmethod
    def _norm(cls, v):
        return v.strip().lower() if isinstance(v, str) else v


def _short(e: ValidationError) -> str:
    return "; ".join(f"{'.'.join(map(str, x['loc']))}: {x['msg']}" for x in e.errors()[:5])


def extract_report(text: str) -> tuple[Report | None, str | None]:
    """Find the first JSON object in the text that validates as a Report.
    Returns (report, None) or (None, reason)."""
    text = _THINK.sub("", text or "")
    dec = json.JSONDecoder()
    err = "no JSON object with a 'verdict' field found"
    i = 0
    while i < len(text):
        if text[i] != "{":
            i += 1
            continue
        try:
            obj, end = dec.raw_decode(text, i)
        except json.JSONDecodeError:
            i += 1
            continue
        if isinstance(obj, dict) and isinstance(obj.get("report"), dict):
            obj = obj["report"]
        if isinstance(obj, dict) and "verdict" in obj:
            try:
                return Report.model_validate(obj), None
            except ValidationError as e:
                err = _short(e)
        i = end
    return None, err