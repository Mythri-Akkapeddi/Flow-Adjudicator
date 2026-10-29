"""Throwaway: check that every logical feature resolves to a real column."""
from flowadj.data.colmap import resolve_all
from flowadj.data.store import agent_columns

cols = agent_columns()
print(len(cols), "agent columns")
for k, v in resolve_all().items():
    print(f"{k:10s} -> {v}")
missing = [k for k, v in resolve_all().items() if v is None]
print("\nMISSING:", missing or "none")
print("\nColumns containing pack/flag/iat/win/len:")
print([c for c in cols if any(w in c for w in ("pack", "pkt", "flag", "iat", "win", "length"))])