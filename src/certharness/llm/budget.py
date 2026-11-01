"""Hard spend cap backed by an append-only ledger.

Every live call first *reserves* its worst-case cost; the reservation fails (and the
call is never sent) if spent + outstanding reservations + this call would exceed the
cap. After the call the reservation is replaced by the actual cost. The ledger
persists across runs, so the cap covers a whole milestone, not one process.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any


class BudgetExceeded(RuntimeError):
    pass


class Budget:
    def __init__(self, cap_usd: float, ledger_path: Path):
        self.cap_usd = cap_usd
        self.ledger_path = Path(ledger_path)
        self._lock = threading.Lock()
        self._reserved = 0.0
        self._spent = sum(e.get("cost_usd", 0.0) for e in read_ledger(self.ledger_path))

    @property
    def spent_usd(self) -> float:
        return self._spent

    def reserve(self, amount: float) -> None:
        with self._lock:
            committed = self._spent + self._reserved
            if committed + amount > self.cap_usd:
                raise BudgetExceeded(
                    f"spend cap ${self.cap_usd:.2f} would be exceeded: spent ${self._spent:.4f}, "
                    f"reserved ${self._reserved:.4f}, next call up to ${amount:.4f}"
                )
            self._reserved += amount

    def release(self, amount: float) -> None:
        with self._lock:
            self._reserved -= amount

    def settle(self, reserved: float, entry: dict[str, Any]) -> None:
        with self._lock:
            self._reserved -= reserved
            self._spent += entry.get("cost_usd", 0.0)
            self._append(entry)

    def log(self, entry: dict[str, Any]) -> None:
        """Record a zero-cost event (e.g. a disk-cache hit)."""
        with self._lock:
            self._append(entry)

    def _append(self, entry: dict[str, Any]) -> None:
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with self.ledger_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, sort_keys=True) + "\n")


def read_ledger(path: Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
