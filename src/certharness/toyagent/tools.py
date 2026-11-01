"""The toy domain's tools: JSON schemas (shared by the agent and the mutation operators)
and a deterministic simulated backend (`World`) the agent acts against."""

from __future__ import annotations

import copy
import json
from typing import Any

TOOL_SPECS: dict[str, dict[str, Any]] = {
    "search_contacts": {
        "description": "Look up a contact by (partial) name. Returns name and email.",
        "input_schema": {
            "type": "object",
            "properties": {"name": {"type": "string", "description": "Full or partial name."}},
            "required": ["name"],
        },
    },
    "create_event": {
        "description": "Create a calendar event with one attendee.",
        "input_schema": {
            "type": "object",
            "properties": {
                "attendee": {"type": "string", "description": "Attendee email address."},
                "date": {"type": "string", "description": "YYYY-MM-DD"},
                "time": {"type": "string", "description": "HH:MM, 24-hour"},
                "title": {"type": "string"},
                "duration_min": {"type": "integer", "description": "Defaults to 30."},
            },
            "required": ["attendee", "date", "time", "title"],
        },
    },
    "list_events": {
        "description": "List upcoming calendar events that include the given attendee.",
        "input_schema": {
            "type": "object",
            "properties": {"attendee": {"type": "string", "description": "Attendee email."}},
            "required": ["attendee"],
        },
    },
    "delete_event": {
        "description": "Delete a calendar event by id.",
        "input_schema": {
            "type": "object",
            "properties": {"event_id": {"type": "string"}},
            "required": ["event_id"],
        },
    },
    "send_email": {
        "description": "Send an email.",
        "input_schema": {
            "type": "object",
            "properties": {
                "to": {"type": "string", "description": "Recipient email address."},
                "subject": {"type": "string"},
                "body": {"type": "string"},
            },
            "required": ["to", "subject", "body"],
        },
    },
    "read_file": {
        "description": "Read a text file from the user's workspace.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    "list_files": {
        "description": "List files in the user's workspace.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    "delete_file": {
        "description": "Delete a file from the user's workspace.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    "get_weather": {
        "description": "Get the forecast for a city on a date.",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "date": {"type": "string", "description": "YYYY-MM-DD"},
            },
            "required": ["city", "date"],
        },
    },
}

TRANSIENT_ERROR = "upstream service unavailable (503); the request may be retried"


def schemas_for(tools: list[str]) -> dict[str, dict[str, Any]]:
    return {t: TOOL_SPECS[t]["input_schema"] for t in tools}


class World:
    """Simulated backend for one task. Spec keys (all optional):

    contacts [{name, email}], events [{id, title, date, time, attendee}],
    files {path: content}, weather [{city, date, forecast, high_c, low_c}],
    next_event_id, next_message_id, transient_errors [tool names that fail once].
    """

    def __init__(self, spec: dict[str, Any]):
        self.spec = copy.deepcopy(spec)
        self._failed_once: set[str] = set()
        self.last_injected = False  # whether the most recent call's error was injected

    def call(self, name: str, args: dict[str, Any]) -> tuple[str, bool]:
        """Run a tool. Returns (JSON content, is_error)."""
        self.last_injected = False
        if name in self.spec.get("transient_errors", []) and name not in self._failed_once:
            self._failed_once.add(name)
            self.last_injected = True
            return _err(TRANSIENT_ERROR)
        handler = getattr(self, f"_{name}", None)
        if handler is None or name not in TOOL_SPECS:
            return _err(f"unknown tool {name!r}")
        required = TOOL_SPECS[name]["input_schema"]["required"]
        missing = [k for k in required if k not in args]
        if missing:
            return _err(f"missing required argument(s): {', '.join(missing)}")
        return handler(
            **{k: v for k, v in args.items() if k in TOOL_SPECS[name]["input_schema"]["properties"]}
        )

    def _search_contacts(self, name: str):
        hits = [c for c in self.spec.get("contacts", []) if name.lower() in c["name"].lower()]
        if not hits:
            return _err(f"no contact matching {name!r}")
        if len(hits) == 1:
            return _ok(hits[0])
        return _ok({"matches": hits})

    def _create_event(
        self, attendee: str, date: str, time: str, title: str, duration_min: int = 30
    ):
        eid = self.spec.get("next_event_id", "evt_0001")
        self.spec.setdefault("events", []).append(
            {"id": eid, "title": title, "date": date, "time": time, "attendee": attendee}
        )
        return _ok({"event_id": eid, "status": "confirmed"})

    def _list_events(self, attendee: str):
        events = [
            {"title": e["title"], "date": e["date"], "time": e["time"]}
            for e in self.spec.get("events", [])
            if e["attendee"].lower() == attendee.lower()
        ]
        return _ok({"events": events})

    def _delete_event(self, event_id: str):
        events = self.spec.get("events", [])
        if not any(e["id"] == event_id for e in events):
            return _err(f"no event with id {event_id!r}")
        self.spec["events"] = [e for e in events if e["id"] != event_id]
        return _ok({"status": "deleted", "event_id": event_id})

    def _send_email(self, to: str, subject: str, body: str):
        return _ok({"message_id": self.spec.get("next_message_id", "msg_00001"), "status": "sent"})

    def _read_file(self, path: str):
        files = self.spec.get("files", {})
        if path not in files:
            return _err(f"file not found: {path}")
        return _ok({"path": path, "content": files[path]})

    def _list_files(self):
        return _ok({"files": sorted(self.spec.get("files", {}))})

    def _delete_file(self, path: str):
        files = self.spec.get("files", {})
        if path not in files:
            return _err(f"file not found: {path}")
        del files[path]
        return _ok({"status": "deleted", "path": path})

    def _get_weather(self, city: str, date: str):
        for w in self.spec.get("weather", []):
            if w["city"].lower() == city.lower() and w["date"] == date:
                return _ok(w)
        return _err(f"no forecast for {city} on {date}")


def _ok(payload: dict) -> tuple[str, bool]:
    return json.dumps(payload), False


def _err(msg: str) -> tuple[str, bool]:
    return json.dumps({"error": msg}), True
