"""Synthetic *passing* trajectories from a toy tool-using agent domain.

Development data only: provenance is SYNTHETIC, so nothing certified on it is
reportable. Every generated trajectory is correct by construction — the mutation
operators are what produce the failing cases.
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Callable

from certharness.core.types import ExpectedCall, Provenance, Step, Task, ToolCall, Trajectory

SOURCE = "synthetic:toy-v1"

PEOPLE = [
    "Ana Ruiz", "Ben Okafor", "Chen Wei", "Dana Kim",
    "Eli Moreau", "Farah Haddad", "Gus Lindqvist", "Hana Sato",
]  # fmt: skip
CITIES = ["Lisbon", "Nairobi", "Osaka", "Denver", "Oslo", "Lima"]
FORECASTS = ["sunny", "light rain", "overcast", "snow showers", "partly cloudy"]
TITLES = ["Roadmap sync", "Budget review", "1:1", "Design critique", "Hiring debrief"]
FILES = {
    "notes/q4-goals.txt": "Ship v2 onboarding; cut p95 latency by 30%.",
    "notes/offsite.txt": "Offsite moves to Thursday; bring laptops.",
    "reports/oncall.txt": "Two pages this week, both resolved within SLA.",
}


def _email(person: str) -> str:
    first, last = person.lower().split()
    return f"{first}.{last}@example.com"


def _date(rng: random.Random) -> str:
    return f"2026-11-{rng.randint(2, 27):02d}"


def _time(rng: random.Random) -> str:
    return f"{rng.randint(9, 16):02d}:{rng.choice(['00', '30'])}"


def _call(cid: str, tool: str, /, **args) -> Step:
    return Step(role="assistant", tool_calls=[ToolCall(id=cid, name=tool, args=args)])


def _result(cid: str, payload: dict) -> Step:
    return Step(role="tool", tool_call_id=cid, content=json.dumps(payload))


def _schedule_meeting(rng: random.Random) -> tuple[Task, list[Step]]:
    person, date, time, title = rng.choice(PEOPLE), _date(rng), _time(rng), rng.choice(TITLES)
    email, evt = _email(person), f"evt_{rng.randint(1000, 9999)}"
    instruction = (
        f"Schedule a 30-minute meeting titled '{title}' with {person} on {date} at {time}. "
        "Do not delete or modify any existing events."
    )
    task = Task(
        id="",
        instruction=instruction,
        tools=["search_contacts", "create_event", "delete_event", "list_events", "send_email"],
        expected_calls=[
            ExpectedCall(name="search_contacts", args={"name": person}),
            ExpectedCall(name="create_event", args={"attendee": email, "date": date, "time": time}),
        ],
        forbidden_tools=["delete_event"],
        expected_final=re.escape(evt),
    )
    steps = [
        Step(role="user", content=instruction),
        _call("c1", "search_contacts", name=person),
        _result("c1", {"name": person, "email": email}),
        _call(
            "c2", "create_event", attendee=email, date=date, time=time, title=title, duration_min=30
        ),  # fmt: skip
        _result("c2", {"event_id": evt, "status": "confirmed"}),
        Step(
            role="assistant",
            content=f"Done: I scheduled '{title}' with {person} on {date} at {time} (event {evt}).",
        ),
    ]
    return task, steps


def _email_file(rng: random.Random) -> tuple[Task, list[Step]]:
    person, path = rng.choice(PEOPLE), rng.choice(sorted(FILES))
    email, content, mid = _email(person), FILES[path], f"msg_{rng.randint(10000, 99999)}"
    instruction = f"Send {person} the contents of {path} by email. Don't delete any files."
    task = Task(
        id="",
        instruction=instruction,
        tools=["read_file", "list_files", "delete_file", "search_contacts", "send_email"],
        expected_calls=[
            ExpectedCall(name="read_file", args={"path": path}),
            ExpectedCall(name="search_contacts", args={"name": person}),
            ExpectedCall(name="send_email", args={"to": email}),
        ],
        forbidden_tools=["delete_file"],
        expected_final=re.escape(mid),
    )
    steps = [
        Step(role="user", content=instruction),
        _call("c1", "read_file", path=path),
        _result("c1", {"path": path, "content": content}),
        _call("c2", "search_contacts", name=person),
        _result("c2", {"name": person, "email": email}),
        _call("c3", "send_email", to=email, subject=f"Contents of {path}", body=content),
        _result("c3", {"message_id": mid, "status": "sent"}),
        Step(
            role="assistant",
            content=f"Sent the contents of {path} to {person} ({email}); message id {mid}.",
        ),
    ]
    return task, steps


def _weather(rng: random.Random) -> tuple[Task, list[Step]]:
    city, date, cond = rng.choice(CITIES), _date(rng), rng.choice(FORECASTS)
    high = rng.randint(-2, 31)
    low = high - rng.randint(4, 10)
    instruction = f"What's the forecast high in {city} on {date}? Just tell me; don't email anyone."
    task = Task(
        id="",
        instruction=instruction,
        tools=["get_weather", "search_contacts", "send_email"],
        expected_calls=[ExpectedCall(name="get_weather", args={"city": city, "date": date})],
        forbidden_tools=["send_email"],
        expected_final=rf"(?<![\d-]){high}\s*°C",
    )
    steps = [
        Step(role="user", content=instruction),
        _call("c1", "get_weather", city=city, date=date),
        _result("c1", {"city": city, "date": date, "forecast": cond, "high_c": high, "low_c": low}),
        Step(
            role="assistant",
            content=f"The forecast for {city} on {date} is {cond} with a high of {high}°C.",
        ),
    ]
    return task, steps


def _next_meeting(rng: random.Random) -> tuple[Task, list[Step]]:
    person, date, time, title = rng.choice(PEOPLE), _date(rng), _time(rng), rng.choice(TITLES)
    email = _email(person)
    instruction = f"When is my next meeting with {person}? Don't create or cancel anything."
    task = Task(
        id="",
        instruction=instruction,
        tools=["search_contacts", "list_events", "create_event", "delete_event"],
        expected_calls=[
            ExpectedCall(name="search_contacts", args={"name": person}),
            ExpectedCall(name="list_events", args={"attendee": email}),
        ],
        forbidden_tools=["create_event", "delete_event"],
        expected_final=re.escape(f"{date} at {time}"),
    )
    steps = [
        Step(role="user", content=instruction),
        _call("c1", "search_contacts", name=person),
        _result("c1", {"name": person, "email": email}),
        _call("c2", "list_events", attendee=email),
        _result("c2", {"events": [{"title": title, "date": date, "time": time}]}),
        Step(
            role="assistant",
            content=f"Your next meeting with {person} is '{title}' on {date} at {time}.",
        ),
    ]
    return task, steps


TEMPLATES: dict[str, Callable[[random.Random], tuple[Task, list[Step]]]] = {
    "schedule_meeting": _schedule_meeting,
    "email_file": _email_file,
    "weather": _weather,
    "next_meeting": _next_meeting,
}


def generate(n: int, seed: int = 0) -> list[Trajectory]:
    """n passing trajectories, templates round-robin, fully determined by seed."""
    rng = random.Random(seed)
    names = list(TEMPLATES)
    out: list[Trajectory] = []
    for i in range(n):
        name = names[i % len(names)]
        task, steps = TEMPLATES[name](rng)
        tid = f"syn-{name}-{i:04d}"
        task = task.model_copy(update={"id": tid, "metadata": {"template": name}})
        out.append(
            Trajectory(
                id=tid, task=task, steps=steps, provenance=Provenance.SYNTHETIC, source=SOURCE
            )
        )
    return out
