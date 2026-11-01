import json

import pytest

from certharness.llm import Budget, BudgetExceeded, LLMClient, cost_usd, read_ledger
from certharness.llm.pricing import worst_case_usd


def _resp(model="claude-haiku-4-5", **usage):
    u = {"input_tokens": 1000, "output_tokens": 200, "cache_creation_input_tokens": 0,
         "cache_read_input_tokens": 0, **usage}  # fmt: skip
    return {"model": model, "stop_reason": "end_turn", "usage": u,
            "content": [{"type": "text", "text": "hi"}]}  # fmt: skip


def test_cost_usd():
    # haiku: 1000 in @ $1/M + 200 out @ $5/M = 0.001 + 0.001
    assert cost_usd("claude-haiku-4-5", _resp()["usage"]) == pytest.approx(0.002)
    # cache read at 0.1x, cache write at 1.25x of the input rate
    u = {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 10_000,
         "cache_creation_input_tokens": 1_000}  # fmt: skip
    assert cost_usd("claude-sonnet-5", u) == pytest.approx((10_000 * 0.2 + 1_000 * 2.5) / 1e6)
    with pytest.raises(KeyError, match="no price"):
        cost_usd("gpt-x", u)


def test_budget_blocks_before_exceeding(tmp_path):
    b = Budget(0.01, tmp_path / "ledger.jsonl")
    b.reserve(0.006)
    with pytest.raises(BudgetExceeded):
        b.reserve(0.006)  # outstanding reservations count
    b.settle(0.006, {"cost_usd": 0.004})
    b.reserve(0.005)


def test_budget_persists_across_processes(tmp_path):
    path = tmp_path / "ledger.jsonl"
    Budget(1.0, path).settle(0.0, {"cost_usd": 0.75})
    b = Budget(1.0, path)
    assert b.spent_usd == pytest.approx(0.75)
    with pytest.raises(BudgetExceeded):
        b.reserve(0.30)


def test_client_caches_and_logs(tmp_path):
    calls = []

    def transport(params):
        calls.append(params)
        return _resp()

    budget = Budget(1.0, tmp_path / "ledger.jsonl")
    client = LLMClient(budget, tmp_path / "cache", transport)
    params = {
        "model": "claude-haiku-4-5",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "x"}],
    }
    a = client.create("judge", **params)
    b = client.create("judge", **params)
    assert a == b and len(calls) == 1
    led = read_ledger(tmp_path / "ledger.jsonl")
    assert [e["cached"] for e in led] == [False, True]
    assert led[0]["cost_usd"] == pytest.approx(0.002) and led[1]["cost_usd"] == 0.0
    assert budget.spent_usd == pytest.approx(0.002)
    # a fresh client over the same cache is still free
    LLMClient(Budget(1.0, tmp_path / "l2.jsonl"), tmp_path / "cache", transport).create(
        "judge", **params
    )
    assert len(calls) == 1


def test_client_aborts_before_sending_when_over_cap(tmp_path):
    def transport(params):
        raise AssertionError("must not be called")

    params = {
        "model": "claude-opus-5",
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": "x"}],
    }
    cap = worst_case_usd("claude-opus-5", 10, 4096) / 2
    client = LLMClient(Budget(cap, tmp_path / "ledger.jsonl"), tmp_path / "cache", transport)
    with pytest.raises(BudgetExceeded):
        client.create("agent", **params)
    assert read_ledger(tmp_path / "ledger.jsonl") == []


def test_transport_failure_releases_reservation(tmp_path):
    def transport(params):
        raise ConnectionError("boom")

    budget = Budget(1.0, tmp_path / "ledger.jsonl")
    client = LLMClient(budget, tmp_path / "cache", transport)
    with pytest.raises(ConnectionError):
        client.create("judge", model="claude-haiku-4-5", max_tokens=10, messages=[])
    budget.reserve(0.99)  # nothing left reserved


def test_fallback_priced_at_higher_rate(tmp_path):
    budget = Budget(1.0, tmp_path / "ledger.jsonl")
    client = LLMClient(budget, tmp_path / "cache", lambda p: _resp(model="claude-opus-5"))
    client.create("agent", model="claude-sonnet-5", max_tokens=10, messages=[])
    assert budget.spent_usd == pytest.approx(cost_usd("claude-opus-5", _resp()["usage"]))
    entry = json.loads((tmp_path / "ledger.jsonl").read_text())
    assert entry["served_model"] == "claude-opus-5"


def test_cache_and_ledger_store_no_headers_or_secrets(tmp_path):
    def transport(params):
        r = _resp()
        r["headers"] = {"x-api-key": "sk-ant-SECRET", "authorization": "Bearer SECRET"}
        r["_request_id"] = "req_123"
        return r

    client = LLMClient(Budget(1.0, tmp_path / "ledger.jsonl"), tmp_path / "cache", transport)
    out = client.create("judge", model="claude-haiku-4-5", max_tokens=10, messages=[])
    assert "headers" not in out and "_request_id" not in out
    for f in [tmp_path / "ledger.jsonl", *(tmp_path / "cache").rglob("*.json")]:
        text = f.read_text()
        assert "SECRET" not in text and "sk-ant" not in text and "x-api-key" not in text
    assert "request_hash" in json.loads((tmp_path / "ledger.jsonl").read_text())


def test_split_params_routes_unknown_fields_to_extra_body():
    from certharness.llm.client import split_params

    def create(*, model, max_tokens, messages, extra_body=None):  # SDK-like signature
        pass

    out = split_params(create, {"model": "m", "max_tokens": 5, "messages": [], "temperature": 0.0})
    assert out == {
        "model": "m",
        "max_tokens": 5,
        "messages": [],
        "extra_body": {"temperature": 0.0},
    }
    assert "extra_body" not in split_params(create, {"model": "m", "max_tokens": 5, "messages": []})


def test_split_params_against_installed_sdk():
    import anthropic

    from certharness.graders.llm_judge import LLMJudge
    from certharness.llm.client import split_params
    from certharness.synthetic import generate

    create = anthropic.Anthropic(api_key="x").messages.create
    params = LLMJudge(client=None).request(generate(1)[0])
    kwargs = split_params(create, params)
    import inspect

    assert set(kwargs) - {"extra_body"} <= set(inspect.signature(create).parameters)
    flat = {k: v for k, v in kwargs.items() if k != "extra_body"} | kwargs.get("extra_body", {})
    assert flat == params  # nothing dropped or altered


def test_key_file_resolution(tmp_path, monkeypatch):
    from certharness.llm import credentials

    keyf = tmp_path / "k"
    monkeypatch.setenv("CERTHARNESS_KEY_FILE", str(keyf))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert credentials.key_source() is None and credentials.client_kwargs() == {}
    keyf.write_text("sk-test\n")
    keyf.chmod(0o644)
    with pytest.raises(PermissionError, match="chmod 600"):
        credentials.client_kwargs()
    keyf.chmod(0o600)
    assert credentials.client_kwargs() == {"api_key": "sk-test"}
    assert credentials.key_source() == str(keyf)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-env")
    assert credentials.client_kwargs() == {} and credentials.key_source() == "env"


_KEY_FORMS = [
    "sk-ant-api03-abc",
    "sk-ant-api03-abc\n",
    "ANTHROPIC_API_KEY=sk-ant-api03-abc",
    'export ANTHROPIC_API_KEY="sk-ant-api03-abc"',
    "ANTHROPIC_API_KEY='sk-ant-api03-abc'\n",
]


@pytest.mark.parametrize("text", _KEY_FORMS)
def test_parse_key_accepts_dotenv_forms(text):
    from certharness.llm.credentials import parse_key

    assert parse_key(text) == "sk-ant-api03-abc"
