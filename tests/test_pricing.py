"""Tests for the model price table and its update script."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from actllminfer.pricing import PriceTable, estimate_cost, get_price, load_prices

ROOT = Path(__file__).resolve().parents[1]


def _load_script():
    spec = importlib.util.spec_from_file_location("update_prices", ROOT / "scripts" / "update_prices.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


update_prices = _load_script()


# ---------------------------------------------------------------------------
# Bundled table
# ---------------------------------------------------------------------------


def test_bundled_table_is_well_formed():
    table = load_prices()
    assert isinstance(table, PriceTable) and table.as_of and table.models
    keys = [(m.provider, m.model.lower()) for m in table.models]
    assert len(keys) == len(set(keys)), "duplicate provider/model entries"
    for m in table.models:
        assert m.input_per_mtok >= 0 and m.output_per_mtok >= 0, m
        assert not m.deprecation_date or m.deprecation_date >= table.as_of, f"retired: {m}"


def test_providers_with_a_client_are_marked():
    table = load_prices()
    assert {m.client for m in table.for_provider("anthropic")} == {"ChatAnthropic"}
    assert {m.client for m in table.for_provider("google")} == {None}  # reference only


def test_bundled_file_matches_what_the_script_writes():
    raw = json.loads((ROOT / "actllminfer" / "data" / "model_prices.json").read_text())
    assert raw["schema_version"] == 1
    assert raw["source"].startswith("https://")  # never a local path
    assert update_prices.dumps(raw) == (ROOT / "actllminfer" / "data" / "model_prices.json").read_text()


# ---------------------------------------------------------------------------
# Lookup and cost
# ---------------------------------------------------------------------------

TABLE = update_prices.build(
    {
        "claude-sonnet-4-6": {"litellm_provider": "anthropic", "mode": "chat",
                              "input_cost_per_token": 3e-6, "output_cost_per_token": 15e-6,
                              "cache_read_input_token_cost": 3e-7, "max_input_tokens": 1000000,
                              "supports_function_calling": True},
        "zai/glm-4.6": {"litellm_provider": "zai", "mode": "chat",
                        "input_cost_per_token": 6e-7, "output_cost_per_token": 2.2e-6},
        "minimax/MiniMax-M2.5": {"litellm_provider": "minimax", "mode": "chat",
                                 "input_cost_per_token": 3e-7, "output_cost_per_token": 1.2e-6},
        "moonshot/kimi-k2.5": {"litellm_provider": "moonshot", "mode": "chat",
                               "input_cost_per_token": 6e-7, "output_cost_per_token": 3e-6},
        "gemini/gemini-3.1-pro-preview": {"litellm_provider": "gemini", "mode": "chat",
                                          "input_cost_per_token": 2e-6, "output_cost_per_token": 12e-6},
    },
    as_of="2026-10-22",
    source="test",
)


@pytest.fixture()
def table(tmp_path):
    path = tmp_path / "prices.json"
    path.write_text(update_prices.dumps(TABLE))
    load_prices.cache_clear()
    yield load_prices(str(path))
    load_prices.cache_clear()


@pytest.mark.parametrize(
    "spec, provider, model",
    [
        ("anthropic/claude-sonnet-4-6", "anthropic", "claude-sonnet-4-6"),
        ("claude:claude-sonnet-4-6", "anthropic", "claude-sonnet-4-6"),
        ("glm-4.6", "zhipuai", "glm-4.6"),               # provider inferred from the name
        ("kimi:kimi-k2.5", "moonshot", "kimi-k2.5"),      # ActLLMInfer alias
        ("minimax-m2.5", "minimax", "MiniMax-M2.5"),      # case-insensitive id
        ("gemini/gemini-3.1-pro-preview", "google", "gemini-3.1-pro-preview"),
    ],
)
def test_get_price_resolves_provider_spellings(table, spec, provider, model):
    price = get_price(spec, table=table)
    assert (price.provider, price.model) == (provider, model)


def test_get_price_unknown_and_unpriced_models(table):
    assert get_price("openai/gpt-0", table=table) is None
    assert get_price("nvidia/meta/llama-3.3-70b-instruct", table=table) is None


def test_estimate_cost_uses_cache_read_price(monkeypatch, table):
    monkeypatch.setattr("actllminfer.pricing.load_prices", lambda path=None: table)
    full = estimate_cost("anthropic/claude-sonnet-4-6", 1_000_000, 100_000)
    assert full == pytest.approx(3.0 + 1.5)
    cached = estimate_cost("anthropic/claude-sonnet-4-6", 1_000_000, 0, cached_prompt_tokens=500_000)
    assert cached == pytest.approx(1.5 + 0.15)
    assert estimate_cost("openai/gpt-0", 10, 10) is None


# ---------------------------------------------------------------------------
# Update script
# ---------------------------------------------------------------------------


def _entry(provider, inp, out, **extra):
    return {"litellm_provider": provider, "mode": "chat",
            "input_cost_per_token": inp / 1e6, "output_cost_per_token": out / 1e6, **extra}


def test_build_reads_official_routes_only():
    built = update_prices.build(
        {
            "deepseek/deepseek-v4-pro": _entry("deepseek", 1.32, 3.96),
            "deepseek-v4-pro": _entry("deepseek", 1.32, 3.96),         # bare alias: merged
            "azure_ai/deepseek-v4-pro": _entry("azure_ai", 1.74, 3.48),  # reseller: ignored
            "text-embedding-3-small": {**_entry("openai", 0.02, 0), "mode": "embedding"},
            "gpt-5.5": _entry("openai", 5, 30),
        },
        as_of="2026-10-22",
        source="test",
    )
    models = {(m["provider"], m["model"]): m for m in built["models"]}
    assert set(models) == {("deepseek", "deepseek-v4-pro"), ("openai", "gpt-5.5")}
    assert models[("deepseek", "deepseek-v4-pro")]["litellm_key"] == "deepseek/deepseek-v4-pro"


def test_build_drops_models_past_their_deprecation_date():
    built = update_prices.build(
        {
            "deepseek/deepseek-chat": _entry("deepseek", 0.28, 0.42, deprecation_date="2026-07-24"),
            "claude-sonnet-4-5": _entry("anthropic", 3, 15, deprecation_date="2026-11-30"),
        },
        as_of="2026-10-22",
        source="test",
    )
    assert [(m["model"], m.get("deprecation_date")) for m in built["models"]] == [
        ("claude-sonnet-4-5", "2026-11-30")
    ]


def test_diff_reports_added_removed_and_repriced():
    old = update_prices.build({"gpt-5.5": _entry("openai", 5, 30), "gpt-4o": _entry("openai", 2.5, 10)},
                              as_of="2026-05-14", source="t")
    new = update_prices.build({"gpt-5.5": _entry("openai", 4, 24), "gpt-5.6": _entry("openai", 5, 30)},
                              as_of="2026-10-22", source="t")
    assert update_prices.diff(old, new) == [
        "added    openai/gpt-5.6",
        "removed  openai/gpt-4o",
        "price    openai/gpt-5.5: 5.0/30.0 -> 4.0/24.0",
    ]


def test_script_check_mode(tmp_path):
    source = tmp_path / "litellm.json"
    source.write_text(json.dumps({"gpt-5.5": _entry("openai", 5, 30)}))
    out = tmp_path / "prices.json"
    args = ["--source", str(source), "--output", str(out), "--as-of", "2026-10-22"]
    assert update_prices.main(args) == 0 and out.exists()
    assert json.loads(out.read_text())["source"] == update_prices.LITELLM_URL
    assert update_prices.main([*args, "--check"]) == 0
    source.write_text(json.dumps({"gpt-5.5": _entry("openai", 4, 24)}))
    assert update_prices.main([*args, "--check"]) == 1
