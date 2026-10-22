"""Tests for the model price table and its update script."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from actllminfer.pricing import ContextTier, ModelPrice, PriceTable, estimate_cost, get_price, load_prices

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
    assert raw["schema_version"] == 2
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


# ---------------------------------------------------------------------------
# Long-context tiers
# ---------------------------------------------------------------------------


def test_threshold_fields_become_context_tiers():
    built = update_prices.build(
        {"gemini/gemini-2.5-pro": _entry(
            "gemini", 1.25, 10, cache_read_input_token_cost=1.25e-7,
            input_cost_per_token_above_200k_tokens=2.5e-6,
            output_cost_per_token_above_200k_tokens=15e-6,
            cache_read_input_token_cost_above_200k_tokens=2.5e-7,
            input_cost_per_token_above_200k_tokens_priority=4.5e-6,   # another service tier: ignored
            input_cost_per_token_above_200k_tokens_batches=1.25e-6,  # batch pricing: ignored
        )},
        as_of="2026-10-22", source="t",
    )
    [m] = built["models"]
    assert m["context_tiers"] == [
        {"above_tokens": 200000, "input_per_mtok": 2.5, "output_per_mtok": 15.0, "cache_read_per_mtok": 0.25}
    ]


def test_a_tier_that_omits_a_rate_keeps_the_base_rate():
    built = update_prices.build(
        {"claude-x": _entry("anthropic", 1, 5, input_cost_per_token_above_100k_tokens=2e-6)},
        as_of="2026-10-22", source="t",
    )
    assert built["models"][0]["context_tiers"] == [
        {"above_tokens": 100000, "input_per_mtok": 2.0, "output_per_mtok": 5.0}
    ]


def test_tiered_pricing_ranges_supply_base_and_tiers():
    # Alibaba states only ranges; such models used to be dropped as unpriced.
    built = update_prices.build(
        {"dashscope/qwen3-max": {"litellm_provider": "dashscope", "mode": "chat", "tiered_pricing": [
            {"input_cost_per_token": 2.4e-6, "output_cost_per_token": 12e-6, "range": [32000, 128000]},
            {"input_cost_per_token": 1.2e-6, "output_cost_per_token": 6e-6, "range": [0, 32000]},
        ]}},
        as_of="2026-10-22", source="t",
    )
    [m] = built["models"]
    assert (m["input_per_mtok"], m["output_per_mtok"]) == (1.2, 6.0)
    assert m["context_tiers"] == [{"above_tokens": 32000, "input_per_mtok": 2.4, "output_per_mtok": 12.0}]


def _tiered(**kw) -> ModelPrice:
    return ModelPrice(
        provider="google", model="m", input_per_mtok=1.0, output_per_mtok=10.0, cache_read_per_mtok=0.1,
        context_tiers=(ContextTier(200_000, 2.0, 15.0, 0.2), ContextTier(1_000_000, 4.0, 30.0)), **kw,
    )


@pytest.mark.parametrize(
    "prompt_tokens, inclusive, rates",
    [
        (199_999, False, (1.0, 10.0, 0.1)),
        (200_000, False, (1.0, 10.0, 0.1)),   # exceeds, not reaches
        (200_000, True, (2.0, 15.0, 0.2)),    # xAI: reaching the threshold is enough
        (200_001, False, (2.0, 15.0, 0.2)),
        (2_000_000, False, (4.0, 30.0, None)),  # highest crossed tier wins
    ],
)
def test_rates_for_picks_the_highest_crossed_tier(prompt_tokens, inclusive, rates):
    assert _tiered(threshold_inclusive=inclusive).rates_for(prompt_tokens) == rates


def test_estimate_cost_bills_the_whole_request_at_the_tier_rate(monkeypatch):
    monkeypatch.setattr("actllminfer.pricing.get_price", lambda model, provider=None: _tiered())
    assert estimate_cost("m", 100_000, 1_000) == pytest.approx(0.1 + 0.01)
    assert estimate_cost("m", 300_000, 1_000) == pytest.approx(0.6 + 0.015)
    # Cached tokens count towards the threshold and use the tier's cache rate.
    assert estimate_cost("m", 300_000, 0, cached_prompt_tokens=100_000) == pytest.approx(0.4 + 0.02)
    # A tier with no cache rate bills cached tokens at its input rate.
    assert estimate_cost("m", 2_000_000, 0, cached_prompt_tokens=1_000_000) == pytest.approx(8.0)


def test_bundled_table_keeps_tiers_and_inclusive_providers():
    pro = get_price("gemini/gemini-2.5-pro")
    assert pro.context_tiers and pro.context_tiers[0].above_tokens == 200_000
    assert get_price("xai/grok-4.20").threshold_inclusive
    assert get_price("dashscope/qwen3-max") is not None


# ---------------------------------------------------------------------------
# Every stored field is compared
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "extra",
    [
        {"cache_read_input_token_cost": 5e-7},
        {"max_input_tokens": 2_000_000},
        {"deprecation_date": "2027-01-01"},
        {"supports_vision": True},
        {"input_cost_per_token_above_200k_tokens": 10e-6},
    ],
)
def test_non_price_changes_are_changes(tmp_path, extra):
    source, out = tmp_path / "litellm.json", tmp_path / "prices.json"
    args = ["--source", str(source), "--output", str(out), "--as-of", "2026-10-22"]
    source.write_text(json.dumps({"gpt-5.5": _entry("openai", 5, 30)}))
    assert update_prices.main(args) == 0
    source.write_text(json.dumps({"gpt-5.5": _entry("openai", 5, 30, **extra)}))
    assert update_prices.main([*args, "--check"]) == 1
    assert update_prices.main(args) == 0
    assert update_prices.main([*args, "--check"]) == 0  # rewritten, now current


def test_diff_names_the_changed_field():
    old = update_prices.build({"gpt-5.5": _entry("openai", 5, 30)}, as_of="2026-10-22", source="t")
    new = update_prices.build(
        {"gpt-5.5": _entry("openai", 5, 30, cache_read_input_token_cost=5e-7)}, as_of="2026-10-22", source="t"
    )
    assert update_prices.diff(old, new) == ["changed  openai/gpt-5.5: cache_read_per_mtok None -> 0.5"]
