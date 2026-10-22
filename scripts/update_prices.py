"""Regenerate actllminfer/data/model_prices.json from LiteLLM's price table.

Each provider is read at its *official* route only (``deepseek/`` for
DeepSeek, ``zai/`` for ZhipuAI, ...), never a reseller's, so a price here is
what calling the provider directly costs. Providers ActLLMInfer has a client
for are marked with that client; the others (Google, xAI, Mistral, Cohere,
Alibaba) are listed as reference prices for routers such as ActRouter.

Models that disappear from the official route, or whose deprecation date
has passed, are dropped and new ones are added, so re-running this script is
how the table follows the providers.

    python scripts/update_prices.py                  # fetch LiteLLM main
    python scripts/update_prices.py --source prices.json
    python scripts/update_prices.py --check          # exit 1 if out of date
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
import urllib.request
from pathlib import Path
from typing import Any, Optional

LITELLM_URL = (
    "https://raw.githubusercontent.com/BerriAI/litellm/main/"
    "model_prices_and_context_window.json"
)
DATA = Path(__file__).resolve().parents[1] / "actllminfer" / "data" / "model_prices.json"

#: our provider name -> (LiteLLM provider, LiteLLM key prefix, ActLLMInfer client)
PROVIDERS: dict[str, tuple[str, str, Optional[str]]] = {
    "openai": ("openai", "", "ChatOpenAI"),
    "anthropic": ("anthropic", "", "ChatAnthropic"),
    "deepseek": ("deepseek", "deepseek/", "ChatDeepSeek"),
    "moonshot": ("moonshot", "moonshot/", "ChatMoonshot"),
    "zhipuai": ("zai", "zai/", "ChatZhipuAI"),
    "minimax": ("minimax", "minimax/", "ChatMiniMax"),
    # Reference prices: no ActLLMInfer client (yet).
    "google": ("gemini", "gemini/", None),
    "xai": ("xai", "xai/", None),
    "mistral": ("mistral", "mistral/", None),
    "cohere": ("cohere_chat", "", None),
    "alibaba": ("dashscope", "dashscope/", None),
}
# NVIDIA NIM and Hugging Face publish no per-token list prices, so they have
# no entries; see actllminfer.pricing for how callers treat that.


#: providers whose long-context rate applies once the prompt *reaches* the
#: threshold rather than exceeds it (LiteLLM's ``_INCLUSIVE_THRESHOLD_PROVIDERS``)
INCLUSIVE_THRESHOLDS = frozenset({"xai"})

#: standard-tier long-context rates, e.g. ``input_cost_per_token_above_200k_tokens``;
#: ``_batches`` / ``_priority`` / ``_flex`` variants are other service tiers
_THRESHOLD_KEY = re.compile(
    r"^(input_cost_per_token|output_cost_per_token|cache_read_input_token_cost)"
    r"_above_(\d+)(k?)_tokens$"
)
_RATE_FIELD = {
    "input_cost_per_token": "input_per_mtok",
    "output_cost_per_token": "output_per_mtok",
    "cache_read_input_token_cost": "cache_read_per_mtok",
}


def _mtok(per_token: Any) -> Optional[float]:
    return None if per_token is None else round(float(per_token) * 1e6, 6)


def _rates(e: dict[str, Any]) -> Optional[tuple[dict[str, float], list[dict[str, Any]]]]:
    """(base rates, context tiers) in USD per Mtok, or None if unpriced.

    LiteLLM states long-context pricing two ways, both all-or-nothing per
    request -- once the prompt crosses a threshold, every token in it is
    billed at that tier's rates:

    * ``*_above_{N}k_tokens`` fields next to the base rates;
    * a ``tiered_pricing`` list of token ``range``s (Alibaba), which also
      carries the base rate: the first range is the base, the rest are tiers.
    """
    tiers_by_threshold: dict[int, dict[str, Any]] = {}
    tiered = e.get("tiered_pricing")
    if isinstance(tiered, list) and tiered:
        ranges = sorted(
            (t for t in tiered if len(t.get("range") or []) == 2
             and t.get("input_cost_per_token") is not None and t.get("output_cost_per_token") is not None),
            key=lambda t: t["range"][0],
        )
        if not ranges:
            return None
        first, rest = ranges[0], ranges[1:]
        base = {
            "input_per_mtok": _mtok(first["input_cost_per_token"]),
            "output_per_mtok": _mtok(first["output_cost_per_token"]),
            "cache_read_per_mtok": _mtok(first.get("cache_read_input_token_cost", e.get("cache_read_input_token_cost"))),
        }
        for t in rest:
            tiers_by_threshold[int(t["range"][0])] = {
                field: _mtok(t.get(key)) for key, field in _RATE_FIELD.items()
            }
    else:
        if e.get("input_cost_per_token") is None or e.get("output_cost_per_token") is None:
            return None
        base = {field: _mtok(e.get(key)) for key, field in _RATE_FIELD.items()}
        for key, value in e.items():
            match = _THRESHOLD_KEY.match(key)
            if match and value is not None:
                threshold = int(match.group(2)) * (1000 if match.group(3) else 1)
                tiers_by_threshold.setdefault(threshold, {})[_RATE_FIELD[match.group(1)]] = _mtok(value)
    tiers = []
    for threshold in sorted(tiers_by_threshold):
        rates = tiers_by_threshold[threshold]
        tier = {
            "above_tokens": threshold,
            # A rate a tier does not restate stays at the base rate, as in LiteLLM.
            "input_per_mtok": rates.get("input_per_mtok") if rates.get("input_per_mtok") is not None else base["input_per_mtok"],
            "output_per_mtok": rates.get("output_per_mtok") if rates.get("output_per_mtok") is not None else base["output_per_mtok"],
            "cache_read_per_mtok": rates.get("cache_read_per_mtok") if rates.get("cache_read_per_mtok") is not None else base["cache_read_per_mtok"],
        }
        tiers.append({k: v for k, v in tier.items() if v is not None})
    return {k: v for k, v in base.items() if v is not None}, tiers


def build(litellm: dict[str, Any], as_of: str, source: str) -> dict[str, Any]:
    models: dict[tuple[str, str], dict[str, Any]] = {}
    for key, e in litellm.items():
        if not isinstance(e, dict) or e.get("mode") != "chat":
            continue
        priced = _rates(e)
        if priced is None:
            continue
        base, tiers = priced
        if e.get("deprecation_date") and str(e["deprecation_date"]) < as_of:
            continue  # retired: listed upstream, but no longer callable
        for provider, (lit_provider, prefix, _) in PROVIDERS.items():
            if e.get("litellm_provider") != lit_provider:
                continue
            model = key[len(prefix):] if prefix and key.startswith(prefix) else key
            if "/" in model:
                break  # another route's key that LiteLLM files under this provider
            entry = {
                "provider": provider,
                "model": model,
                **base,
                "context_tiers": tiers or None,
                "context_window": e.get("max_input_tokens"),
                "max_output_tokens": e.get("max_output_tokens"),
                "supports_tools": bool(e.get("supports_function_calling")),
                "supports_vision": bool(e.get("supports_vision")),
                "supports_reasoning": bool(e.get("supports_reasoning")),
                "deprecation_date": e.get("deprecation_date"),
                "litellm_key": key,
            }
            entry = {k: v for k, v in entry.items() if v is not None}
            existing = models.get((provider, model.lower()))
            # Prefer the prefixed key ("deepseek/deepseek-chat") over a bare alias.
            if existing is None or (prefix and key.startswith(prefix)):
                models[(provider, model.lower())] = entry
            break
    return {
        "schema_version": 2,  # 2: context_tiers, providers.*.threshold_inclusive
        "as_of": as_of,
        "source": source,
        "providers": {
            name: {
                "client": client,
                "litellm_provider": lit,
                "threshold_inclusive": lit in INCLUSIVE_THRESHOLDS,
            }
            for name, (lit, _, client) in PROVIDERS.items()
        },
        "models": [models[k] for k in sorted(models)],
    }


def diff(old: dict[str, Any], new: dict[str, Any]) -> list[str]:
    def index(table: dict[str, Any]) -> dict[str, dict[str, Any]]:
        return {f"{m['provider']}/{m['model']}": m for m in table.get("models", [])}

    before, after = index(old), index(new)
    lines = []
    for field in ("schema_version", "providers"):
        if old.get(field) != new.get(field):
            lines.append(f"format   {field} changed")
    lines += [f"added    {k}" for k in sorted(set(after) - set(before))]
    lines += [f"removed  {k}" for k in sorted(set(before) - set(after))]
    # Every stored field counts: a cache-rate, tier, context-window,
    # deprecation-date or capability change must reach the table too.
    for k in sorted(set(before) & set(after)):
        a, b = before[k], after[k]
        if (a.get("input_per_mtok"), a.get("output_per_mtok")) != (b.get("input_per_mtok"), b.get("output_per_mtok")):
            lines.append(
                f"price    {k}: {a.get('input_per_mtok')}/{a.get('output_per_mtok')} -> "
                f"{b.get('input_per_mtok')}/{b.get('output_per_mtok')}"
            )
        for field in sorted((set(a) | set(b)) - {"provider", "model", "input_per_mtok", "output_per_mtok"}):
            if a.get(field) != b.get(field):
                lines.append(f"changed  {k}: {field} {a.get(field)!r} -> {b.get(field)!r}")
    return lines


def dumps(table: dict[str, Any]) -> str:
    return json.dumps(table, indent=1, sort_keys=False) + "\n"


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--source", default=LITELLM_URL, help="LiteLLM price JSON (file or URL)")
    parser.add_argument("--output", default=str(DATA))
    parser.add_argument("--check", action="store_true", help="exit 1 if the table is out of date")
    parser.add_argument("--as-of", default=dt.date.today().isoformat())
    args = parser.parse_args(argv)

    if Path(args.source).exists():
        litellm = json.loads(Path(args.source).read_text())
    else:
        with urllib.request.urlopen(args.source, timeout=60) as response:
            litellm = json.loads(response.read().decode())

    out = Path(args.output)
    old = json.loads(out.read_text()) if out.exists() else {}
    # The data always names the upstream table, even when built from a local copy of it.
    new = build(litellm, args.as_of, LITELLM_URL)
    changes = diff(old, new)
    for line in changes:
        print(line)
    print(f"{len(new['models'])} models, {len(changes)} changes")
    if args.check:
        return 1 if changes else 0
    if changes or not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(dumps(new))
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
