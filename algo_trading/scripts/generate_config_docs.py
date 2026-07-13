"""
scripts/generate_config_docs.py — Generate configuration reference documentation.

Introspects the Pydantic Settings class and generates a Markdown table
of all config fields with their types, defaults, and descriptions.

P2-FIX-91: Config schema docs generated from code.

Usage:
    python -m algo_trading.scripts.generate_config_docs
    python -m algo_trading.scripts.generate_config_docs --output docs/CONFIG_REFERENCE.md
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)

# ── Type helpers ─────────────────────────────────────────────────────────────

def _type_name(annotation: Any) -> str:
    """
    Return a human-readable type name for a Pydantic field annotation.

    Handles common patterns: Optional[X], Literal[...], list[X], dict[K, V],
    and bare types (str, int, bool, Decimal, SecretStr, etc.).
    """
    import typing

    origin = getattr(annotation, "__origin__", None)

    # Optional[X] → "X (optional)"
    if origin is typing.Union:
        args = [a for a in annotation.__args__ if a is not type(None)]
        none_args = [a for a in annotation.__args__ if a is type(None)]
        inner = ", ".join(_type_name(a) for a in args)
        if none_args:
            return f"{inner} (optional)"
        return inner

    # Literal[...] → "Literal['a', 'b']"
    if origin is typing.Literal:
        choices = ", ".join(repr(v) for v in annotation.__args__)
        return f"Literal[{choices}]"

    # list[X] → "list[X]"
    if origin is list:
        inner = _type_name(annotation.__args__[0]) if annotation.__args__ else "Any"
        return f"list[{inner}]"

    # dict[K, V] → "dict[K, V]"
    if origin is dict:
        args = annotation.__args__ if annotation.__args__ else ("Any", "Any")
        k = _type_name(args[0])
        v = _type_name(args[1])
        return f"dict[{k}, {v}]"

    # Plain class → use __name__
    name = getattr(annotation, "__name__", None)
    if name:
        return name

    return str(annotation)


def _is_secret(field_name: str, field_info: Any) -> bool:
    """
    Return True if the field holds a secret value (SecretStr or Optional[SecretStr]).

    Checked by field name convention (ends with _password, _secret, _token,
    _key, _api_secret) AND by annotation type inspection.
    """
    import pydantic

    SECRET_SUFFIXES = (
        "_password", "_secret", "_token", "_api_secret", "_key",
    )
    if any(field_name.lower().endswith(s) for s in SECRET_SUFFIXES):
        return True

    # Also check annotation for SecretStr
    ann = field_info.annotation
    try:
        import typing
        origin = getattr(ann, "__origin__", None)
        if origin is typing.Union:
            types_in_union = ann.__args__
        else:
            types_in_union = (ann,)
        for t in types_in_union:
            if t is pydantic.SecretStr:
                return True
    except Exception:
        pass

    return False


def _default_repr(field_name: str, field_info: Any, is_secret: bool) -> str:
    """
    Return the human-readable default value string.

    Secrets always show [SECRET]. Required fields show *(required)*.
    """
    if is_secret:
        return "[SECRET]"

    # Pydantic v2: field_info.default is PydanticUndefined when there is no default
    try:
        from pydantic_core import PydanticUndefinedType
        undefined_type = PydanticUndefinedType
    except ImportError:
        undefined_type = type(None)  # fallback

    default = field_info.default

    # Check for PydanticUndefined (required field)
    try:
        from pydantic.fields import PydanticUndefined
        if default is PydanticUndefined:
            return "*(required)*"
    except ImportError:
        pass

    if default is None:
        return "`None`"

    # default_factory — callable that produces a default
    factory = getattr(field_info, "default_factory", None)
    if factory is not None and callable(factory):
        try:
            val = factory()
            return f"`{val!r}`"
        except Exception:
            return "*(factory)*"

    if isinstance(default, str):
        return f"`{default!r}`"
    if isinstance(default, bool):
        return f"`{default}`"
    if default is not None:
        return f"`{default!r}`"

    return "`None`"


def _category_for_field(field_name: str) -> str:
    """
    Assign a category label based on the field name prefix.

    Used to group related fields in the output table.
    """
    PREFIXES: list[tuple[str, str]] = [
        ("shoonya_", "Shoonya Broker"),
        ("truedata_", "TrueData Feed"),
        ("trading_", "Trading Mode"),
        ("enable_", "Trading Mode"),
        ("data_feed", "Data Feed"),
        ("use_micro", "Data Feed"),
        ("bar_", "Bar Config"),
        ("session_", "Session / Calendar"),
        ("no_new_", "Session / Calendar"),
        ("force_", "Session / Calendar"),
        ("skip_", "Session / Calendar"),
        ("strategy_", "Compliance"),
        ("broker_", "Compliance"),
        ("approved_", "Compliance"),
        ("otr_", "Compliance"),
        ("audit_", "Compliance"),
        ("kill_switch", "Compliance"),
        ("capital_", "Risk"),
        ("max_daily_", "Risk"),
        ("max_position_", "Risk"),
        ("max_gross_", "Risk"),
        ("max_drawdown_", "Risk"),
        ("max_orders_", "Execution"),
        ("stale_", "Execution"),
        ("max_price_", "Execution"),
        ("max_order_", "Execution"),
        ("product_", "Execution"),
        ("exchange_", "Execution"),
        ("ema_", "Strategy (EMA Cross)"),
        ("atr_", "Strategy (EMA Cross)"),
        ("min_bars_", "Strategy (EMA Cross)"),
        ("zmq_", "ZeroMQ IPC"),
        ("questdb_", "QuestDB"),
        ("telegram_", "Alerts"),
        ("log_", "System"),
        ("symbols", "System"),
        ("log_dir", "System"),
    ]
    for prefix, category in PREFIXES:
        if field_name.startswith(prefix) or field_name == prefix.rstrip("_"):
            return category
    return "Other"


# ── Introspection ─────────────────────────────────────────────────────────────

def _get_model_fields() -> dict[str, Any]:
    """
    Return the Pydantic model_fields dict for the Settings class.

    The module-level ``settings = Settings()`` call in config.py runs at import
    time and requires a valid .env file.  We bypass this by loading the module
    source text, compiling it without executing the module-level singleton line,
    and then extracting the class definition.

    Fallback chain:
    1. Compile config.py with the singleton line suppressed → extract Settings class.
    2. Direct ``from algo_trading.config import Settings`` (works when .env is valid).
    3. Return empty dict and log a warning.
    """
    import importlib
    import sys
    import types

    # Strategy 1: Load the source and suppress the singleton instantiation.
    try:
        import inspect
        from pathlib import Path as _Path
        config_path = (
            _Path(__file__).parent.parent.parent
            / "algo_trading" / "config.py"
        )
        source = config_path.read_text(encoding="utf-8")

        # Replace the module-level singleton call with a no-op assignment so the
        # module can be compiled and executed without needing a .env file.
        # The class definition is fully defined before that line.
        patched = source.replace(
            "settings: Settings = Settings()  # type: ignore[call-arg]",
            "settings = None  # patched by generate_config_docs.py",
        )

        module_name = "_algo_trading_config_stub"
        stub_module = types.ModuleType(module_name)
        stub_module.__file__ = str(config_path)

        # Provide a dummy .env path so pydantic_settings doesn't error on the
        # SettingsConfigDict itself.
        exec(compile(patched, str(config_path), "exec"), stub_module.__dict__)  # noqa: S102

        settings_cls = stub_module.__dict__.get("Settings")
        if settings_cls is not None and hasattr(settings_cls, "model_fields"):
            log.debug("_get_model_fields: loaded via source patching")
            return settings_cls.model_fields
    except Exception as exc_patch:
        log.debug("Source-patching strategy failed: %s", exc_patch)

    # Strategy 2: Normal import (works when .env exists and is valid).
    try:
        # If already imported, use the cached module; otherwise import fresh.
        if "algo_trading.config" in sys.modules:
            cfg_mod = sys.modules["algo_trading.config"]
        else:
            cfg_mod = importlib.import_module("algo_trading.config")
        settings_cls = getattr(cfg_mod, "Settings", None)
        if settings_cls is not None and hasattr(settings_cls, "model_fields"):
            log.debug("_get_model_fields: loaded via normal import")
            return settings_cls.model_fields
    except Exception as exc_import:
        log.debug("Normal import strategy failed: %s", exc_import)

    log.warning(
        "Could not introspect Settings class — generating empty docs. "
        "Run with a valid .env file or ensure pydantic-settings is installed."
    )
    return {}


def _collect_field_rows(model_fields: dict[str, Any]) -> list[dict[str, str]]:
    """
    Transform model_fields into a list of row dicts for the Markdown table.

    Each row has: field, category, type, default, required, secret, description.
    """
    rows: list[dict[str, str]] = []

    for field_name, field_info in model_fields.items():
        secret = _is_secret(field_name, field_info)
        type_str = _type_name(field_info.annotation)
        default_str = _default_repr(field_name, field_info, secret)
        description = (field_info.description or "").strip() if field_info.description else ""

        # Required = no default and not a secret-masked field
        is_required = default_str == "*(required)*"

        rows.append({
            "field": field_name,
            "env_var": field_name.upper(),
            "category": _category_for_field(field_name),
            "type": type_str,
            "default": default_str,
            "required": "Yes" if is_required else "No",
            "secret": "Yes" if secret else "No",
            "description": description,
        })

    return rows


# ── Markdown rendering ────────────────────────────────────────────────────────

def _render_markdown(rows: list[dict[str, str]], project_name: str = "NSE AlgoTrader") -> str:
    """
    Render the field list as a categorised Markdown document.

    Produces:
      - A header section
      - One H2 section per category
      - A Markdown table per category
    """
    from collections import defaultdict

    lines: list[str] = []

    lines.append(f"# {project_name} — Configuration Reference")
    lines.append("")
    lines.append(
        "> **Auto-generated** by `python -m algo_trading.scripts.generate_config_docs`  "
    )
    lines.append(
        "> Do not edit manually — update `algo_trading/config.py` and regenerate."
    )
    lines.append("")
    lines.append("## Legend")
    lines.append("")
    lines.append("| Symbol | Meaning |")
    lines.append("|--------|---------|")
    lines.append("| *(required)* | Field has no default; must be set in `.env` |")
    lines.append("| `[SECRET]` | Value is never shown — stored as `SecretStr` |")
    lines.append("| `None` | Default is Python `None` (optional field) |")
    lines.append("")

    # Group by category, preserving insertion order
    by_category: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_category[row["category"]].append(row)

    for category, cat_rows in by_category.items():
        lines.append(f"## {category}")
        lines.append("")
        lines.append("| Field (env var) | Type | Default | Required | Secret | Description |")
        lines.append("|----------------|------|---------|----------|--------|-------------|")

        for row in cat_rows:
            env_var = f"`{row['env_var']}`"
            type_col = row["type"]
            default_col = row["default"]
            required_col = row["required"]
            secret_col = "**Yes**" if row["secret"] == "Yes" else "No"
            desc = row["description"].replace("|", "\\|") if row["description"] else "—"

            lines.append(
                f"| {env_var} | {type_col} | {default_col} | {required_col} | {secret_col} | {desc} |"
            )

        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append(
        "_Generated from `algo_trading/config.py` — "
        f"{len(rows)} fields across {len(by_category)} categories._"
    )
    lines.append("")

    return "\n".join(lines)


# ── Entry point ───────────────────────────────────────────────────────────────

def main(argv: Optional[list[str]] = None) -> int:
    """
    CLI entry point.

    Returns 0 on success, 1 on error.
    """
    parser = argparse.ArgumentParser(
        description="Generate Markdown config reference from algo_trading.config.Settings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--output",
        default="docs/CONFIG_REFERENCE.md",
        metavar="PATH",
        help="Output file path (default: docs/CONFIG_REFERENCE.md)",
    )
    parser.add_argument(
        "--project-name",
        default="NSE AlgoTrader",
        metavar="NAME",
        help="Project name used in the document header",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Print output to stdout instead of a file",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    log.info("Introspecting algo_trading.config.Settings ...")
    model_fields = _get_model_fields()

    if not model_fields:
        log.error("No fields found in Settings — check that pydantic-settings is installed.")
        return 1

    log.info("Found %d fields", len(model_fields))
    rows = _collect_field_rows(model_fields)
    markdown = _render_markdown(rows, project_name=args.project_name)

    if args.stdout:
        print(markdown)
        return 0

    output_path = Path(args.output)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(markdown, encoding="utf-8")
        log.info("Written to %s (%d bytes)", output_path, len(markdown))
    except OSError as exc:
        log.error("Failed to write %s: %s", output_path, exc)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
