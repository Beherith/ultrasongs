"""Advanced-settings collection, coercion, and validation."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from dash import ALL, Dash, Input, Output, State, ctx, no_update

from ultrasongs.config import AppSettings


def coerce_setting_value(metadata: dict[str, Any], value: Any) -> Any:
    """Coerce a Dash control value using its central JSON-schema metadata."""

    value_type = metadata.get("type")
    nullable = any(option.get("type") == "null" for option in metadata.get("anyOf", []))
    if nullable and (value is None or (isinstance(value, str) and not value.strip())):
        return None
    if value_type == "array":
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError as exc:
                raise ValueError("must be a JSON array") from exc
        if not isinstance(value, list | tuple):
            raise ValueError("must be a JSON array")
        return list(value)
    if value_type == "integer":
        return int(value)
    if value_type == "number":
        return float(value)
    if value_type == "boolean":
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.lower() in {"true", "false"}:
            return value.lower() == "true"
        raise ValueError("must be enabled or disabled")
    return value


def collect_ui_overrides(
    settings: AppSettings,
    component_ids: Sequence[dict[str, str]],
    values: Sequence[Any],
) -> dict[str, Any]:
    """Build and fully validate a flat override document from pattern IDs."""

    if len(component_ids) != len(values):
        raise ValueError("advanced setting controls are incomplete")
    schema = settings.ui_override_schema()
    overrides: dict[str, Any] = {}
    for component_id, value in zip(component_ids, values, strict=True):
        path = component_id.get("path", "")
        if path not in schema:
            raise ValueError(f"unknown UI setting: {path}")
        try:
            overrides[path] = coerce_setting_value(schema[path], value)
        except (TypeError, ValueError) as exc:
            title = schema[path].get("title", path)
            raise ValueError(f"{title}: {exc}") from exc
    try:
        settings.apply_ui_overrides(overrides)
    except ValueError as exc:
        error_text = str(exc)
        title = next(
            (
                str(metadata.get("title", path))
                for path, metadata in schema.items()
                if path in error_text
            ),
            "Advanced settings",
        )
        raise ValueError(f"{title}: {exc}") from exc
    return overrides


def default_control_values(
    settings: AppSettings, component_ids: Sequence[dict[str, str]]
) -> list[Any]:
    schema = settings.ui_override_schema()
    defaults: list[Any] = []
    for component_id in component_ids:
        path = component_id["path"]
        value = schema[path]["default"]
        defaults.append(json.dumps(value) if schema[path].get("type") == "array" else value)
    return defaults


def register_settings_callbacks(app: Dash, settings: AppSettings) -> None:
    @app.callback(
        Output("settings-overrides-store", "data"),
        Output("settings-status", "children"),
        Output({"type": "setting-input", "path": ALL}, "value"),
        Input("apply-settings-button", "n_clicks"),
        Input("reset-settings-button", "n_clicks"),
        State({"type": "setting-input", "path": ALL}, "id"),
        State({"type": "setting-input", "path": ALL}, "value"),
        prevent_initial_call=True,
    )
    def update_settings(
        _apply_clicks: int,
        _reset_clicks: int,
        component_ids: list[dict[str, str]],
        values: list[Any],
    ):
        if ctx.triggered_id == "reset-settings-button":
            return {}, "Using central defaults", default_control_values(settings, component_ids)
        try:
            overrides = collect_ui_overrides(settings, component_ids, values)
        except ValueError as exc:
            return no_update, f"Settings were not applied: {exc}", no_update
        return overrides, f"Applied {len(overrides)} run settings", no_update


__all__ = [
    "coerce_setting_value",
    "collect_ui_overrides",
    "default_control_values",
    "register_settings_callbacks",
]
