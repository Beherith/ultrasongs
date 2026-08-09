"""Generation/validation mode presentation."""

from __future__ import annotations

from dataclasses import dataclass

from dash import Dash, Input, Output


@dataclass(frozen=True, slots=True)
class ModePresentation:
    reference_style: dict[str, str]
    description: str
    action_label: str


def mode_presentation(mode: str | None) -> ModePresentation:
    if mode == "validate":
        return ModePresentation(
            reference_style={"display": "block"},
            description=(
                "Run the complete pipeline, compare its output with a reference UltraStar "
                "TXT, and produce a visual similarity report."
            ),
            action_label="Run validation",
        )
    return ModePresentation(
        reference_style={"display": "none"},
        description=(
            "Create a new UltraStar song from audio or video, supplied lyrics, and song metadata."
        ),
        action_label="Generate UltraStar song",
    )


def register_mode_callbacks(app: Dash) -> None:
    @app.callback(
        Output("reference-upload-section", "style"),
        Output("mode-description", "children"),
        Output("run-button", "children"),
        Input("mode-selector", "value"),
    )
    def switch_mode(mode: str | None) -> tuple[dict[str, str], str, str]:
        presentation = mode_presentation(mode)
        return (
            presentation.reference_style,
            presentation.description,
            presentation.action_label,
        )


__all__ = ["ModePresentation", "mode_presentation", "register_mode_callbacks"]
