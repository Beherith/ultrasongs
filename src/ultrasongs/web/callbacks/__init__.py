"""Feature-organized Dash callbacks."""

from .mode import register_mode_callbacks
from .pipeline import register_pipeline_callbacks
from .reference import register_reference_callbacks
from .settings import register_settings_callbacks

__all__ = [
    "register_mode_callbacks",
    "register_pipeline_callbacks",
    "register_reference_callbacks",
    "register_settings_callbacks",
]
