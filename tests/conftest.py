"""Shared pytest options for opt-in, real-model end-to-end checks."""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("ultrasongs end-to-end")
    group.addoption(
        "--e2e-audio",
        help="MP3/audio fixture for the opt-in repair CLI end-to-end test",
    )
    group.addoption(
        "--e2e-song",
        help="existing UltraStar TXT fixture for the opt-in repair CLI test",
    )
    group.addoption(
        "--e2e-config",
        help="optional JSON/TOML configuration for the real-model repair test",
    )
