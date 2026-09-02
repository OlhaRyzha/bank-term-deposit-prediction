"""Shared project configuration."""

from collections.abc import Mapping
from typing import Final

TARGET_COLUMN: Final[str] = "y"
TARGET_MAPPING: Final[Mapping[str, int]] = {
    "no": 0,
    "yes": 1,
}
FEATURES_TO_DROP: Final[tuple[str, ...]] = ("duration",)
RANDOM_STATE: Final[int] = 42
