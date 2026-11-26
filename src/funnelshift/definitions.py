"""Funnel definitions loaded from YAML and validated before any SQL is rendered.

Every identifier that ends up in generated SQL (funnel name, event names, property names,
segment columns) must match ``[a-z_][a-z0-9_]*``; property names and segment columns must also
be columns that exist in the event table. Property values are restricted to a small safe
character set and are quoted again at render time.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated, Self

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

IDENTIFIER = re.compile(r"^[a-z_][a-z0-9_]*$")

#: Event columns a step filter may test.
EVENT_PROPERTIES: tuple[str, ...] = ("platform", "source", "plan", "page")
#: Per-user attributes (taken from the first signup) that results can be split by.
SEGMENT_COLUMNS: tuple[str, ...] = ("platform", "source")
#: The event that defines a user's cohort week and segment.
SIGNUP_EVENT = "signup"

MAX_STEPS = 8

Identifier = Annotated[str, StringConstraints(pattern=r"^[a-z_][a-z0-9_]*$", max_length=63)]
PropertyValue = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_. -]{1,64}$")]


class Step(BaseModel):
    """One funnel step: an event name plus optional property-equality filters."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    event: Identifier
    where: dict[Identifier, PropertyValue] = Field(default_factory=dict)

    @field_validator("where")
    @classmethod
    def _known_properties(cls, value: dict[str, str]) -> dict[str, str]:
        unknown = sorted(set(value) - set(EVENT_PROPERTIES))
        if unknown:
            raise ValueError(
                f"unknown event properties {unknown}; known properties: {list(EVENT_PROPERTIES)}"
            )
        return value


class Funnel(BaseModel):
    """An ordered funnel with a conversion window measured from the first step."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: Identifier
    window_days: int = Field(default=14, ge=1, le=365)
    steps: list[Step] = Field(min_length=2, max_length=MAX_STEPS)
    segments: list[Identifier] = Field(default_factory=lambda: list(SEGMENT_COLUMNS))

    @model_validator(mode="after")
    def _check(self) -> Self:
        names = [step.event for step in self.steps]
        repeated = sorted({n for n in names if names.count(n) > 1})
        if repeated:
            raise ValueError(
                f"event names may not repeat in a funnel (a repeated name is ambiguous "
                f"about which occurrence matches which step): {repeated}"
            )
        validate_segments(self.segments)
        return self

    @property
    def step_names(self) -> list[str]:
        return [step.event for step in self.steps]


def validate_segments(segments: list[str]) -> list[str]:
    """Check a list of segment columns; used for both YAML and ``--by`` overrides."""
    if not segments:
        raise ValueError("at least one segment column is required")
    if len(set(segments)) != len(segments):
        raise ValueError(f"duplicate segment columns: {segments}")
    for column in segments:
        if not IDENTIFIER.match(column):
            raise ValueError(f"segment column is not a valid identifier: {column!r}")
        if column not in SEGMENT_COLUMNS:
            raise ValueError(f"unknown segment column {column!r}; known: {list(SEGMENT_COLUMNS)}")
    return segments


def parse_funnel(text: str) -> Funnel:
    """Parse and validate YAML text. Raises ``pydantic.ValidationError`` on any problem."""
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError("a funnel definition must be a YAML mapping")
    return Funnel.model_validate(data)


def load_funnel(path: str | Path) -> Funnel:
    return parse_funnel(Path(path).read_text(encoding="utf-8"))
