"""Unit tests for the measured batch-size probe's decision logic.

The probe itself needs a real model on a real device, so these cover the parts
that decide *what to report* — OOM classification, the schema cap, and the
safety back-off — without loading anything.
"""

from __future__ import annotations

import pytest

import batch_probe
from schemas import DinoHyperParams, TunetHyperParams


@pytest.mark.parametrize(
    "message",
    [
        "MPS backend out of memory (MPS allocated: 100.00 GB)",
        "CUDA out of memory. Tried to allocate 2.00 GiB",
        "Invalid buffer size: insufficient memory",
        "RuntimeError: can't allocate memory",
        "Cannot allocate 4096 bytes",
    ],
)
def test_recognises_out_of_memory_messages(message: str) -> None:
    assert batch_probe._is_oom(RuntimeError(message))


@pytest.mark.parametrize(
    "message",
    [
        "mat1 and mat2 shapes cannot be multiplied",
        "Expected 4-dimensional input for 4-dimensional weight",
        "Checkpoint not found",
        "",
    ],
)
def test_real_bugs_are_not_mistaken_for_a_memory_ceiling(message: str) -> None:
    """Misreading a genuine error as OOM would silently report a wrong ceiling,
    so anything unrecognised must propagate instead."""
    assert not batch_probe._is_oom(RuntimeError(message))


def test_schema_cap_uses_each_family_s_own_bound() -> None:
    """A suggestion above the schema's limit would be rejected on submit."""
    assert batch_probe.schema_batch_cap(DinoHyperParams()) == 16
    assert batch_probe.schema_batch_cap(TunetHyperParams()) == 32


def test_schema_cap_is_still_bounded_by_the_probe_ceiling() -> None:
    class Boundless:
        pass

    assert batch_probe.schema_batch_cap(Boundless(), default=8) == 8


def test_schema_cap_survives_an_unrecognisable_model() -> None:
    """Introspection is best-effort — it must never break the probe."""
    assert batch_probe.schema_batch_cap(object()) == batch_probe._MAX_PROBE


@pytest.mark.parametrize(
    ("largest_ok", "cap", "expected"),
    [
        (1, 16, 1),    # never suggest 0
        (2, 16, 1),
        (4, 16, 3),
        (8, 16, 6),
        (16, 16, 12),
        (32, 32, 25),
    ],
)
def test_safety_backoff(largest_ok: int, cap: int, expected: int) -> None:
    """The probe runs synthetic zeros on an idle device, so the reported value
    keeps headroom below whatever actually completed."""
    suggested = batch_probe._suggest(largest_ok, cap)
    assert suggested == expected
    assert 1 <= suggested <= largest_ok


@pytest.mark.parametrize(("cap", "expected"), [(16, 5), (32, 6), (1, 1)])
def test_attempt_count_covers_the_doubling_sequence(cap: int, expected: int) -> None:
    """Progress is counted in attempts: 1,2,4,…,cap."""
    assert batch_probe._attempt_count(cap) == expected
