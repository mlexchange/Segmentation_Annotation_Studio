"""How a batch of uploaded images maps onto samples.

Browse treats each container under the ingest root as one *sample* and its array
children as that sample's slices. Before grouping existed, every upload landed in
a single container, so uploading 47 images of three different specimens produced
one 47-slice "sample" — and because those images had different pixel dimensions,
the stack could not even be read past its first slice.
"""

from __future__ import annotations

import pytest

from ingest import (
    DEFAULT_GROUPING,
    GROUPING_MODES,
    plan_grouping,
    sample_name_for,
    validate_grouping,
)

# The batch shape that motivated this: three specimens in one drop.
MIXED_BATCH = [
    "almond_control__slice_01.tif",
    "almond_control__slice_02.tif",
    "almond_drought__slice_01.tif",
    "lantana_camara__LaCa5_5x_Slide_279.png",
]


def test_default_is_prefix_grouping() -> None:
    assert DEFAULT_GROUPING == "prefix"
    assert validate_grouping("") == "prefix"
    assert validate_grouping(None) == "prefix"  # type: ignore[arg-type]


@pytest.mark.parametrize("mode", GROUPING_MODES)
def test_every_declared_mode_validates(mode: str) -> None:
    assert validate_grouping(mode) == mode


def test_unknown_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="grouping must be one of"):
        validate_grouping("by_vibes")


def test_prefix_splits_a_mixed_batch_into_one_sample_per_specimen() -> None:
    plan = plan_grouping(MIXED_BATCH, "prefix")

    assert {k: len(v) for k, v in plan.items()} == {
        "almond_control": 2,
        "almond_drought": 1,
        "lantana_camara": 1,
    }


def test_single_keeps_the_whole_batch_as_one_sample() -> None:
    """The pre-existing behaviour, still available for a genuine z-stack."""
    plan = plan_grouping(MIXED_BATCH, "single")

    assert list(plan) == [""]  # "" = straight into the target container
    assert len(plan[""]) == 4


def test_per_image_gives_every_file_its_own_sample() -> None:
    plan = plan_grouping(MIXED_BATCH, "per_image")

    assert len(plan) == 4
    assert all(len(v) == 1 for v in plan.values())
    assert "almond_control__slice_01" in plan


def test_prefix_keeps_an_unprefixed_z_stack_as_one_sample() -> None:
    """A conventional stack has no `__` at all; splitting it per file would turn
    30 slices of one specimen into 30 single-image samples."""
    stack = [f"img_{i:05d}.tif" for i in range(1, 31)]

    plan = plan_grouping(stack, "prefix")

    assert list(plan) == [""]  # "" = the target container itself
    assert len(plan[""]) == 30


def test_mixed_batch_promotes_unprefixed_files_to_their_own_sample() -> None:
    """A batch with SOME "__"-declared files and some without can't fold the
    undeclared ones into one "" catch-all alongside the named samples — that
    would put an array and sample containers as siblings, a layout Browse
    cannot describe (browse_helpers._describe_sample: a container is either
    the sample, or a parent OF samples, never both at once). So each
    undeclared file gets its own sample instead, exactly as "per_image" would.
    """
    plan = plan_grouping(["plain.tif", "grouped__01.tif", "grouped__02.tif"], "prefix")

    assert {k: len(v) for k, v in plan.items()} == {"plain": 1, "grouped": 2}
    assert "" not in plan


def test_mixed_batch_promotes_every_undeclared_file_independently() -> None:
    """Two undeclared files in a mixed batch become two SEPARATE samples, not
    one shared "" sample between them."""
    plan = plan_grouping(["a.tif", "b.tif", "grouped__01.tif"], "prefix")

    assert {k: len(v) for k, v in plan.items()} == {"a": 1, "b": 1, "grouped": 1}


def test_all_unprefixed_batch_is_unaffected_by_the_mixed_batch_rule() -> None:
    """The mixed-batch promotion only fires when the batch is ACTUALLY mixed —
    an all-unprefixed batch (no file declares "__" at all) has nothing to
    disambiguate from, so it keeps the original one-sample z-stack behaviour."""
    plan = plan_grouping(["a.tif", "b.tif", "c.tif"], "prefix")

    assert list(plan) == [""]
    assert len(plan[""]) == 3


def test_all_declared_batch_is_unaffected_by_the_mixed_batch_rule() -> None:
    """Symmetric case: every file declares a sample, so there is no undeclared
    file to promote."""
    plan = plan_grouping(["a__01.tif", "b__01.tif"], "prefix")

    assert {k: len(v) for k, v in plan.items()} == {"a": 1, "b": 1}


@pytest.mark.parametrize(
    ("stem", "expected"),
    [
        ("almond_control__slice_01", "almond_control"),
        ("a__b__c", "a"),  # first separator wins
        ("__leading", None),  # empty head declares no sample
        ("trailing__", "trailing"),
        ("single_underscore_only", None),  # one underscore is not the separator
        ("img_00002", None),
    ],
)
def test_prefix_edge_cases(stem: str, expected: str | None) -> None:
    assert sample_name_for(stem, "prefix") == expected


def test_single_mode_returns_no_subcontainer() -> None:
    assert sample_name_for("anything__01", "single") is None


def test_grouping_preserves_every_file() -> None:
    """No mode may drop or duplicate an upload."""
    for mode in GROUPING_MODES:
        plan = plan_grouping(MIXED_BATCH, mode)
        flat = [name for names in plan.values() for name in names]
        assert sorted(flat) == sorted(MIXED_BATCH), mode
