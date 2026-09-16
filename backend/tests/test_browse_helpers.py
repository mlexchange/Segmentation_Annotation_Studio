"""Tests for browse_helpers.py using lightweight fake Tiled nodes — duck-typed
to the subset of the Tiled client API this module actually uses
(__iter__/__getitem__/.metadata/.structure_family/.search()/.distinct()), so
no real Tiled server is needed. search() is driven by REAL tiled.queries.Eq/
Contains objects (what Key(...) == val / Contains(...) actually produce),
not a further mock of the query layer itself.
"""
from __future__ import annotations

from collections import Counter

import pytest

import browse_helpers as bh

tiled_queries = pytest.importorskip("tiled.queries")


class FakeNode:
    def __init__(self, children=None, metadata=None, is_container=True, search_raises=False):
        self._children = children or {}
        self.metadata = metadata or {}
        self.structure_family = "container" if is_container else "array"
        self._search_raises = search_raises

    def __iter__(self):
        return iter(self._children)

    def __getitem__(self, key):
        return self._children[key]

    def __len__(self):
        return len(self._children)

    def search(self, query):
        if self._search_raises:
            raise RuntimeError("search exploded")
        matched = {}
        for k, child in self._children.items():
            meta = child.metadata or {}
            if isinstance(query, tiled_queries.Contains):
                val = meta.get(query.key)
                if isinstance(val, (list, tuple)) and query.value in val:
                    matched[k] = child
            else:  # Eq
                if meta.get(query.key) == query.value:
                    matched[k] = child
        return FakeNode(children=matched, metadata=self.metadata)

    def distinct(self, key, counts=True):
        counter: Counter = Counter()
        for child in self._children.values():
            val = (child.metadata or {}).get(key)
            if val is None:
                continue
            counter[val] += 1
        return {"metadata": {key: [{"value": v, "count": n} for v, n in counter.items()]}}


# ---------------------------------------------------------------------------
# Field discovery
# ---------------------------------------------------------------------------

class TestDisplayName:
    def test_strips_thinfilm_prefix(self):
        assert bh._display_name("thinfilm_AnnealingTemp") == "Temp"

    def test_thinfilm_prefix_without_alias_returns_stripped(self):
        assert bh._display_name("thinfilm_Custom") == "Custom"

    def test_studio_key_alias(self):
        assert bh._display_name("studio_annotated") == "Annotated"

    def test_unknown_key_passthrough(self):
        assert bh._display_name("foo") == "foo"


class TestIsRichEnough:
    def test_thinfilm_key_alone_is_enough(self):
        assert bh._is_rich_enough(["thinfilm_x"]) is True

    def test_important_keys_with_min_count(self):
        keys = ["PI", "sample_name", "a", "b"]
        assert bh._is_rich_enough(keys) is True

    def test_important_key_without_min_count_not_enough(self):
        assert bh._is_rich_enough(["PI"]) is False

    def test_plain_key_richness_threshold(self):
        assert bh._is_rich_enough([f"k{i}" for i in range(10)]) is True
        assert bh._is_rich_enough([f"k{i}" for i in range(9)]) is False


class TestBuildFieldMapping:
    def test_finds_first_rich_enough_sample(self):
        sparse = FakeNode(metadata={"a": 1})
        rich = FakeNode(metadata={f"k{i}": i for i in range(10)})
        node = FakeNode(children={"s0": sparse, "s1": rich})
        mapping = bh.build_field_mapping(node)
        assert "k0" in mapping.raw_to_display

    def test_falls_back_to_last_scanned_if_none_rich_enough(self):
        node = FakeNode(children={"s0": FakeNode(metadata={"a": 1})})
        mapping = bh.build_field_mapping(node)
        assert "a" in mapping.raw_to_display

    def test_skips_samples_that_error_on_open(self):
        class ExplodingNode:
            @property
            def metadata(self):
                raise RuntimeError("boom")

        node = FakeNode(children={"bad": ExplodingNode(), "good": FakeNode(metadata={f"k{i}": i for i in range(10)})})
        mapping = bh.build_field_mapping(node)
        assert "k0" in mapping.raw_to_display

    def test_always_injects_studio_and_ingest_keys(self):
        node = FakeNode(children={"s0": FakeNode(metadata={})})
        mapping = bh.build_field_mapping(node)
        assert "Annotated" in mapping.all_display_keys
        assert "studio_annotated" in mapping.raw_to_display

    def test_empty_container_does_not_raise(self):
        mapping = bh.build_field_mapping(FakeNode(children={}))
        assert "studio_annotated" in mapping.raw_to_display


# ---------------------------------------------------------------------------
# Distinct values
# ---------------------------------------------------------------------------

class TestScopedMetadataRows:
    def test_reads_each_childs_metadata(self):
        node = FakeNode(children={"a": FakeNode(metadata={"x": 1}), "b": FakeNode(metadata={"x": 2})})
        rows = bh.scoped_metadata_rows(node)
        assert rows == [{"x": 1}, {"x": 2}]

    def test_respects_limit(self):
        node = FakeNode(children={str(i): FakeNode(metadata={"x": i}) for i in range(10)})
        rows = bh.scoped_metadata_rows(node, limit=3)
        assert len(rows) == 3

    def test_skips_children_that_fail_to_open(self):
        class ExplodingNode:
            @property
            def metadata(self):
                raise RuntimeError("boom")

        node = FakeNode(children={"bad": ExplodingNode(), "good": FakeNode(metadata={"x": 1})})
        assert bh.scoped_metadata_rows(node) == [{"x": 1}]


class TestDistinctFromRows:
    def test_tallies_scalar_values(self):
        rows = [{"k": "a"}, {"k": "a"}, {"k": "b"}]
        result = bh.distinct_from_rows(rows, "k")
        by_value = {e["value"]: e["count"] for e in result}
        assert by_value == {"a": 2, "b": 1}

    def test_explodes_list_valued_metadata(self):
        rows = [{"k": ["x", "y"]}, {"k": ["x"]}]
        result = bh.distinct_from_rows(rows, "k")
        by_value = {e["value"]: e["count"] for e in result}
        assert by_value == {"x": 2, "y": 1}

    def test_missing_key_ignored(self):
        assert bh.distinct_from_rows([{"other": 1}], "k") == []


class TestTiledDistinctValues:
    def _node(self):
        return FakeNode(children={
            "a": FakeNode(metadata={"studio_annotated": "yes"}),
            "b": FakeNode(metadata={"studio_annotated": "yes"}),
            "c": FakeNode(metadata={"studio_annotated": "no"}),
        })

    def test_global_distinct_uses_node_distinct(self):
        result = bh.tiled_distinct_values(self._node(), "studio_annotated")
        by_value = {e["value"]: e["count"] for e in result["values"]}
        assert by_value == {"yes": 2, "no": 1}
        assert result["total"] == 2

    def test_scoped_distinct_uses_iteration(self):
        result = bh.tiled_distinct_values(self._node(), "studio_annotated", scoped=True)
        by_value = {e["value"]: e["count"] for e in result["values"]}
        assert by_value == {"yes": 2, "no": 1}

    def test_invalid_values_filtered_out(self):
        node = FakeNode(children={
            "a": FakeNode(metadata={"k": "real"}),
            "b": FakeNode(metadata={"k": "NaN"}),
        })
        result = bh.tiled_distinct_values(node, "k")
        assert result["values"] == [{"value": "real", "count": 1, "sample_paths": []}]

    def test_field_mapping_translates_display_key_to_raw(self):
        mapping = bh.FieldMapping(
            display_to_raw={}, raw_to_display={"studio_annotated": "Annotated"}, all_display_keys=[],
        )
        result = bh.tiled_distinct_values(self._node(), "studio_annotated", field_mapping=mapping)
        assert result["field"] == "Annotated"

    def test_filter_narrows_node_before_distinct(self):
        mapping = bh.FieldMapping(
            display_to_raw={"Annotated": "studio_annotated"}, raw_to_display={}, all_display_keys=[],
        )
        result = bh.tiled_distinct_values(
            self._node(), "studio_annotated", filters={"Annotated": "yes"}, field_mapping=mapping,
        )
        assert result["total"] == 1  # only "yes" remains after filtering to studio_annotated == "yes"

    def test_distinct_call_failure_returns_empty(self):
        node = FakeNode(children={})
        node.distinct = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
        result = bh.tiled_distinct_values(node, "k")
        assert result == {"values": [], "total": 0, "field": "k"}


# ---------------------------------------------------------------------------
# Item search
# ---------------------------------------------------------------------------

class TestTiledSearchItemsDispatch:
    def test_no_array_only_filters_uses_container_only_path(self):
        node = FakeNode(children={"a": FakeNode(metadata={"PI": "smith"}, is_container=False)})
        result = bh.tiled_search_items(node, filters={"PI": "smith"})
        assert result["total"] == 1
        assert result["items"][0]["sample"] == "a"

    def test_only_array_only_filters_uses_array_only_path(self):
        # bar's metadata is a real int here, matching how Tiled actually stores
        # this field — _apply_filters coerces the filter string "1" to int 1
        # via _typed_query_value, so a string-valued fixture would (correctly)
        # never match and silently fail the test for the wrong reason.
        node = FakeNode(children={"a": FakeNode(metadata={"bar": 1}, is_container=False)})
        result = bh.tiled_search_items(node, filters={"bar": "1"})
        assert result["total"] == 1

    def test_mixed_filters_uses_mixed_path(self):
        array_child = FakeNode(metadata={"bar": 1}, is_container=False)
        sample = FakeNode(children={"arr": array_child}, metadata={"PI": "smith"})
        node = FakeNode(children={"s0": sample})
        result = bh.tiled_search_items(node, filters={"PI": "smith", "bar": "1"})
        assert result["total"] == 1
        assert result["items"][0]["sample"] == "s0"

    def test_no_filters_returns_everything(self):
        node = FakeNode(children={"a": FakeNode(metadata={}), "b": FakeNode(metadata={})})
        result = bh.tiled_search_items(node)
        assert result["total"] == 2


class TestSearchContainerOnly:
    def test_reports_n_slices_for_flat_array_stack(self):
        stack = FakeNode(children={
            str(i): FakeNode(metadata={}, is_container=False) for i in range(5)
        }, metadata={})
        node = FakeNode(children={"sample1": stack})
        result = bh._search_container_only(node, [], "", 500)
        assert result["items"][0]["n_slices"] == 5

    def test_borrows_first_childs_metadata_when_container_has_none(self):
        stack = FakeNode(children={"0": FakeNode(metadata={"x": 1}, is_container=False)}, metadata={})
        node = FakeNode(children={"sample1": stack})
        result = bh._search_container_only(node, [], "", 500)
        assert result["items"][0]["metadata"] == {"x": 1}

    def test_leaf_array_reports_n_slices_one(self):
        node = FakeNode(children={"a": FakeNode(metadata={}, is_container=False)})
        result = bh._search_container_only(node, [], "", 500)
        assert result["items"][0]["n_slices"] == 1

    def test_path_prefix_applied(self):
        node = FakeNode(children={"a": FakeNode(metadata={}, is_container=False)})
        result = bh._search_container_only(node, [], "parent", 500)
        assert result["items"][0]["path"] == "parent/a"

    def test_children_that_fail_to_open_are_skipped(self):
        class Exploding:
            metadata = property(lambda self: (_ for _ in ()).throw(RuntimeError()))

        node = FakeNode(children={"bad": Exploding(), "good": FakeNode(metadata={}, is_container=False)})
        result = bh._search_container_only(node, [], "", 500)
        assert result["total"] == 1


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

class TestRawFilters:
    def test_translates_display_to_raw_and_stringifies(self):
        out = bh._raw_filters({"Temp": 5}, {"Temp": "thinfilm_AnnealingTemp"})
        assert out == [("thinfilm_AnnealingTemp", "5")]

    def test_none_values_dropped(self):
        assert bh._raw_filters({"Temp": None}, {}) == []

    def test_unmapped_display_key_passthrough(self):
        assert bh._raw_filters({"raw_key": "v"}, {}) == [("raw_key", "v")]


class TestMatchesContainerFilters:
    def test_scalar_case_insensitive_match(self):
        assert bh._matches_container_filters({"PI": "Smith"}, [("PI", "smith")]) is True

    def test_scalar_mismatch(self):
        assert bh._matches_container_filters({"PI": "Jones"}, [("PI", "smith")]) is False

    def test_missing_key_fails(self):
        assert bh._matches_container_filters({}, [("PI", "smith")]) is False

    def test_list_valued_membership_match(self):
        assert bh._matches_container_filters({"keywords": ["Alpha", "Beta"]}, [("keywords", "alpha")]) is True

    def test_list_valued_membership_mismatch(self):
        assert bh._matches_container_filters({"keywords": ["Beta"]}, [("keywords", "alpha")]) is False


class TestTypedQueryValue:
    def test_int_roundtrip(self):
        assert bh._typed_query_value("5") == 5

    def test_float_roundtrip(self):
        assert bh._typed_query_value("5.5") == 5.5

    def test_non_numeric_passthrough(self):
        assert bh._typed_query_value("abc") == "abc"

    def test_leading_zero_not_coerced_to_int(self):
        # "007" != str(int("007")) == "7", so it must stay a string.
        assert bh._typed_query_value("007") == "007"


class TestIsValidValue:
    @pytest.mark.parametrize("value", [None, "", "None", "NaN", "nan", "   "])
    def test_invalid_values(self, value):
        assert bh._is_valid_value(value) is False

    def test_valid_value(self):
        assert bh._is_valid_value("real") is True


class TestJoin:
    def test_with_prefix(self):
        assert bh._join("a", "b") == "a/b"

    def test_without_prefix(self):
        assert bh._join("", "b") == "b"


class TestApplyFilters:
    def test_eq_filter_narrows_node(self):
        node = FakeNode(children={"a": FakeNode(metadata={"k": "v"}), "b": FakeNode(metadata={"k": "other"})})
        result = bh._apply_filters(node, [("k", "v")])
        assert list(result) == ["a"]

    def test_contains_filter_for_list_valued_key(self):
        node = FakeNode(children={
            "a": FakeNode(metadata={"keywords": ["x", "y"]}),
            "b": FakeNode(metadata={"keywords": ["z"]}),
        })
        result = bh._apply_filters(node, [("keywords", "x")])
        assert list(result) == ["a"]

    def test_search_failure_is_swallowed_and_node_unchanged(self):
        node = FakeNode(children={"a": FakeNode(metadata={})}, search_raises=True)
        result = bh._apply_filters(node, [("k", "v")])
        assert result is node
