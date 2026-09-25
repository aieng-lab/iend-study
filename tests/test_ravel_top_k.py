"""RAVEL class-count selection for larger-K tasks.

One-pole trains K models where pairwise training needs K(K-1)/2 plus an
aggregation rule. Those counts are equal at K=3 -- the largest class count in
the current suite -- so demonstrating the scaling claim needs a K>=4 task, which
means raising RAVEL's ``top_k`` past its 3-entry preset.

The preset path used to slice (``preset[:top_k]``), which for top_k > 3 returned
the whole 3-entry preset: a task config declaring 8 classes would silently train
on 3, with no error anywhere.
"""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

from study.data import mib


class TestPresetPath:
    def test_exact_preset_length_returns_the_preset(self):
        assert mib._ravel_top_values("Country", top_k=3) == (
            "United States",
            "China",
            "Russia",
        )

    def test_smaller_than_preset_slices(self):
        assert mib._ravel_top_values("Country", top_k=2) == ("United States", "China")

    def test_larger_than_preset_does_not_silently_return_the_preset(self):
        """The regression: top_k=8 used to yield 3 classes with no signal."""
        rows = [
            {"ID": f"{name}-{i}", "City": f"City-{name}-{i}", "URL": f"https://x/{name}/{i}", "Country": name}
            for name, count in zip("ABCDEFGH", range(8, 0, -1))
            for i in range(count)
        ]
        with patch.object(
            mib,
            "_load_ravel_sources",
            return_value=(pd.DataFrame(rows), pd.DataFrame()),
        ):
            values = mib._ravel_top_values("Country", top_k=8)
        assert len(values) == 8, f"expected 8 classes, got {len(values)}: {values}"
        assert values[0] == "A"

    def test_unsatisfiable_top_k_raises_rather_than_undercounting(self):
        rows = [
            {"ID": str(i), "City": f"City-{i}", "URL": f"https://x/{i}", "Country": n}
            for i, n in enumerate(("A", "B", "C", "D"))
        ]
        with patch.object(
            mib,
            "_load_ravel_sources",
            return_value=(pd.DataFrame(rows), pd.DataFrame()),
        ):
            with pytest.raises(ValueError, match="only 4 distinct values"):
                mib._ravel_top_values("Country", top_k=8)


class TestPublishableDerivative:
    def test_balances_entity_splits_and_removes_unidentifiable_surfaces(self):
        values = ("United States", "China", "Russia")
        rows = []
        for class_index, value in enumerate(values):
            for i in range(21):
                rows.append(
                    {
                        "ID": f"{class_index}-{i}",
                        "City": f"City {class_index} {i}",
                        "URL": f"https://example.test/{class_index}/{i}",
                        "Country": value,
                        "Continent": "X",
                        "Language": "Y",
                        "source_entity_split": "train",
                    }
                )
        # The model-facing prompt cannot distinguish these two records.
        rows.extend(
            [
                {"ID": "99-1", "City": "Amsterdam", "URL": "https://x/us", "Country": "United States", "Continent": "X", "Language": "Y", "source_entity_split": "train"},
                {"ID": "99-2", "City": "Amsterdam", "URL": "https://x/cn", "Country": "China", "Continent": "X", "Language": "Y", "source_entity_split": "test"},
            ]
        )
        prompts = pd.DataFrame(
            [
                {
                    "Template": f"{split} template {i}: %s is in",
                    "Attribute": "Country",
                    "Source": "RAVEL",
                    "Entity": "",
                    "source_prompt_split": split,
                }
                for split in ("train", "validation", "test")
                for i in range(2)
            ]
            + [
                {
                    "Template": "People in %s speak",
                    "Attribute": "Language",
                    "Source": "RAVEL",
                    "Entity": "",
                    "source_prompt_split": "train",
                }
            ]
        )
        with patch.object(
            mib,
            "_load_ravel_sources",
            return_value=(pd.DataFrame(rows), prompts),
        ):
            frame, _neutral = mib.build_ravel_attribute_df(
                "Country",
                top_values=values,
                seed=7,
            )

        assert "Amsterdam" not in set(frame["entity"])
        assert frame["masked"].is_unique
        assert frame.groupby("entity_id")["split"].nunique().max() == 1
        assert set(frame["source"]) == {"hij/ravel"}
        counts = frame.groupby(["label_class", "split"]).size().unstack()
        assert counts.nunique().eq(1).all()
        assert counts.iloc[0].to_dict() == {"test": 2, "train": 16, "validation": 2}
        assert frame.attrs["ravel_audit"]["ambiguous_surface_count"] == 1


class TestScalingArithmetic:
    """Why K>=4 is required to demonstrate the claim at all."""

    @staticmethod
    def _counts(k):
        return k * (k - 1) // 2, k

    def test_k3_gives_no_model_count_advantage(self):
        assert self._counts(3) == (3, 3)

    def test_advantage_appears_at_k4_and_grows(self):
        assert self._counts(4) == (6, 4)
        assert self._counts(8) == (28, 8)

    def test_scaling_and_one_pole_overlays_are_hidden_from_main_all(self):
        from study.config import list_tasks

        main_tasks = set(list_tasks())
        hidden_tasks = set(list_tasks(include_hidden=True))
        derived = {
            "ravel_country_k5",
            "ravel_country_k10",
            "race_one_pole",
            "religion_one_pole",
        }
        assert derived.isdisjoint(main_tasks)
        assert derived <= hidden_tasks


class TestPairFlag:
    """The pair arm is a CLI choice, not a post-hoc file edit.

    The generator previously always wrote pair: false, so a parity arm required
    hand-editing the generated config -- a silent manual step between two
    submitted jobs.
    """

    @staticmethod
    def _render(pair):
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        src = (root / "scripts/make_ravel_task.py").read_text(encoding="utf-8")
        ns = {}
        exec(src[src.index("_NL = chr(10)") : src.index("def main()")], ns)
        return ns["TEMPLATE"].format(
            task="ravel_country_k5", base="ravel_country", attr="Country",
            k=5, pairs=10, classes="a, b",
            pair_block=ns["PAIR_ON"] if pair else ns["PAIR_OFF"],
        )

    def test_pair_arm_enables_pairwise(self):
        assert "  pair: true" in self._render(True)
        assert "pair: false" not in self._render(True)

    def test_default_leaves_pairwise_off(self):
        rendered = self._render(False)
        assert "  pair: false" in rendered
        assert "pair: true" not in rendered

    def test_one_pole_is_always_on(self):
        for pair in (True, False):
            assert "one_pole: true" in self._render(pair)

    def test_generated_scaling_task_is_hidden(self):
        for pair in (True, False):
            assert "study_hidden: true" in self._render(pair)

    def test_flag_exists_on_the_parser(self):
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        src = (root / "scripts/make_ravel_task.py").read_text(encoding="utf-8")
        assert '"--pair"' in src
        assert "PAIR_ON if args.pair else PAIR_OFF" in src


class TestBuilderHonoursItsConfig:
    """build_ravel_task ignored data.attribute and data.top_k entirely.

    A generated larger-K task resolves to the ravel_country builder, so paths
    derived from task_id alone served the base K=3 CSV -- which exists, so no
    regeneration was triggered -- and the run died with "No rows for class
    'brazil'". Nulling local_path did not help: the fallback path is the same
    file by convention.
    """

    def test_larger_k_gets_its_own_artifact_path(self):
        from study.data.mib import ravel_top_k_suffix

        assert ravel_top_k_suffix(3) == ""
        assert ravel_top_k_suffix(5) == "_k5"
        assert ravel_top_k_suffix(10) == "_k10"

    def test_base_build_path_is_unchanged(self):
        """K=3 must keep writing ravel_country.csv, or every task rebuilds."""
        from study.data.mib import DEFAULT_TOP_K, ravel_top_k_suffix

        assert ravel_top_k_suffix(DEFAULT_TOP_K) == ""

    def test_builder_reads_attribute_and_top_k_from_config(self):
        from pathlib import Path

        import study.data.mib as mib

        src = Path(mib.__file__).read_text(encoding="utf-8")
        block = src[src.index("def build_ravel_task") :][:2200]
        assert 'data_cfg.get("attribute")' in block
        assert 'data_cfg.get("top_k")' in block
        assert "top_k=top_k" in block, "generation must receive the task's top_k"

    def test_generation_cannot_clobber_the_base_csv(self):
        from pathlib import Path

        import study.data.mib as mib

        src = Path(mib.__file__).read_text(encoding="utf-8")
        block = src[src.index("def generate_ravel_attribute") :][:1200]
        assert "ravel_top_k_suffix(top_k)" in block
        assert 'f"{task_id}{suffix}.csv"' in block
