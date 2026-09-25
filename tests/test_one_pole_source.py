"""One-pole training always uses source='both' (2026-08-31).

source='alternative' collapses a one-pole model's rivals into an incoherent -1
pole; 'both' gives a coherent per-prompt factual-vs-alternative target. The
race_one_pole/religion_one_pole check-tasks proved 'both' >> 'alternative' for
one-pole on identical data (race ACTIEND one-pole AUC_o 0.59 -> 0.98). This
folds that into the default so every one-pole task gets it, while pairs keep
their task source.
"""

from __future__ import annotations

from study.stages.train import one_pole_train_source


class TestOnePoleSource:
    def test_one_pole_forces_both_over_alternative(self):
        assert one_pole_train_source({"source": "alternative"}, is_one_pole=True) == "both"

    def test_one_pole_forces_both_even_when_source_absent(self):
        assert one_pole_train_source({}, is_one_pole=True) == "both"

    def test_one_pole_task_already_both_is_unchanged(self):
        assert one_pole_train_source({"source": "both"}, is_one_pole=True) == "both"

    def test_pair_keeps_task_source(self):
        assert one_pole_train_source({"source": "alternative"}, is_one_pole=False) == "alternative"
        assert one_pole_train_source({"source": "both"}, is_one_pole=False) == "both"

    def test_pair_with_no_source_stays_none(self):
        """A pair must not be silently forced onto 'both'; that's one-pole-only."""
        assert one_pole_train_source({}, is_one_pole=False) is None


class TestSourceInArtifactHash:
    """The train-artifact hash must reflect the ACTUAL one-pole source (both).

    The mark-done path passes shared_kw (override applied); the skip-check path
    passes raw shared. If the hash read raw shared.get('source') they would
    disagree for one-pole, and skip-existing would never see the source change.
    """

    def _hash(self, *, source, one_pole):
        from study.stages.train import _train_artifact_hash
        from study.config import load_study_config

        cfg = load_study_config(model="gpt2-small", task="race", suite="full_plus")
        kw = dict(
            feature_class="asian" if one_pole else None,
            counterfactual_classes="all" if one_pole else None,
        )
        return _train_artifact_hash(
            cfg,
            backend="gradiend",
            shared={"source": source, "learning_rate": 1e-4},
            split_mode="none",
            target_classes=["asian"] if one_pole else ["asian", "black"],
            **kw,
        )

    def test_one_pole_hash_ignores_raw_source_and_uses_both(self):
        # Same one-pole cell, raw shared says 'alternative' vs 'both' -> the hash
        # must be identical, because both resolve to source='both'.
        assert self._hash(source="alternative", one_pole=True) == self._hash(
            source="both", one_pole=True
        )

    def test_pair_hash_still_tracks_raw_source(self):
        # Pairs keep their task source, so the hash must distinguish them.
        assert self._hash(source="alternative", one_pole=False) != self._hash(
            source="both", one_pole=False
        )
