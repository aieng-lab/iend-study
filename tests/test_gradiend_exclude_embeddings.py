"""``gradiend_exclude_embeddings`` — main-experiment default since 2026-08-21
(``configs/defaults.yaml``; previously opt-in only via
configs/suites/gradiend_no_embed.yaml).

GRADIEND has no activation SignalScope by default (unlike ACTIEND, which always
restricts to SignalScope.layers() — see CLAUDE.md's "GRADIEND and ACTIEND scope"
note): its package default scope_mode="default" prunes gradients over the whole
HF backbone/head split, which includes the word/position embedding matrices
(wte/wpe, or embed_in for gpt_neox). This flag now sets
``signal_scope=SignalScope.layers()`` — the same scope object ACTIEND already
uses — so GRADIEND's effective scope matches ACTIEND's exactly: residual
transformer blocks only, excluding both the embedding matrices and the final
layer norm (see study/training_profiles.py's comment above this constant for
why ln_f is excluded too, and CLAUDE.md's "GRADIEND main-experiment scope"
note for the package fix that made ``.layers()`` work for a gradient signal at
all). ``build_training_arguments``'s own default (when the flag key is
altogether absent, e.g. called directly without going through study config
loading) is still ``False`` — only ``configs/defaults.yaml`` flips the
study-level default; direct callers must opt in explicitly.

Per-architecture resolution (does this arch's topology support ``.layers()``
at all?) happens later, at actual model-construction time in the ``gradiend``
package (``gradient_params_from_selector``/``ModelTopology``) — not here.
``build_training_arguments`` itself is architecture-agnostic now; there is no
study-level allow-list to test against any more (previously
``_GRADIEND_NO_EMBED_SCOPE_PARAMS``, removed 2026-08-21 along with the
per-arch hand-maintained pattern dict it fed).
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gradiend import SignalScope

from study.config import load_study_config
from study.training_profiles import shared_training_kwargs, build_training_arguments


def test_suite_overrides_the_default_back_off_and_sets_two_pole_only_ablations():
    # gradiend_no_embed's question inverted 2026-08-21 when exclude-embeddings
    # became the main-experiment default: this suite now explicitly opts back
    # INTO embeddings (False) to measure the other arm of the comparison.
    cfg = load_study_config(model="pythia-70m-deduped", task="gender_en", suite="gradiend_no_embed")
    assert cfg.training.get("gradiend_exclude_embeddings") is False
    assert cfg.raw["ablations"]["pair"] is True
    assert cfg.raw["ablations"]["one_pole"] is False


def test_other_suites_inherit_the_main_experiment_default():
    # gradiend_exclude_embeddings became the main-experiment default in
    # configs/defaults.yaml on 2026-08-21 (was opt-in-only via
    # configs/suites/gradiend_no_embed.yaml before that) -- core/full/full_plus
    # now inherit True rather than leaving the flag unset.
    for suite in ("core", "full", "full_plus"):
        cfg = load_study_config(model="gpt2-small", task="gender_en", suite=suite)
        assert cfg.training.get("gradiend_exclude_embeddings") is True


def test_gradiend_gets_layers_scope_for_gpt2():
    args = build_training_arguments(
        "gradiend",
        experiment_dir="scratch",
        shared={"gradiend_exclude_embeddings": True, "learning_rate": 1e-5},
        metadata={"model_key": "gpt2-small", "arch": "gpt2", "task": "gender_en"},
    )
    assert args.signal_scope == SignalScope.layers()
    assert args.metadata["gradiend_exclude_embeddings"] is True
    assert args.metadata["gradiend_scope"] == "layers"


def test_gradiend_gets_layers_scope_regardless_of_arch():
    # Architecture-agnostic at this layer -- per-arch topology support is
    # only checked later, at model-construction time in the gradiend package.
    args = build_training_arguments(
        "gradiend",
        experiment_dir="scratch",
        shared={"gradiend_exclude_embeddings": True, "learning_rate": 1e-5},
        metadata={"model_key": "pythia-70m-deduped", "arch": "gpt_neox", "task": "gender_en"},
    )
    assert args.signal_scope == SignalScope.layers()


def test_flag_off_leaves_gradiend_scope_unset():
    args = build_training_arguments(
        "gradiend",
        experiment_dir="scratch",
        shared={"learning_rate": 1e-5},
        metadata={"model_key": "gpt2-small", "arch": "gpt2", "task": "gender_en"},
    )
    assert args.signal_scope is None
    assert args.metadata["gradiend_exclude_embeddings"] is False


def test_flag_does_not_leak_into_actiend_training_arguments():
    # ACTIEND always forces SignalScope.layers() regardless of this GRADIEND-only
    # flag; the flag must not survive into TrainingArguments(**kwargs) for actiend
    # (it isn't a real field there) or override actiend's own scope.
    args = build_training_arguments(
        "actiend",
        experiment_dir="scratch",
        shared={
            "gradiend_exclude_embeddings": True,
            "learning_rate": 1e-5,
            "activation_site": "prediction",
            "target_activation_site": "prediction",
        },
        metadata={"model_key": "gpt2-small", "arch": "gpt2", "task": "gender_en"},
    )
    assert args.signal_scope.activation_selector == ("layers", None)


def test_shared_training_kwargs_forwards_the_flag():
    cfg = load_study_config(model="pythia-70m-deduped", task="gender_en", suite="gradiend_no_embed")
    shared = shared_training_kwargs(cfg, backend="gradiend")
    assert shared.get("gradiend_exclude_embeddings") is False
