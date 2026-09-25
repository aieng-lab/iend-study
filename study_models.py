"""Study model presets: HF load id ↔ SAE Lens release ↔ residual hook sites.

Used by the legacy gender script and as a code fallback. Prefer YAML under
``configs/models/`` via ``run_study.py`` for multi-task runs
(``runs/{model_key}/{task_id}/``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class StudyModel:
    """One backbone + pretrained SAE suite used by the gender-EN study."""

    key: str
    # SAE Lens / TransformerLens registry name (results.json ``model`` field).
    sae_model: str
    # Hugging Face id for GRADIEND / transformers load.
    hf_model: str
    # SAE Lens ``release`` for resid-post (or closest resid) SAEs.
    # Empty string = no SAELens suite wired yet (use ``--skip-sae``).
    sae_release: str
    n_layers: int
    # HF module path for residual-stream activations at layer L.
    hf_resid_template: str
    # SAE Lens ``sae_id`` template for the same site.
    sae_id_template: str
    # Architecture family (localization write-out mapping, notes).
    arch: str
    output_dirname: str
    # If set, SAE / multi-layer sweeps only use these layers (else ``range(n_layers)``).
    sae_layers: Optional[Tuple[int, ...]] = None

    def sites(self, layer: int) -> Tuple[str, str]:
        """Return ``(hf_module_path, sae_id)`` for resid-post (or equivalent) at ``layer``."""
        L = int(layer)
        return (
            self.hf_resid_template.format(layer=L),
            self.sae_id_template.format(layer=L),
        )

    @property
    def layers(self) -> Tuple[int, ...]:
        if self.sae_layers is not None:
            return tuple(int(x) for x in self.sae_layers)
        return tuple(range(int(self.n_layers)))

    @property
    def has_sae(self) -> bool:
        return bool(str(self.sae_release or "").strip())

    @property
    def output_dir(self) -> Path:
        return Path("runs") / self.output_dirname

    def smoke_output_dir(self) -> Path:
        return Path("runs") / f"{self.output_dirname}_smoke"


def _layers(*xs: int) -> Tuple[int, ...]:
    return tuple(int(x) for x in xs)


# ----- registry -----
# Keys are CLI ``--model`` values.

STUDY_MODELS: Dict[str, StudyModel] = {
    "gpt2-small": StudyModel(
        key="gpt2-small",
        sae_model="gpt2-small",
        hf_model="gpt2",
        sae_release="gpt2-small-resid-post-v5-32k",
        n_layers=12,
        hf_resid_template="transformer.h.{layer}",
        sae_id_template="blocks.{layer}.hook_resid_post",
        arch="gpt2",
        output_dirname="gender_en_gpt2_small",
    ),
    "pythia-70m-deduped": StudyModel(
        key="pythia-70m-deduped",
        sae_model="pythia-70m-deduped",
        hf_model="EleutherAI/pythia-70m-deduped",
        sae_release="pythia-70m-deduped-res-sm",
        n_layers=6,
        hf_resid_template="gpt_neox.layers.{layer}",
        sae_id_template="blocks.{layer}.hook_resid_post",
        arch="gpt_neox",
        output_dirname="gender_en_pythia_70m_deduped",
    ),
    # Llama
    "llama-3.1-8b": StudyModel(
        key="llama-3.1-8b",
        sae_model="meta-llama/Llama-3.1-8B",
        hf_model="meta-llama/Llama-3.1-8B",
        # Llama Scope residual (32×); ids ``l{layer}r_32x``.
        sae_release="llama_scope_lxr_32x",
        n_layers=32,
        hf_resid_template="model.layers.{layer}",
        sae_id_template="l{layer}r_32x",
        arch="llama",
        output_dirname="gender_en_llama_3_1_8b",
    ),
    "llama-3.3-70b-instruct": StudyModel(
        key="llama-3.3-70b-instruct",
        sae_model="meta-llama/Llama-3.3-70B-Instruct",
        hf_model="meta-llama/Llama-3.3-70B-Instruct",
        # Goodfire SAE via SAELens (single site: layer 50).
        sae_release="goodfire-llama-3.3-70b-instruct",
        n_layers=80,
        hf_resid_template="model.layers.{layer}",
        sae_id_template="layer_{layer}",
        arch="llama",
        output_dirname="gender_en_llama_3_3_70b_instruct",
        sae_layers=_layers(50),
    ),
    # Qwen 3.5 (Qwen-Scope TopK; SAELens >= 6.43)
    "qwen3.5-2b-base": StudyModel(
        key="qwen3.5-2b-base",
        sae_model="Qwen/Qwen3.5-2B-Base",
        hf_model="Qwen/Qwen3.5-2B-Base",
        sae_release="qwen-scope-3.5-2b-base-w32k-l50",
        n_layers=24,
        hf_resid_template="model.layers.{layer}",
        sae_id_template="layer{layer}",
        arch="qwen3_5",
        output_dirname="gender_en_qwen3_5_2b_base",
    ),
    "qwen3.5-9b-base": StudyModel(
        key="qwen3.5-9b-base",
        sae_model="Qwen/Qwen3.5-9B-Base",
        hf_model="Qwen/Qwen3.5-9B-Base",
        sae_release="qwen-scope-3.5-9b-base-w64k-l50",
        n_layers=32,
        hf_resid_template="model.layers.{layer}",
        sae_id_template="layer{layer}",
        arch="qwen3_5",
        output_dirname="gender_en_qwen3_5_9b_base",
    ),
    "qwen3.5-35b-a3b-base": StudyModel(
        key="qwen3.5-35b-a3b-base",
        sae_model="Qwen/Qwen3.5-35B-A3B-Base",
        hf_model="Qwen/Qwen3.5-35B-A3B-Base",
        sae_release="qwen-scope-3.5-35b-a3b-base-w32k-l50",
        n_layers=40,
        hf_resid_template="model.layers.{layer}",
        sae_id_template="layer{layer}",
        arch="qwen3_5_moe",
        output_dirname="gender_en_qwen3_5_35b_a3b_base",
    ),
    # Qwen3.5-27B (dense, post-trained: no -Base release exists); mirrors configs/models/qwen3.5-27b.yaml.
    "qwen3.5-27b": StudyModel(
        key="qwen3.5-27b",
        sae_model="Qwen/Qwen3.5-27B",
        hf_model="Qwen/Qwen3.5-27B",
        sae_release="qwen-scope-3.5-27b-w80k-l50",
        n_layers=64,
        hf_resid_template="model.layers.{layer}",
        sae_id_template="layer{layer}",
        arch="qwen3_5",
        output_dirname="gender_en_qwen3_5_27b",
    ),
    # Gemma 3 (Gemma Scope 2 resid_post_all, width 16k / small L0)
    "gemma-3-270m": StudyModel(
        key="gemma-3-270m",
        sae_model="google/gemma-3-270m",
        hf_model="google/gemma-3-270m",
        sae_release="gemma-scope-2-270m-pt-res-all",
        n_layers=18,
        hf_resid_template="model.layers.{layer}",
        sae_id_template="layer_{layer}_width_16k_l0_small",
        arch="gemma3",
        output_dirname="gender_en_gemma_3_270m",
    ),
    "gemma-2-2b": StudyModel(
        key="gemma-2-2b",
        sae_model="google/gemma-2-2b",
        hf_model="google/gemma-2-2b",
        sae_release="gemma-scope-2b-pt-res-canonical",
        n_layers=26,
        hf_resid_template="model.layers.{layer}",
        sae_id_template="layer_{layer}/width_16k/canonical",
        arch="gemma2",
        output_dirname="gender_en_gemma_2_2b",
    ),
    "gemma-3-4b-pt": StudyModel(
        key="gemma-3-4b-pt",
        sae_model="google/gemma-3-4b-pt",
        hf_model="google/gemma-3-4b-pt",
        sae_release="gemma-scope-2-4b-pt-res-all",
        n_layers=34,
        hf_resid_template="model.layers.{layer}",
        sae_id_template="layer_{layer}_width_16k_l0_small",
        arch="gemma3",
        output_dirname="gender_en_gemma_3_4b_pt",
    ),
    "gemma-3-27b-pt": StudyModel(
        key="gemma-3-27b-pt",
        sae_model="google/gemma-3-27b-pt",
        hf_model="google/gemma-3-27b-pt",
        sae_release="gemma-scope-2-27b-pt-res-all",
        n_layers=62,
        hf_resid_template="model.layers.{layer}",
        sae_id_template="layer_{layer}_width_16k_l0_small",
        arch="gemma3",
        output_dirname="gender_en_gemma_3_27b_pt",
    ),
}

# Default study backbone (override via CLI ``--model``).
DEFAULT_STUDY_MODEL = "pythia-70m-deduped"

_ACTIVE: Optional[StudyModel] = None


def get_study_model(key: Optional[str] = None) -> StudyModel:
    """Resolve a preset by key; ``None`` → active / default."""
    if key is None:
        if _ACTIVE is not None:
            return _ACTIVE
        key = DEFAULT_STUDY_MODEL
    k = str(key).strip()
    if k not in STUDY_MODELS:
        known = ", ".join(sorted(STUDY_MODELS))
        raise ValueError(f"Unknown study model {key!r}. Known: {known}")
    return STUDY_MODELS[k]


def set_active_model(key_or_model: str | StudyModel) -> StudyModel:
    """Set process-wide active model (used by sae_eval / localization helpers)."""
    global _ACTIVE
    model = key_or_model if isinstance(key_or_model, StudyModel) else get_study_model(key_or_model)
    _ACTIVE = model
    return model


def active_model() -> StudyModel:
    return get_study_model(None)


def resid_sites(layer: int) -> Tuple[str, str]:
    """``(hf_module, sae_id)`` for the active study model."""
    return active_model().sites(int(layer))


