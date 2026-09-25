"""Signal extraction and post-hoc decoder utilities of the study.

``package_adapter``/``schema``/``projection``/``learned``/``moments`` extract the
paired factual/alternative signals that CAGA's closed-form fit reuses
(``caga_eval.py``). ``decoder_causal``/``bridge``/``evaluation``/``spectral``
implement the opt-in ``actiend_ridge`` post-hoc decoder variant fitted by
``causal_study.py`` (enabled in the ``full``/``full_plus`` suites; not part of
the paper's headline results).
"""

from study.signals.bridge import (
    FrozenEncoderRidgeFit,
    LearnedSpectralBridgeResult,
    evaluate_learned_spectral_bridge,
    fit_frozen_encoder_ridge_decoder,
)
from study.signals.decoder_causal import (
    DecoderCausalVariant,
    apply_decoder_causal_variant,
    fit_checkpoint_ridge_decoders,
    ridge_causal_variants,
)
from study.signals.evaluation import (
    CrossSplitRecovery,
    cross_split_rank_one_recovery,
    directional_counterpart_batch,
    directed_one_pole_view,
    infer_contrast_mode,
    paired_source_mse_comparison,
    target_geometry_summary,
)
from study.signals.learned import LearnedIendAccumulator, LearnedIendMeasurements
from study.signals.package_adapter import PairedSignalExtraction
from study.signals.projection import CountSketchProjector
from study.signals.schema import CompiledIendBatch, IendSignalBatch, SourceKind
from study.signals.spectral import (
    RankKSpectralResult,
    SpectralPermutationNull,
    SpectralResult,
    reduced_rank_spectral,
    reduced_rank_spectral_k,
    spectral_permutation_null,
)

__all__ = [
    "CompiledIendBatch",
    "CountSketchProjector",
    "CrossSplitRecovery",
    "DecoderCausalVariant",
    "FrozenEncoderRidgeFit",
    "IendSignalBatch",
    "LearnedIendAccumulator",
    "LearnedIendMeasurements",
    "LearnedSpectralBridgeResult",
    "PairedSignalExtraction",
    "RankKSpectralResult",
    "SourceKind",
    "SpectralPermutationNull",
    "SpectralResult",
    "apply_decoder_causal_variant",
    "cross_split_rank_one_recovery",
    "directional_counterpart_batch",
    "directed_one_pole_view",
    "evaluate_learned_spectral_bridge",
    "fit_frozen_encoder_ridge_decoder",
    "fit_checkpoint_ridge_decoders",
    "infer_contrast_mode",
    "paired_source_mse_comparison",
    "reduced_rank_spectral",
    "reduced_rank_spectral_k",
    "ridge_causal_variants",
    "spectral_permutation_null",
    "target_geometry_summary",
]
