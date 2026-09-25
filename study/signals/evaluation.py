"""Held-out evaluation for source-aware IEND recovery theory."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence

import numpy as np

from study.signals.moments import antisymmetry_error, row_cosines
from study.signals.schema import CompiledIendBatch, IendSignalBatch, SourceKind
from study.signals.spectral import (
    SpectralResult,
    _factor_source,
    reduced_rank_spectral,
)


DEFAULT_RIDGE_GRID = (1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0)


def _pearson(left: Any, right: Any) -> float:
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    if len(a) != len(b) or len(a) < 2:
        return float("nan")
    a = a - np.mean(a)
    b = b - np.mean(b)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 0.0 else float("nan")


def _binary_auc(scores: Any, labels: Any) -> float:
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    truth = np.asarray(labels, dtype=bool).reshape(-1)
    if len(values) != len(truth):
        raise ValueError("scores and labels must have equal length")
    positive = values[truth]
    negative = values[~truth]
    if len(positive) == 0 or len(negative) == 0:
        return float("nan")
    # Pairwise form handles ties and avoids a scipy/sklearn dependency.
    comparisons = positive[:, None] - negative[None, :]
    return float(np.mean(comparisons > 0.0) + 0.5 * np.mean(comparisons == 0.0))


def infer_contrast_mode(batch: IendSignalBatch) -> str:
    signs = set(np.sign(batch.orientation[batch.orientation != 0]).astype(int).tolist())
    if signs == {-1, 1}:
        return "pair"
    if len(signs) == 1:
        return "one_pole"
    return "unlabeled"


def directed_one_pole_view(
    batch: IendSignalBatch,
    *,
    factual_sign: int,
    include_neutral: bool = True,
) -> IendSignalBatch:
    """Keep one factual→counterfactual direction, optionally with neutral identities."""
    sign = int(factual_sign)
    if sign not in (-1, 1):
        raise ValueError("factual_sign must be -1 or +1")
    selected = batch.orientation == sign
    if include_neutral:
        selected |= batch.orientation == 0
    n_feature = int(np.sum(batch.orientation == sign))
    n_neutral = int(np.sum(batch.orientation == 0))
    if n_feature == 0:
        raise ValueError(f"directed one-pole view has no factual-sign {sign} rows")
    orientation = (batch.orientation[selected] == sign).astype(np.float64)
    pole_name = "positive" if sign > 0 else "negative"
    return batch.subset(
        selected,
        orientation=orientation,
        metadata={
            "derived_contrast": f"{pole_name}_directed_one_pole",
            "factual_sign_in_parent": sign,
            "includes_neutral_identity_rows": bool(include_neutral and n_neutral),
        },
    )


def directional_counterpart_batch(
    batch: IendSignalBatch,
    *,
    factual_sign: int,
    source_kind: SourceKind,
) -> CompiledIendBatch:
    """Build the unseen reverse-context support test for a directed one-pole fit.

    If training keeps A→B rows, factual-source transfer uses counterfactual A
    signals from B→A contexts, while alternative-source transfer uses natural
    factual B signals from B→A contexts. Targets are oriented back to the A→B
    reconstruction direction. `both` retains the natural roles of both reverse
    signals and their corresponding signs.
    """
    sign = int(factual_sign)
    if sign not in (-1, 1):
        raise ValueError("factual_sign must be -1 or +1")
    if source_kind == "diff":
        raise ValueError("diff is not a source-support transfer condition")
    reverse = batch.orientation == -sign
    if not np.any(reverse):
        raise ValueError(f"no reverse factual-sign {-sign} rows are available")
    factual = batch.factual[reverse]
    alternative = batch.alternative[reverse]
    reverse_difference = factual - alternative
    aligned_target = -reverse_difference
    ids = (
        tuple(np.asarray(batch.sample_ids, dtype=object)[reverse].tolist())
        if batch.sample_ids is not None
        else None
    )
    if source_kind == "factual":
        # Natural A during training → counterfactual A in reverse contexts.
        source = alternative
        target = aligned_target
        support = "counterfactual_same_class"
    elif source_kind == "alternative":
        # Counterfactual B during training → natural factual B contexts.
        source = factual
        target = aligned_target
        support = "natural_counterfactual_class"
    elif source_kind == "both":
        # Reverse factual B has target -d; reverse alternative A has target +d.
        source = np.concatenate([factual, alternative], axis=0)
        target = np.concatenate([reverse_difference, aligned_target], axis=0)
        support = "reverse_context_both_poles"
        if ids is not None:
            ids = tuple(f"{value}:reverse-factual" for value in ids) + tuple(
                f"{value}:reverse-alternative" for value in ids
            )
    else:
        raise ValueError(f"unknown source_kind {source_kind!r}")
    return CompiledIendBatch(
        source=source,
        target=target,
        orientation=np.zeros(len(source), dtype=np.float64),
        factual_orientation=np.zeros(len(source), dtype=np.float64),
        source_kind=source_kind,
        sample_ids=ids,
        metadata={
            **dict(batch.metadata or {}),
            "diagnostic_role": "opposite_context_support_transfer",
            "support_kind": support,
            "training_factual_sign": sign,
        },
    )


def target_geometry_summary(batch: IendSignalBatch) -> Dict[str, Any]:
    """Describe target differences without pretending they are source predictions."""
    target = batch.factual - batch.alternative
    centered = target - target.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    energy = singular_values**2
    total = float(np.sum(energy))
    rank_one = float(energy[0] / total) if total > 0.0 else float("nan")
    sigma2 = float(singular_values[1]) if len(singular_values) > 1 else 0.0
    gap = (
        float((singular_values[0] - sigma2) / singular_values[0])
        if len(singular_values) and singular_values[0] > 0.0
        else float("nan")
    )
    mode = infer_contrast_mode(batch)
    result: Dict[str, Any] = {
        "diagnostic_role": "target_geometry",
        "contrast_mode": mode,
        "target_pca_rank_one_fraction": rank_one,
        "target_pca_spectral_gap_fraction": gap,
        "target_pca_sigma1": float(singular_values[0]) if len(singular_values) else 0.0,
        "target_pca_sigma2": sigma2,
        "target_antisymmetry_error_empirical": (
            antisymmetry_error(target, batch.orientation) if mode == "pair" else float("nan")
        ),
    }
    feature = batch.orientation != 0
    nonzero = target[feature]
    if mode == "one_pole" and len(nonzero):
        mean_direction = nonzero.mean(axis=0)
        tiled = np.repeat(mean_direction.reshape(1, -1), len(nonzero), axis=0)
        cosines = row_cosines(nonzero, tiled)
        finite = np.abs(cosines[np.isfinite(cosines)])
        result["one_pole_target_direction_coherence_abs_mean"] = (
            float(np.mean(finite)) if len(finite) else float("nan")
        )
    else:
        result["one_pole_target_direction_coherence_abs_mean"] = float("nan")
    if mode == "one_pole" and batch.alternative_classes is not None:
        alternatives = np.asarray(batch.alternative_classes, dtype=object)
        group_means = []
        group_names = []
        for name in sorted(set(alternatives[feature].tolist())):
            mask = feature & (alternatives == name)
            if np.any(mask):
                group_names.append(str(name))
                group_means.append(target[mask].mean(axis=0))
        pairwise = []
        for left in range(len(group_means)):
            for right in range(left + 1, len(group_means)):
                denom = float(
                    np.linalg.norm(group_means[left]) * np.linalg.norm(group_means[right])
                )
                if denom > 0.0:
                    pairwise.append(
                        abs(float(np.dot(group_means[left], group_means[right]) / denom))
                    )
        result["one_pole_counterfactual_groups"] = group_names
        result["one_pole_counterfactual_group_count"] = len(group_names)
        result["one_pole_counterfactual_mean_direction_cosine_abs_mean"] = (
            float(np.mean(pairwise)) if pairwise else float("nan")
        )
        result["one_pole_counterfactual_mean_direction_cosine_abs_min"] = (
            float(np.min(pairwise)) if pairwise else float("nan")
        )
    result["constructed_antisymmetry"] = False
    return result


def _prediction_metrics(
    model: SpectralResult,
    batch: CompiledIendBatch,
    *,
    contrast_mode: str,
    score_sign: float,
) -> Dict[str, float]:
    prediction = model.predict(batch.source)
    residual = prediction - batch.target
    mse = float(np.mean(residual**2))
    baseline = float(np.mean((batch.target - model.target_mean) ** 2))
    cosines = row_cosines(prediction - model.target_mean, batch.target - model.target_mean)
    finite = cosines[np.isfinite(cosines)]
    score = score_sign * model.score(batch.source)
    out = {
        "reconstruction_mse": mse,
        "constant_mean_mse": baseline,
        "heldout_explained_variance": float(1.0 - mse / baseline)
        if baseline > 0.0
        else float("nan"),
        "prediction_target_cosine_mean": float(np.mean(finite))
        if len(finite)
        else float("nan"),
    }
    if contrast_mode == "pair":
        # Labels describe the signal actually exposed as source. In particular,
        # alternative and the second half of both have inverted class polarity.
        mask = batch.orientation != 0
        labels = batch.orientation[mask]
        out["feature_score_correlation"] = _pearson(score[mask], labels)
        out["feature_score_auc"] = _binary_auc(score[mask], labels > 0)
    elif contrast_mode == "one_pole":
        labels = batch.factual_orientation != 0
        if batch.source_kind == "both":
            # The two rows for one feature example have opposite constructed
            # reconstruction targets. Signed AUC would therefore cancel and
            # is not a feature-presence statistic.
            out["directed_vs_neutral_score_correlation"] = float("nan")
            out["directed_vs_neutral_score_auc"] = float("nan")
            out["directed_vs_neutral_abs_score_correlation"] = _pearson(
                np.abs(score), labels.astype(float)
            )
            out["directed_vs_neutral_abs_score_auc"] = _binary_auc(
                np.abs(score), labels
            )
            pole = batch.orientation != 0
            out["constructed_pole_score_correlation"] = _pearson(
                score[pole], batch.orientation[pole]
            )
        else:
            out["directed_vs_neutral_score_correlation"] = _pearson(
                score, labels.astype(float)
            )
            out["directed_vs_neutral_score_auc"] = _binary_auc(score, labels)
    else:
        out["feature_score_correlation"] = float("nan")
        out["feature_score_auc"] = float("nan")
    return out


def _orientation_sign(
    model: SpectralResult,
    batch: CompiledIendBatch,
    *,
    contrast_mode: str,
) -> float:
    score = model.score(batch.source)
    if contrast_mode == "pair":
        mask = batch.orientation != 0
        association = _pearson(score[mask], batch.orientation[mask])
    elif contrast_mode == "one_pole":
        feature = batch.factual_orientation != 0
        if not np.any(feature) or np.all(feature):
            return 1.0
        association = float(np.mean(score[feature]) - np.mean(score[~feature]))
    else:
        return 1.0
    return -1.0 if np.isfinite(association) and association < 0.0 else 1.0


def _compile_with_permuted_targets(
    batch: IendSignalBatch,
    source_kind: SourceKind,
    permutation: np.ndarray,
) -> CompiledIendBatch:
    compiled = batch.compile(source_kind)
    difference = batch.factual - batch.alternative
    permuted = difference[permutation]
    if source_kind == "both":
        target = np.concatenate([permuted, -permuted], axis=0)
    else:
        target = permuted
    return CompiledIendBatch(
        source=compiled.source,
        target=target,
        orientation=compiled.orientation,
        source_kind=source_kind,
        factual_orientation=compiled.factual_orientation,
        pole=compiled.pole,
        sample_ids=compiled.sample_ids,
        metadata=compiled.metadata,
    )


@dataclass(frozen=True)
class CrossSplitRecovery:
    source_kind: SourceKind
    contrast_mode: str
    selected_ridge: float
    model: SpectralResult
    validation_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    ridge_validation_mse: Dict[str, float]
    permutation_test_explained_variance: np.ndarray
    permutation_models: tuple[SpectralResult, ...]
    permutation_seed: int

    def evaluate_compiled(self, batch: CompiledIendBatch) -> Dict[str, float]:
        """Evaluate reconstruction on an externally constructed support set."""
        return _prediction_metrics(
            self.model,
            batch,
            contrast_mode="unlabeled",
            score_sign=1.0,
        )

    def evaluate_compiled_with_permutation(
        self, batch: CompiledIendBatch
    ) -> Dict[str, float]:
        """Evaluate an external support set against the same permuted-train fits."""
        metrics = self.evaluate_compiled(batch)
        observed = float(metrics["heldout_explained_variance"])
        null = np.asarray(
            [
                _prediction_metrics(
                    model,
                    batch,
                    contrast_mode="unlabeled",
                    score_sign=1.0,
                )["heldout_explained_variance"]
                for model in self.permutation_models
            ],
            dtype=np.float64,
        )
        out = dict(metrics)
        if len(null):
            out.update(
                permutation_test_explained_variance_mean=float(np.mean(null)),
                permutation_test_explained_variance_q95=float(np.quantile(null, 0.95)),
                permutation_effect_size=observed - float(np.mean(null)),
                permutation_p_upper=float(
                    (1 + np.sum(null >= observed)) / (len(null) + 1)
                ),
            )
        return out

    def summary(self) -> Dict[str, Any]:
        null = self.permutation_test_explained_variance
        observed = float(self.test_metrics["heldout_explained_variance"])
        out: Dict[str, Any] = {
            "diagnostic_role": "heldout_source_recovery",
            "source_kind": self.source_kind,
            "contrast_mode": self.contrast_mode,
            "selected_ridge": self.selected_ridge,
            "ridge_validation_mse": self.ridge_validation_mse,
            **{f"validation_{k}": v for k, v in self.validation_metrics.items()},
            **{f"test_{k}": v for k, v in self.test_metrics.items()},
            **{f"train_{k}": v for k, v in self.model.summary().items()},
        }
        if len(null):
            out.update(
                permutation_n=int(len(null)),
                permutation_seed=self.permutation_seed,
                permutation_test_explained_variance_mean=float(np.mean(null)),
                permutation_test_explained_variance_q95=float(np.quantile(null, 0.95)),
                permutation_effect_size=observed - float(np.mean(null)),
                permutation_p_upper=float(
                    (1 + np.sum(null >= observed)) / (len(null) + 1)
                ),
            )
        return out


def paired_source_mse_comparison(
    factual: CrossSplitRecovery,
    alternative: CrossSplitRecovery,
    test: IendSignalBatch,
    *,
    n_bootstrap: int = 2000,
    seed: int = 0,
) -> Dict[str, Any]:
    """Paired uncertainty for factual versus alternative held-out prediction error."""
    factual_view = test.compile("factual")
    alternative_view = test.compile("alternative")
    if not np.allclose(factual_view.target, alternative_view.target):
        raise ValueError("factual and alternative comparison targets differ")
    factual_prediction = factual.model.predict(factual_view.source)
    alternative_prediction = alternative.model.predict(alternative_view.source)
    factual_error = np.mean((factual_prediction - factual_view.target) ** 2, axis=1)
    alternative_error = np.mean(
        (alternative_prediction - alternative_view.target) ** 2, axis=1
    )
    paired_difference = factual_error - alternative_error
    observed = float(np.mean(paired_difference))
    denominator = float((np.mean(factual_error) + np.mean(alternative_error)) / 2.0)
    n_bootstrap = int(n_bootstrap)
    rng = np.random.default_rng(int(seed))
    bootstrap = np.empty(max(0, n_bootstrap), dtype=np.float64)
    for index in range(len(bootstrap)):
        sampled = rng.integers(0, len(paired_difference), size=len(paired_difference))
        bootstrap[index] = float(np.mean(paired_difference[sampled]))
    if len(bootstrap):
        ci_low, ci_high = np.quantile(bootstrap, (0.025, 0.975))
    else:
        ci_low = ci_high = float("nan")
    if ci_low > 0.0:
        conclusion = "alternative_lower_mse"
    elif ci_high < 0.0:
        conclusion = "factual_lower_mse"
    else:
        conclusion = "inconclusive"
    return {
        "diagnostic_role": "paired_source_comparison",
        "source_a": "factual",
        "source_b": "alternative",
        "n_paired_test_rows": int(len(paired_difference)),
        "mse_factual": float(np.mean(factual_error)),
        "mse_alternative": float(np.mean(alternative_error)),
        "mse_difference_factual_minus_alternative": observed,
        "mse_relative_difference_vs_mean": (
            float(observed / denominator) if denominator > 0.0 else float("nan")
        ),
        "bootstrap_n": int(len(bootstrap)),
        "bootstrap_seed": int(seed),
        "mse_difference_ci95_low": float(ci_low),
        "mse_difference_ci95_high": float(ci_high),
        "source_comparison_conclusion": conclusion,
    }

def cross_split_rank_one_recovery(
    train: IendSignalBatch,
    validation: IendSignalBatch,
    test: IendSignalBatch,
    *,
    source_kind: SourceKind,
    ridge_grid: Sequence[float] = DEFAULT_RIDGE_GRID,
    relative_ridge: bool = True,
    n_permutations: int = 0,
    permutation_seed: int = 0,
) -> CrossSplitRecovery:
    """Fit on train, select ridge on validation, and report untouched test metrics."""
    if source_kind == "diff":
        raise ValueError("diff is an oracle target-geometry view, not a source-recovery test")
    train_view = train.compile(source_kind)
    validation_view = validation.compile(source_kind)
    test_view = test.compile(source_kind)
    mode = infer_contrast_mode(train)
    candidates = tuple(float(value) for value in ridge_grid)
    if not candidates or any(value < 0.0 for value in candidates):
        raise ValueError("ridge_grid must contain non-negative values")
    fits: Dict[float, SpectralResult] = {}
    validation_by_ridge: Dict[float, Dict[str, float]] = {}
    train_factorization = _factor_source(train_view.source)
    for ridge in candidates:
        model = reduced_rank_spectral(
            train_view.source,
            train_view.target,
            ridge=ridge,
            relative_ridge=relative_ridge,
            _source_factorization=train_factorization,
        )
        sign = _orientation_sign(model, train_view, contrast_mode=mode)
        fits[ridge] = model
        validation_by_ridge[ridge] = _prediction_metrics(
            model, validation_view, contrast_mode=mode, score_sign=sign
        )
    selected = min(candidates, key=lambda value: validation_by_ridge[value]["reconstruction_mse"])
    model = fits[selected]
    sign = _orientation_sign(model, train_view, contrast_mode=mode)
    validation_metrics = validation_by_ridge[selected]
    test_metrics = _prediction_metrics(
        model, test_view, contrast_mode=mode, score_sign=sign
    )

    n_permutations = int(n_permutations)
    null = np.empty(max(0, n_permutations), dtype=np.float64)
    permutation_models = []
    rng = np.random.default_rng(int(permutation_seed))
    for index in range(len(null)):
        permuted_train = _compile_with_permuted_targets(
            train, source_kind, rng.permutation(train.n_samples)
        )
        permuted_model = reduced_rank_spectral(
            permuted_train.source,
            permuted_train.target,
            ridge=selected,
            relative_ridge=relative_ridge,
            _source_factorization=train_factorization,
        )
        permuted_sign = _orientation_sign(
            permuted_model, train_view, contrast_mode=mode
        )
        permutation_models.append(permuted_model)
        null[index] = _prediction_metrics(
            permuted_model,
            test_view,
            contrast_mode=mode,
            score_sign=permuted_sign,
        )["heldout_explained_variance"]

    return CrossSplitRecovery(
        source_kind=source_kind,
        contrast_mode=mode,
        selected_ridge=selected,
        model=model,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        ridge_validation_mse={
            str(ridge): values["reconstruction_mse"]
            for ridge, values in validation_by_ridge.items()
        },
        permutation_test_explained_variance=null,
        permutation_models=tuple(permutation_models),
        permutation_seed=int(permutation_seed),
    )
