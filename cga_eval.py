"""CGA — Contrastive Gradient Addition (untrained gradient-space mean-diff).

CGA is the gradient-signal analogue of CAA (``caa_eval.py``) and the untrained
null for GRADIEND.  It completes the study's (signal) x (estimator) grid::

                     mean-difference        learned encoder-decoder
    activation       CAA                    ACTIEND
    gradient         CGA  (this module)     GRADIEND

Without it, "GRADIEND beats CAA" changes both factors at once and cannot say
whether the win comes from the gradient signal or from learning an
encoder-decoder.  CGA holds the signal fixed at GRADIEND's and removes the
learning.

Estimator
---------
For every paired (factual, counterfactual) training row the package already
produces ``target = grad(factual) - grad(alternative)`` over GRADIEND's own
scoped parameter space (``TrainingArguments.target='diff'``).  CGA is the
class-signed mean of exactly those vectors::

    delta = sum_i label_i * target_i / sum_i label_i**2

with ``label_i`` the row's +-1 feature-pole label (0 for neutral identity rows,
which drop out of both sums).  This is the closed form of the package's own
``supervised_decoder`` baseline mode (``decoder(label) ~ target`` under MSE),
computed directly instead of by SGD so there is no seed, no convergence
threshold, and no checkpoint selection.  ``delta`` is then normalized to unit
norm, exactly like ``caa_eval.normalize_vec`` does for activations.

The signed paired expression is the computational form of the semantic
contrastive mean ``mu_positive - mu_negative``.  Multiplication by ``label_i``
reorients factual-minus-alternative gradients when the factual class swaps; an
unoriented average of those differences could cancel.  It equals CAA's direct
class-pool expression when both constructions induce the same observation
weights.  For multiclass one-vs-rest data, CGA inherits the multiplicities of
the constructed pairs whereas CAA pools rival rows directly, so arbitrary
unbalanced data can give different numerical weighting despite the shared
intended estimator.

Interpretation: ``theta + alpha * delta`` is one step of contrastive
fine-tuning, i.e. the first-order form of a task vector
(``theta_finetuned - theta_pretrained``, Ilharco et al. 2023).

Deployment
----------
``delta`` is installed into a ``latent_dim=1`` GRADIEND encoder-decoder
(:func:`install_cga_direction`) rather than carried around as a bare vector, so
every existing evaluation path -- encoder metrics, the decoder grid, the causal
strength sweep, plots, method-id/report/LaTeX machinery -- runs unmodified:

* decoder (``bias_decoder=False``, identity activation) gives
  ``decode(ff) = ff * delta``, so the package's own
  ``rewrite_base_model(lr, ff)`` is precisely ``theta + lr * ff * delta``;
* encoder (``tanh``, as GRADIEND's own) gives the readout
  ``tanh(<delta, g(x)> / s)``.

**The encoder half of this checkpoint is never used and never installed.**
An earlier version routed the "how well does this direction separate the
classes" evaluation through the package's own ``trainer.evaluate_encoder()``
(``tanh(W*g)`` through a real ``nn.Module``) -- which is GRADIEND's readout
convention, not CAA's, and a strictly worse one for this purpose: it does not
discard the magnitude of ``g(x)`` the way cosine does, so two rows with
identical directional alignment to Delta but different gradient norms score
differently. That reintroduces exactly the confound CAA's cosine similarity
is designed to remove, which defeats the point of holding the estimator fixed
across the (signal) x (estimator) grid. (There is also no way to route around
it with an unsquashed readout: the package's ``activation_encoder="id"``
resolves to ``nn.LayerNorm(1)`` for an encoder, which maps every scalar latent
to exactly 0 -- a real trap, see the CLAUDE.md gotcha.)

Instead, :func:`fair_cga_encoder_eval` computes ``cos(g(x), Delta)`` directly
from a fresh per-row gradient (:func:`score_labeled_rows` /
:func:`score_neutral_texts`). On full-width models the package reduces the dot
product and norms parameter-by-parameter during backward, without ever
materializing a second flat model-width tensor, and feeds those scores into
the *same* ``sae_eval.class_vs_neutral_metrics`` CAA and SAE already use.
Neutral (free-text, unmasked) rows get a gradient via the package's own
``create_masked_pair_from_text`` convention (auto-picks a token to mask and
predict) -- the same mechanism the package's own neutral-dataset encoder
evaluation already uses internally, just reused directly here instead of
re-derived.

Only the decoder half of the installed checkpoint is real: ``decode(ff) = ff *
Delta`` is what the causal strength sweep uses via
``rewrite_base_model``, and that path never calls ``.encode()`` at all
(confirmed against the package: ``evaluate_decoder`` sweeps ``feature_factor``
purely through the decoder/rewrite, never the encoder) -- so reusing
``ModelWithGradiend`` for the weight-space intervention is legitimate; reusing
it for the *readout* was the actual bug.

Memory
------
The scoped parameter space is large (~85M entries for gpt2-small under
``SignalScope.layers()``), so per-example gradients are **never** stored.
:class:`ContrastiveGradientAccumulator` keeps a single running mean and folds
each row in as it arrives, so fit memory is O(input_dim), independent of the
number of rows.  ``accumulate_device`` moves that one buffer off the GPU when
VRAM is tight, at the cost of one host transfer per row.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd


from study.method_ids import (
    CGA_DEFAULT_VARIANT,
    CGA_VARIANTS,
    cga_backend_id,
    cga_backend_ids,
    cga_variant_from_backend,
    resolve_cga_variants,
)

__all__ = [
    "CGA_DEFAULT_VARIANT",
    "CGA_ENCODER_EVAL_VERSION",
    "CGA_LAYERWISE_ENCODER_EVAL_VERSION",
    "CGA_VARIANTS",
    "ContrastiveGradientAccumulator",
    "CgaFit",
    "cga_backend_id",
    "cga_backend_ids",
    "cga_causal_id",
    "cga_layer_slices",
    "cga_variant_from_backend",
    "fit_cga_direction",
    "set_cga_direction_semantics",
    "install_cga_direction",
    "normalize_direction",
    "param_segment_bounds",
    "resolve_cga_variants",
    "tensor_normalized_direction",
]


class ContrastiveGradientAccumulator:
    """Stream the orientation-corrected contrastive mean in O(input_dim) memory.

    Keeps an incremental mean of ``label_i * target_i`` (Welford-style update,
    so no unbounded running sum) plus the scalar ``sum(label**2)`` needed to
    turn that mean back into the least-squares direction.  Rows with a zero or
    non-finite label contribute nothing and are counted separately.  Here
    ``target_i`` is the factual-minus-alternative gradient and ``label_i``
    orients it from the negative semantic pole toward the positive one.  For
    balanced paired observations this is algebraically the same direction as
    ``mean(positive signals) - mean(negative signals)``.
    """

    def __init__(
        self,
        size: int,
        *,
        device: Any = "cpu",
        dtype: torch.dtype = torch.float32,
        buffer: Optional[torch.Tensor] = None,
    ) -> None:
        if int(size) <= 0:
            raise ValueError(f"size must be positive, got {size}")
        self.size = int(size)
        if buffer is not None:
            flat = buffer.detach().reshape(-1)
            if flat.numel() != self.size:
                raise ValueError(
                    f"buffer has {flat.numel()} entries, accumulator expects {self.size}"
                )
            self.device = flat.device
            self.dtype = flat.dtype
            self._mean = flat
            with torch.no_grad():
                self._mean.zero_()
        else:
            self.device = torch.device(device)
            self.dtype = dtype
            self._mean = torch.zeros(self.size, device=self.device, dtype=self.dtype)
        self._n = 0
        self._label_sq = 0.0
        self._label_sum = 0.0
        self._skipped_zero_label = 0
        self._skipped_nonfinite = 0

    @property
    def n_used(self) -> int:
        return self._n

    @property
    def n_skipped_zero_label(self) -> int:
        return self._skipped_zero_label

    @property
    def n_skipped_nonfinite(self) -> int:
        return self._skipped_nonfinite

    @property
    def label_sum(self) -> float:
        return float(self._label_sum)

    def update(self, target: torch.Tensor, label: Any) -> bool:
        """Fold one row's ``target`` gradient in. Returns True when counted."""
        try:
            lab = float(label)
        except (TypeError, ValueError):
            self._skipped_zero_label += 1
            return False
        if not math.isfinite(lab) or lab == 0.0:
            self._skipped_zero_label += 1
            return False
        flat = target.detach().reshape(-1)
        if flat.numel() != self.size:
            raise ValueError(
                f"target has {flat.numel()} entries, accumulator expects {self.size}"
            )
        # Check in bounded chunks: torch.isfinite over a 7B-element vector would
        # itself allocate several GiB of temporary bool storage.
        finite = True
        chunk_size = 16 * 1024 * 1024
        for start in range(0, flat.numel(), chunk_size):
            if not bool(torch.isfinite(flat[start : start + chunk_size]).all()):
                finite = False
                break
        if not finite:
            self._skipped_nonfinite += 1
            return False
        self._n += 1
        self._label_sq += lab * lab
        self._label_sum += lab
        # mean_n = mean_(n-1) * (1 - 1/n) + label*x/n. Both operations are
        # in-place, avoiding the former full-width ``contrib`` temporary.
        with torch.no_grad():
            self._mean.mul_(1.0 - 1.0 / float(self._n))
            self._mean.add_(flat.to(device=self.device), alpha=lab / float(self._n))
        return True

    def direction(self) -> torch.Tensor:
        """Least-squares direction ``sum(label*target) / sum(label**2)``.

        ``sum(label*target) == n * mean``; dividing by ``sum(label**2)`` keeps
        the magnitude meaningful (it equals ``mean`` exactly for +-1 labels).
        """
        if self._n == 0 or self._label_sq <= 0.0:
            raise ValueError(
                "no labelled rows accumulated; cannot form a CGA direction "
                f"(skipped zero-label={self._skipped_zero_label}, "
                f"non-finite={self._skipped_nonfinite})"
            )
        scale = float(self._n) / float(self._label_sq)
        if scale != 1.0:
            with torch.no_grad():
                self._mean.mul_(scale)
        return self._mean


def normalize_direction(vec: torch.Tensor) -> torch.Tensor:
    """Unit-normalize, mirroring ``caa_eval.normalize_vec`` (zero stays zero)."""
    norm = _safe_norm(vec.detach().float())
    if not math.isfinite(norm) or norm <= 0.0:
        return vec.detach().clone()
    return vec.detach() / norm


def param_segment_bounds(gradiend_model: Any) -> List[Tuple[str, int, int]]:
    """``[(param_name, start, end)]`` over the flat GRADIEND input space.

    The input space is the concatenation of each mapped parameter's selected
    positions, in ``param_map`` insertion order -- the same convention
    ``ParamMappedGradiendModel`` uses when it remaps a pruned map.
    """
    param_map = getattr(gradiend_model, "param_map", None)
    if not param_map:
        raise ValueError(
            "gradiend model has no param_map; CGA needs the parameter-space layout"
        )
    bounds: List[Tuple[str, int, int]] = []
    offset = 0
    for name, spec in param_map.items():
        repr_kind = str(spec.get("repr") or "all")
        if repr_kind == "all":
            count = 1
            for dim in tuple(spec["shape"]):
                count *= int(dim)
        elif repr_kind == "mask":
            count = int(spec["mask"].sum().item())
        elif repr_kind == "indices":
            count = int(spec["indices"].numel())
        else:
            raise ValueError(f"unknown param_map repr {repr_kind!r} for {name!r}")
        bounds.append((str(name), offset, offset + count))
        offset += count
    expected = int(getattr(gradiend_model, "input_dim", offset))
    if offset != expected:
        raise ValueError(
            f"param_map covers {offset} entries but input_dim is {expected}"
        )
    return bounds


_COMMON_LAYER_RE = re.compile(r"(?:^|\.)(?:h|layers|layer)\.(\d+)(?:\.|$)")


def _layer_index_from_param_name(name: str, prefixes: Sequence[str] = ()) -> Optional[int]:
    value = str(name)
    for index, prefix in enumerate(prefixes):
        if value == prefix or value.startswith(f"{prefix}."):
            return int(index)
    match = _COMMON_LAYER_RE.search(value)
    return int(match.group(1)) if match else None


def cga_layer_slices(model: Any) -> Dict[int, Tuple[int, int]]:
    """Map residual layer index to its exact contiguous CGA coordinate slice.

    The bounds come from :func:`param_segment_bounds`, hence use precisely the
    persisted decoder/``param_map`` layout.  Model topology supplies the real
    architecture-specific block prefixes (GPT-2, GPT-NeoX, Llama/Qwen/Gemma,
    OPT, ...); a conservative name parser remains for lightweight/reloaded test
    doubles.  A non-contiguous or interleaved layer is rejected rather than
    silently masking unrelated coordinates.
    """
    gradiend_model = getattr(model, "gradiend", model)
    prefixes: Sequence[str] = ()
    base_model = getattr(model, "base_model", None)
    if base_model is not None:
        try:
            from gradiend.model_topology import infer_model_topology

            topology = infer_model_topology(base_model)
            if topology is not None:
                prefixes = tuple(str(x) for x in topology.layers)
        except Exception:
            prefixes = ()

    grouped: Dict[int, List[Tuple[int, int]]] = {}
    all_bounds = param_segment_bounds(gradiend_model)
    for name, start, end in all_bounds:
        layer = _layer_index_from_param_name(name, prefixes)
        if layer is not None and end > start:
            grouped.setdefault(layer, []).append((int(start), int(end)))

    out: Dict[int, Tuple[int, int]] = {}
    for layer, ranges in sorted(grouped.items()):
        ranges = sorted(ranges)
        for previous, current in zip(ranges, ranges[1:]):
            if previous[1] != current[0]:
                raise ValueError(
                    f"CGA coordinates for layer {layer} are not contiguous: {ranges}"
                )
        lo, hi = ranges[0][0], ranges[-1][1]
        interlopers = [
            name
            for name, start, end in all_bounds
            if start < hi
            and end > lo
            and _layer_index_from_param_name(name, prefixes) != layer
        ]
        if interlopers:
            raise ValueError(
                f"CGA layer {layer} slice would include unrelated parameters: {interlopers}"
            )
        out[int(layer)] = (int(lo), int(hi))
    if not out:
        raise ValueError("no residual-layer parameter ranges found in the CGA param_map")
    return out


def cga_causal_id(
    class_id: str,
    *,
    pair: Optional[str] = None,
    layer: Optional[int] = None,
    backend: str = "cga",
) -> str:
    """CGA method id, optionally resolved to one residual layer."""
    base = f"{backend}:{pair}:{class_id}" if pair else f"{backend}:{class_id}"
    return f"{base}:L{int(layer)}" if layer is not None else base


def tensor_normalized_direction(
    vec: torch.Tensor,
    bounds: Sequence[Tuple[str, int, int]],
) -> torch.Tensor:
    """Ablation variant: unit-normalize each base tensor's block, then globally.

    Raw gradient magnitude varies by orders of magnitude across tensors (an
    attention projection vs. a layer-norm gain), so an unweighted mean-diff over
    the concatenated space is dominated by whichever tensors happen to have the
    largest gradients.  This rescales each block first, so a weak ``plain``
    result cannot be blamed on that heterogeneity alone.  Empty blocks stay zero.
    """
    out = vec.detach().clone()
    for _name, start, end in bounds:
        if end <= start:
            continue
        block = out[start:end]
        norm = _safe_norm(block.float())
        if math.isfinite(norm) and norm > 0.0:
            block /= norm
    return normalize_direction(out)


@dataclass
class CgaFit:
    """Fitted CGA direction plus the provenance needed to read it later."""

    direction: torch.Tensor
    variant: str
    input_dim: int
    n_used: int
    n_rows_seen: int
    n_skipped_zero_label: int
    n_skipped_nonfinite: int
    label_sum: float
    class_counts: Dict[str, int] = field(default_factory=dict)
    raw_norm: float = 0.0
    split: str = "train"
    accumulate_device: str = "cpu"
    accumulate_dtype: str = "torch.float32"

    def stats(self) -> Dict[str, Any]:
        return {
            "variant": self.variant,
            "input_dim": int(self.input_dim),
            "n_used": int(self.n_used),
            "n_rows_seen": int(self.n_rows_seen),
            "n_skipped_zero_label": int(self.n_skipped_zero_label),
            "n_skipped_nonfinite": int(self.n_skipped_nonfinite),
            "label_sum": float(self.label_sum),
            "class_counts": dict(self.class_counts),
            "raw_norm": float(self.raw_norm),
            "split": str(self.split),
            "accumulate_device": str(self.accumulate_device),
            "accumulate_dtype": str(self.accumulate_dtype),
        }


def _row_class_key(item: Dict[str, Any]) -> Optional[str]:
    """Best-effort class label for fit diagnostics (never used in the math)."""
    for key in ("feature_class_id", "feature_pole", "factual_token"):
        val = item.get(key)
        if val is None:
            continue
        if isinstance(val, (list, tuple)):
            if len(val) != 1:
                return None
            val = val[0]
        if isinstance(val, torch.Tensor):
            if val.numel() != 1:
                return None
            val = val.item()
        return str(val)
    return None


def fit_cga_direction(
    trainer: Any,
    model_with_gradiend: Any,
    *,
    variant: str = CGA_DEFAULT_VARIANT,
    split: str = "train",
    max_size: Optional[int] = None,
    accumulate_device: Optional[Any] = None,
    progress_every: int = 25,
    output_dir: Optional[Any] = None,
) -> CgaFit:
    """Stream the paired training gradients and return the CGA direction.

    Rows are pulled one at a time (``base_gradient_batch_size=1``) so each item
    is a single (factual, counterfactual) pair with one scalar label -- a merged
    multi-row batch would yield one mixed gradient and a *list* of labels, which
    has no well-defined contrastive sign.  Gradients are folded into a running
    mean and released immediately; none are retained.

    Args:
        trainer: A constructed (untrained) ``TextPredictionTrainer``.
        model_with_gradiend: Its ``ModelWithGradiend`` (``trainer.get_model()``).
        variant: One of :data:`CGA_VARIANTS`.
        split: Data split to fit on. Fitting on anything but ``train`` leaks.
        max_size: Optional per-balance-group row cap passed to the trainer.
        accumulate_device: Device for the single running-mean buffer. ``None``
            keeps it wherever the gradients already are (no host transfer);
            pass ``"cpu"`` to keep it off the GPU when VRAM is tight.
        progress_every: Print a progress line every N rows (0 disables).
    """
    name = str(variant).strip().lower()
    if name not in CGA_VARIANTS:
        raise ValueError(
            f"Unknown CGA variant {name!r}. Choose from: {', '.join(CGA_VARIANTS)}"
        )
    gradiend_model = getattr(model_with_gradiend, "gradiend", None)
    if gradiend_model is None:
        raise ValueError("model_with_gradiend has no gradiend model")
    input_dim = int(getattr(gradiend_model, "input_dim", 0))
    if input_dim <= 0:
        raise ValueError(f"gradiend model has non-positive input_dim {input_dim}")

    training_data = trainer.create_training_data(
        model_with_gradiend,
        split=split,
        batch_size=1,
        max_size=max_size,
    )
    n_rows = len(training_data)
    if n_rows == 0:
        raise ValueError(
            f"no {split!r} rows available for the CGA fit; check the task frame"
        )
    dataset = trainer.create_gradient_training_dataset(
        training_data,
        model_with_gradiend,
        cache_dir=None,
        use_cached_gradients=False,
        # ``target='diff'`` is the paired contrast CGA averages. ``source`` is
        # never read here, but source='both' would alternate poles across
        # visits for no gain, so pin the cheap deterministic side.
        source="factual",
        target="diff",
        dtype=gradiend_model.torch_dtype,
        device=gradiend_model.device_encoder,
        timing_label="cga-fit",
        # CGA consumes and discards each target immediately.  Reusing the
        # factual gradient for the difference avoids a third full-width tensor
        # (~26 GiB for the 8B models) without introducing CPU transfers.
        combine_diff_in_place=True,
    )

    device = (
        gradiend_model.device_encoder
        if accumulate_device is None
        else torch.device(accumulate_device)
    )
    decoder_weight = gradiend_model.decoder[0].linear.weight
    reuse_decoder = (
        device.type != "cpu"
        and decoder_weight.device == device
        and decoder_weight.numel() == input_dim
        # The accumulator is a repeated running mean, not a persisted model
        # parameter.  Large bf16 models must still accumulate in fp32; reuse
        # the decoder only for the historical fp32 small-model fast path.
        and decoder_weight.dtype == torch.float32
    )
    acc = ContrastiveGradientAccumulator(
        input_dim,
        device=device,
        dtype=torch.float32,
        buffer=decoder_weight if reuse_decoder else None,
    )
    class_counts: Dict[str, int] = {}
    total = len(dataset)
    print(
        f"  cga fit: variant={name} split={split} rows={total} "
        f"input_dim={input_dim} accumulate_device={acc.device} "
        f"accumulate_dtype={acc.dtype} "
        f"storage={'decoder_in_place' if reuse_decoder else 'standalone'}",
        flush=True,
    )
    for index in range(total):
        item = dataset[index]
        target = item.get("target")
        if target is None:
            raise ValueError(
                "gradient dataset produced no target tensor; CGA requires "
                "target='diff' paired gradients"
            )
        label = item.get("label")
        if isinstance(label, (list, tuple)):
            if len(label) != 1:
                raise ValueError(
                    f"expected one label per row for the CGA fit, got {len(label)}; "
                    "the fit dataset must use batch_size=1"
                )
            label = label[0]
        if acc.update(target, label):
            key = _row_class_key(item)
            if key is not None:
                class_counts[key] = class_counts.get(key, 0) + 1
        del target
        del item
        if progress_every and (index + 1) % int(progress_every) == 0:
            print(
                f"  cga fit: {index + 1}/{total} rows "
                f"(used={acc.n_used} skipped={acc.n_skipped_zero_label})",
                flush=True,
            )
            try:
                from study.progress import write_pipeline_progress

                write_pipeline_progress(
                    output_dir, stage="cga_fit", variant=name, row=int(index + 1)
                )
            except Exception:
                pass

    raw = acc.direction()
    raw_norm = _safe_norm(raw.float())
    if name == "tensor_norm":
        direction = tensor_normalized_direction(
            raw, param_segment_bounds(gradiend_model)
        )
    else:
        norm = _safe_norm(raw.float())
        if math.isfinite(norm) and norm > 0.0:
            with torch.no_grad():
                raw.div_(norm)
        direction = raw
    if _safe_norm(direction.float()) <= 0.0:
        raise ValueError(
            "CGA direction is all zeros; the paired gradients cancelled exactly "
            f"(n_used={acc.n_used}, label_sum={acc.label_sum})"
        )
    print(
        f"  cga fit: done used={acc.n_used}/{total} "
        f"skipped(zero-label)={acc.n_skipped_zero_label} "
        f"skipped(non-finite)={acc.n_skipped_nonfinite} "
        f"raw_norm={raw_norm:.6g} classes={class_counts}",
        flush=True,
    )
    return CgaFit(
        direction=direction,
        variant=name,
        input_dim=input_dim,
        n_used=acc.n_used,
        n_rows_seen=total,
        n_skipped_zero_label=acc.n_skipped_zero_label,
        n_skipped_nonfinite=acc.n_skipped_nonfinite,
        label_sum=acc.label_sum,
        class_counts=class_counts,
        raw_norm=raw_norm,
        split=str(split),
        accumulate_device=str(acc.device),
        accumulate_dtype=str(acc.dtype),
    )


def direction_from_decoder(
    model_with_gradiend: Any,
    *,
    copy: bool = True,
) -> torch.Tensor:
    """Inverse of :func:`install_cga_direction` -- read ``delta`` back out of a
    reloaded checkpoint's decoder weight.

    A CGA checkpoint's ``direction`` is not persisted anywhere else (it is
    never installed into the encoder, and the fit's own copy is dropped after
    installation to free the 340MB-on-gpt2-small tensor) -- this is how a
    reload path recomputes encoding metrics for an artifact whose ``done.json``
    doesn't already have them cached.
    """
    gradiend_model = getattr(model_with_gradiend, "gradiend", None)
    if gradiend_model is None:
        raise ValueError("model_with_gradiend has no gradiend model")
    direction = gradiend_model.decoder[0].linear.weight.detach().reshape(-1)
    return direction.clone() if copy else direction


def set_cga_direction_semantics(model_with_gradiend: Any) -> None:
    """Record the fixed factual/diff orientation of a fitted CGA vector.

    The surrounding trainer may use another source, but
    :func:`fit_cga_direction` always accumulates ``label * (grad factual - grad
    alternative)``. Decoder feature-factor derivation must therefore read the
    estimator's orientation rather than the trainer's setting. This also
    repairs legacy CGA checkpoints whose decoder weights are valid but whose
    source metadata inherited the trainer setting.
    """
    setattr(model_with_gradiend, "_source", "factual")
    setattr(model_with_gradiend, "_target", "diff")


def install_cga_direction(model_with_gradiend: Any, direction: torch.Tensor) -> None:
    """Write ``direction`` into the latent_dim=1 decoder in place. Only the
    decoder is touched -- the encoder half is never used (see the module
    docstring) and is left exactly as constructed (random, irrelevant).

    After this call the wrapper's ``rewrite_base_model``/causal strength sweep
    behaves as a GRADIEND checkpoint whose weights were computed rather than
    trained::

        decode(ff)  = ff * direction
        rewrite(lr, ff) = theta + lr * ff * direction
    """
    set_cga_direction_semantics(model_with_gradiend)
    gradiend_model = getattr(model_with_gradiend, "gradiend", None)
    if gradiend_model is None:
        raise ValueError("model_with_gradiend has no gradiend model")
    if int(getattr(gradiend_model, "latent_dim", 0)) != 1:
        raise ValueError(
            "CGA needs latent_dim=1 (one direction), got "
            f"{getattr(gradiend_model, 'latent_dim', None)!r}"
        )
    decoder_linear = gradiend_model.decoder[0].linear
    flat = direction.detach().reshape(-1)
    if flat.numel() != int(gradiend_model.input_dim):
        raise ValueError(
            f"direction has {flat.numel()} entries, model input_dim is "
            f"{gradiend_model.input_dim}"
        )
    with torch.no_grad():
        destination = decoder_linear.weight.reshape(-1)
        if destination.data_ptr() != flat.data_ptr():
            destination.copy_(
                flat.to(device=destination.device, dtype=destination.dtype)
            )
        if decoder_linear.bias is not None:
            # A non-zero decoder bias would add a constant weight delta at
            # feature_factor=0, so the sweep would no longer pass through the
            # unmodified model.
            decoder_linear.bias.zero_()


# ---------------------------------------------------------------------------
# CAA-style encoding evaluation: cosine(g(x), Delta), never the package's
# encoder module. See the module docstring for why this is the correct design
# and what the earlier tanh-through-evaluate_encoder version got wrong.
# ---------------------------------------------------------------------------


# torch.dot / BLAS nrm2 pass the element count as an int32 `n`, so they raise
# "dot only supports n, incx, incy with the bound [val] <= 2147483647" on vectors
# with more than ~2.1B elements — which happens for a flat CGA gradient/direction on
# an 8B-parameter model (llama-3.1-8b), but NEVER for gpt2-small/pythia-70m (a few
# million elements). These helpers are byte-for-byte the original fast path when
# numel is under the limit (so smaller models are completely unaffected), and only
# fall back to a chunked reduction for oversized vectors.
_TORCH_DOT_INT32_LIMIT = 2147483647
_SAFE_DOT_CHUNK = 1 << 30  # 1,073,741,824 < int32 limit


def _safe_dot(a: torch.Tensor, b: torch.Tensor) -> float:
    n = a.numel()
    if n <= _TORCH_DOT_INT32_LIMIT:
        return float(torch.dot(a, b))
    total = 0.0
    for i in range(0, n, _SAFE_DOT_CHUNK):
        total += float(torch.dot(a[i:i + _SAFE_DOT_CHUNK], b[i:i + _SAFE_DOT_CHUNK]))
    return total


def _safe_norm(a: torch.Tensor) -> float:
    if a.numel() <= _TORCH_DOT_INT32_LIMIT:
        return float(torch.linalg.vector_norm(a))
    return math.sqrt(_safe_dot(a, a))


def cosine_to_direction(gradient: torch.Tensor, direction: torch.Tensor) -> float:
    """``<g, d> / (||g|| * ||d||)`` -- plain cosine, no calibration of any kind.

    Robust to ``direction`` not being exactly unit norm (it always is here,
    but this makes that an invariant this function doesn't silently rely on).
    """
    g = gradient.detach().reshape(-1).float()
    d = direction.detach().reshape(-1).float()
    # The gradient/activation comes off the model (often CUDA); the direction may be
    # a CPU tensor (e.g. built from a numpy fit). Align devices so torch.dot never
    # raises a device-mismatch that would silently drop every scored row.
    if d.device != g.device:
        d = d.to(g.device)
    norm_g = _safe_norm(g)
    norm_d = _safe_norm(d)
    if not math.isfinite(norm_g) or norm_g <= 0.0 or not math.isfinite(norm_d) or norm_d <= 0.0:
        return 0.0
    return _safe_dot(g, d) / (norm_g * norm_d)


_SINGLE_LABEL_TOKEN_ERROR = "Only a single label token is supported"
CGA_ENCODER_EVAL_VERSION = 2
# Version 3 adds the single-pass per-layer readouts.  Aggregate-only suites keep
# accepting v2 artifacts; requesting ``layers`` requires v3 and therefore turns
# an old checkpoint into an augmented one without retraining it.
CGA_LAYERWISE_ENCODER_EVAL_VERSION = 3


def _decoder_only_training_inputs(
    model_with_gradiend: Any,
    text: str,
    label: str,
) -> Optional[Dict[str, torch.Tensor]]:
    """Reproduce the standard decoder-only *training* objective for one row.

    ``TextTrainingDataset`` trains ordinary CLMs on the prefix before
    ``[MASK]`` and supervises the first token id of the target string.  The
    package's convenience ``create_inputs`` historically did neither: it kept
    the literal mask/suffix in the context, prepended a space to the target,
    and rejected targets that tokenize to multiple ids.  CGA detection must
    score the same gradient signal that its fitted direction averaged.
    """
    if not bool(getattr(model_with_gradiend, "is_decoder_only_model", False)):
        return None
    objective = getattr(
        getattr(model_with_gradiend, "base_model", None),
        "_gradiend_prediction_objective",
        None,
    )
    if objective in {"clm_mlm_head", "clm_sequence_cloze"}:
        return None
    masked = str(text)
    if "[MASK]" not in masked:
        return None
    prefix = masked.split("[MASK]", 1)[0]
    tokenizer = model_with_gradiend.tokenizer
    encoded = tokenizer(prefix, return_tensors="pt", truncation=True)
    input_ids = encoded["input_ids"]
    attention_mask = encoded.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
        encoded["attention_mask"] = attention_mask
    # The label id comes from the package's single resolver, under the protocol
    # the fitted direction's training dataset used (stamped on the model by its
    # trainer). Do not re-implement the rule here.
    from gradiend.trainer.text.prediction.dataset import training_label_token_id
    from gradiend.util.positions import assert_labels_on_real_tokens, last_real_token_positions

    target_id = training_label_token_id(
        tokenizer,
        label,
        getattr(model_with_gradiend, "label_token_protocol", "legacy"),
    )
    labels = torch.full_like(input_ids, -100)
    if attention_mask.sum() > 0:
        labels[torch.arange(input_ids.size(0)), last_real_token_positions(attention_mask, allow_empty=True)] = int(
            target_id
        )
    # Same item contract as the package's training dataset.
    assert_labels_on_real_tokens(labels, attention_mask, where="CGA detection item")
    encoded["labels"] = labels
    return dict(encoded)


def _score_one_row(
    model_with_gradiend: Any,
    direction: torch.Tensor,
    text: str,
    label: str,
    *,
    device: Any,
) -> float:
    """Build the item via ``create_inputs``, score via ``forward()``.

    Root-caused 2026-08-27 via device diagnostics on a real cluster run:
    every device-introspection field always reported the correct GPU (model,
    embeddings, ``gradiend.device_encoder``, the package's own
    ``_get_base_forward_device()``) -- the model was never the problem.
    ``create_inputs()`` tokenizes via ``self.tokenizer(text,
    return_tensors="pt")``, whose return type (``transformers.BatchEncoding``)
    is a ``UserDict`` subclass, **not** a ``dict`` subclass. The package's
    ``_move_batch_to_device`` checked ``isinstance(batch, dict)``, which is
    ``False`` for a ``BatchEncoding`` -- so ``create_inputs()``'s own device-
    placement call silently no-opped on every call, and ``input_ids`` stayed
    on whatever device the tokenizer put it on (always CPU) regardless of the
    model's device. Fixed in the package (``Mapping`` instead of ``dict``;
    ``/c/Git/gradiend``, needs its own sync to the cluster's package
    checkout, separate from this study repo's). This also explains why the
    fit itself never hit this: its dataset path builds tensors through a
    completely different route that never calls ``create_inputs`` at all.

    Worked around here too, defensively, so this does not have to wait on a
    second sync round-trip: ``create_inputs()``'s returned mapping is
    converted to a plain dict with every tensor explicitly moved, before
    ``forward()`` ever sees it (whose own internal placement call would
    otherwise hit the identical package bug on an unpatched checkout).
    """
    item = _decoder_only_training_inputs(
        model_with_gradiend, str(text), str(label)
    )
    if item is None:
        item = model_with_gradiend.create_inputs(str(text), str(label))
    # Target-at-position-0 rows (mask at the very start of the sentence, e.g.
    # race/religion's "[MASK] Cup.") have an empty prefix: nothing precedes the
    # target for the CLM to predict from, so there is no gradient signal. This
    # is legitimate data, not a row to filter. The other methods hit the same
    # rows through the package's signal extractor, which returns a zero vector
    # for them (``_mean_selected_tokens(allow_empty_rows=True)``: "a first-token
    # target simply has zero dL/dh"). Mirror that here -- an empty input carries
    # zero alignment, so score it 0.0 -- instead of letting a shape ``[.., 0]``
    # ``input_ids`` reach the model, where ``input_ids.view(-1, 0)`` raises an
    # uncaught RuntimeError that ``_score_rows_tolerant`` cannot swallow.
    _ids = item.get("input_ids") if isinstance(item, Mapping) else None
    if _ids is not None and torch.is_tensor(_ids) and _ids.shape[-1] == 0:
        del item
        return 0.0
    item = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in item.items()}
    streaming_cosine = getattr(model_with_gradiend, "forward_gradient_cosine", None)
    if callable(streaming_cosine):
        score = float(streaming_cosine(item, direction))
    else:
        # Compatibility for older package versions and lightweight test doubles.
        g = model_with_gradiend.forward(item, return_dict=False)
        score = cosine_to_direction(g, direction)
        del g
    del item
    return score


def _score_one_row_with_components(
    model_with_gradiend: Any,
    direction: torch.Tensor,
    component_slices: Mapping[str, Tuple[int, int]],
    text: str,
    label: str,
    *,
    device: Any,
) -> Tuple[float, Dict[str, float]]:
    """Extract one row gradient once, then score aggregate and layer slices."""
    item = _decoder_only_training_inputs(model_with_gradiend, str(text), str(label))
    if item is None:
        item = model_with_gradiend.create_inputs(str(text), str(label))
    ids = item.get("input_ids") if isinstance(item, Mapping) else None
    if ids is not None and torch.is_tensor(ids) and ids.shape[-1] == 0:
        return 0.0, {str(part): 0.0 for part in component_slices}
    item = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in item.items()}
    many_api = getattr(model_with_gradiend, "forward_gradient_cosines", None)
    if callable(many_api):
        aggregate, components = many_api(item, direction, component_slices)
        return float(aggregate), {
            str(part): float(value) for part, value in components.items()
        }
    if all(
        hasattr(model_with_gradiend, name)
        for name in (
            "exclusive_base_gradient_access",
            "_place_inputs_for_base_forward",
            "_get_base_forward_model",
            "_zero_base_grad",
        )
    ) and hasattr(getattr(model_with_gradiend, "gradiend", None), "_get_compiled_param_selectors"):
        return _streaming_gradient_cosines(
            model_with_gradiend, item, direction, component_slices
        )
    gradient = model_with_gradiend.forward(item, return_dict=False)
    flat = gradient.detach().reshape(-1)
    aggregate = cosine_to_direction(flat, direction)
    components = {
        str(part): cosine_to_direction(flat[int(lo):int(hi)], direction[int(lo):int(hi)])
        for part, (lo, hi) in component_slices.items()
    }
    del gradient, item
    return aggregate, components


def _streaming_gradient_cosines(
    model_with_gradiend: Any,
    inputs: Mapping[str, Any],
    direction: torch.Tensor,
    component_slices: Mapping[str, Tuple[int, int]],
) -> Tuple[float, Dict[str, float]]:
    """One-backward, bounded-memory aggregate + component gradient cosines.

    This mirrors the package's ``gradient_cosine_streaming`` hook strategy but
    accumulates every requested layer at the same time.  No flattened
    model-width gradient is ever materialized, which keeps layerwise CGA viable
    for large checkpoints as well as GPT-2/Pythia.
    """
    gm = model_with_gradiend.gradiend
    direction_flat = direction.detach().reshape(-1)
    slices = {
        str(part): (int(lo), int(hi))
        for part, (lo, hi) in component_slices.items()
    }
    selectors = [
        selector
        for selector in gm._get_compiled_param_selectors()
        if int(selector.num_selected) > 0
    ]
    expected = sum(int(selector.num_selected) for selector in selectors)
    if direction_flat.numel() != expected:
        raise ValueError(
            f"direction has {direction_flat.numel()} entries, mapped gradient has {expected}"
        )

    with model_with_gradiend.exclusive_base_gradient_access():
        placed = model_with_gradiend._place_inputs_for_base_forward(dict(inputs))
        placed = {
            key: value.unsqueeze(0)
            if torch.is_tensor(value) and value.ndim == 1
            else (
                value.squeeze(dim=1)
                if torch.is_tensor(value) and value.ndim == 3 and value.shape[1] == 1
                else value
            )
            for key, value in placed.items()
        }
        base_forward = model_with_gradiend._get_base_forward_model()
        if bool(getattr(model_with_gradiend, "use_seq2seq_encoder_mlm", False)):
            from gradiend.trainer.text.prediction.seq2seq import seq2seq_encoder_mlm_loss

            loss = seq2seq_encoder_mlm_loss(base_forward, placed)
        else:
            loss = base_forward(**placed).loss
        if loss is None:
            raise ValueError("Base model forward returned no loss for gradient cosine")
        model_with_gradiend._zero_base_grad(set_to_none=True)

        raw_model = base_forward.module if hasattr(base_forward, "module") else base_forward
        lookup: Dict[str, Any] = {}
        for name, param in raw_model.named_parameters():
            lookup[name] = param
            lookup.setdefault(name.removeprefix("module."), param)

        aggregate_dot = 0.0
        aggregate_g2 = 0.0
        norm_cache_key = (
            direction_flat.data_ptr(),
            int(getattr(direction_flat, "_version", 0)),
            direction_flat.numel(),
            str(direction_flat.device),
            str(direction_flat.dtype),
            tuple((part, lo, hi) for part, (lo, hi) in slices.items()),
        )
        norm_cache = getattr(gm, "_cga_layerwise_direction_norm_cache", None)
        if norm_cache and norm_cache[0] == norm_cache_key:
            aggregate_d2 = float(norm_cache[1])
            component_d2 = dict(norm_cache[2])
        else:
            direction_for_norm = direction_flat.float()
            aggregate_d2 = _safe_dot(direction_for_norm, direction_for_norm)
            component_d2 = {
                part: _safe_dot(
                    direction_for_norm[lo:hi], direction_for_norm[lo:hi]
                )
                for part, (lo, hi) in slices.items()
            }
            gm._cga_layerwise_direction_norm_cache = (
                norm_cache_key,
                aggregate_d2,
                dict(component_d2),
            )
        component_dot = {part: 0.0 for part in slices}
        component_g2 = {part: 0.0 for part in slices}
        handles = []
        mapped_params = []
        seen = set()
        offset = 0

        for selector in selectors:
            param = lookup.get(selector.name)
            if param is None:
                param = lookup.get(str(selector.name).removeprefix("module."))
            if param is None:
                raise KeyError(f"CGA mapped parameter {selector.name!r} not found")
            mapped_params.append(param)
            start = offset
            stop = start + int(selector.num_selected)
            offset = stop

            def _make_hook(param_selector: Any, lo_param: int, hi_param: int):
                def _hook(grad: torch.Tensor):
                    nonlocal aggregate_dot, aggregate_g2
                    signal = param_selector.select_from_param_grad(grad).reshape(-1).float()
                    block_direction = direction_flat[lo_param:hi_param]
                    if block_direction.device != signal.device:
                        block_direction = block_direction.to(signal.device)
                    block_direction = block_direction.float()
                    aggregate_dot += _safe_dot(signal, block_direction)
                    aggregate_g2 += _safe_dot(signal, signal)
                    for part, (lo, hi) in slices.items():
                        overlap_lo = max(lo, lo_param)
                        overlap_hi = min(hi, hi_param)
                        if overlap_hi <= overlap_lo:
                            continue
                        local_lo = overlap_lo - lo_param
                        local_hi = overlap_hi - lo_param
                        g_part = signal[local_lo:local_hi]
                        d_part = block_direction[local_lo:local_hi]
                        component_dot[part] += _safe_dot(g_part, d_part)
                        component_g2[part] += _safe_dot(g_part, g_part)
                    seen.add(param_selector.name)
                    return grad

                return _hook

            handles.append(param.register_hook(_make_hook(selector, start, stop)))
            if hasattr(param, "register_post_accumulate_grad_hook"):
                handles.append(
                    param.register_post_accumulate_grad_hook(
                        lambda parameter: setattr(parameter, "grad", None)
                    )
                )
        try:
            backward = getattr(model_with_gradiend, "_backward_through_base_model", None)
            backward(loss) if callable(backward) else loss.backward()
        finally:
            for handle in handles:
                handle.remove()
            for param in mapped_params:
                param.grad = None
        missing = [selector.name for selector in selectors if selector.name not in seen]
        if missing:
            raise RuntimeError(f"Gradients were not collected for parameters: {missing}")

    def _cos(dot: float, g2: float, d2: float) -> float:
        if not all(math.isfinite(x) for x in (dot, g2, d2)) or g2 <= 0.0 or d2 <= 0.0:
            return 0.0
        return float(dot / math.sqrt(g2 * d2))

    return _cos(aggregate_dot, aggregate_g2, aggregate_d2), {
        part: _cos(component_dot[part], component_g2[part], component_d2[part])
        for part in slices
    }


def _score_rows_tolerant(
    model_with_gradiend: Any,
    direction: torch.Tensor,
    items: Sequence[Tuple[str, str]],
    *,
    device: Any,
) -> Tuple["np.ndarray", "np.ndarray"]:
    """Score ``(masked_text, label)`` pairs with a defensive legacy fallback.

    Confirmed live on real cluster data (2026-08-27), not a neutral-only
    artifact: task vocabularies are not all curated to single-token completions
    the way the pronoun tasks are -- "handicaps" (`language`), "IMAM"
    (`religion`), "United States" (`ravel_country`) each split into 2+ GPT-2
    BPE tokens regardless of leading-space convention (checked directly:
    ``tok(' handicaps')`` -> 2 tokens either way). ``create_inputs`` requires
    exactly one, a real limitation of that convenience API this study's own
    training path does not share (the fit ran hundreds of rows through the
    same tasks without ever hitting this, via a different tokenization route).
    Standard decoder-only models now bypass that inconsistent convenience API
    and reproduce ``TextTrainingDataset`` exactly (prefix context plus the
    first target token), so multi-token labels are retained. The drop remains
    only for other/legacy objective implementations that still raise the old
    single-token error. Returns ``(scores, kept_mask)`` so parallel arrays stay
    aligned if that fallback is exercised.
    """
    import numpy as np

    scores: List[float] = []
    kept = np.zeros(len(items), dtype=bool)
    n_skipped = 0
    for i, (text, label) in enumerate(items):
        try:
            scores.append(_score_one_row(model_with_gradiend, direction, text, label, device=device))
            kept[i] = True
        except ValueError as exc:
            if exc.args and exc.args[0] == _SINGLE_LABEL_TOKEN_ERROR:
                n_skipped += 1
                continue
            raise
    if n_skipped:
        print(
            f"  cga eval: dropped {n_skipped}/{len(items)} rows whose label "
            "token split into >1 token on re-tokenization",
            flush=True,
        )
    return np.asarray(scores, dtype=np.float64), kept


def _score_rows_tolerant_with_components(
    model_with_gradiend: Any,
    direction: torch.Tensor,
    component_slices: Mapping[str, Tuple[int, int]],
    items: Sequence[Tuple[str, str]],
    *,
    device: Any,
) -> Tuple["np.ndarray", "np.ndarray", Dict[str, "np.ndarray"]]:
    """Multi-readout counterpart to :func:`_score_rows_tolerant`."""
    import numpy as np

    scores: List[float] = []
    component_scores: Dict[str, List[float]] = {
        str(part): [] for part in component_slices
    }
    kept = np.zeros(len(items), dtype=bool)
    n_skipped = 0
    for i, (text, label) in enumerate(items):
        try:
            score, parts = _score_one_row_with_components(
                model_with_gradiend,
                direction,
                component_slices,
                text,
                label,
                device=device,
            )
            scores.append(score)
            for part in component_scores:
                component_scores[part].append(float(parts[part]))
            kept[i] = True
        except ValueError as exc:
            if exc.args and exc.args[0] == _SINGLE_LABEL_TOKEN_ERROR:
                n_skipped += 1
                continue
            raise
    if n_skipped:
        print(
            f"  cga eval: dropped {n_skipped}/{len(items)} rows whose label "
            "token split into >1 token on re-tokenization",
            flush=True,
        )
    return (
        np.asarray(scores, dtype=np.float64),
        kept,
        {part: np.asarray(values, dtype=np.float64) for part, values in component_scores.items()},
    )


def score_labeled_rows_tolerant(
    model_with_gradiend: Any,
    direction: torch.Tensor,
    masked_texts: Sequence[str],
    label_tokens: Sequence[str],
) -> Tuple["np.ndarray", "np.ndarray"]:
    """Production row scorer returning ``(scores, kept_mask)``.

    Decoder-only rows use the exact training objective and retain multi-token
    target strings by supervising their first token, as the fitted CGA signal
    does. Unsupported legacy objectives still receive the old tolerant
    single-token-error fallback; ``kept_mask`` keeps class labels aligned.
    """
    if len(masked_texts) != len(label_tokens):
        raise ValueError(
            f"masked_texts ({len(masked_texts)}) and label_tokens "
            f"({len(label_tokens)}) must be the same length"
        )
    device = model_with_gradiend._get_base_forward_device()
    return _score_rows_tolerant(
        model_with_gradiend, direction, list(zip(masked_texts, label_tokens)), device=device
    )


def score_labeled_rows(
    model_with_gradiend: Any,
    direction: torch.Tensor,
    masked_texts: Sequence[str],
    label_tokens: Sequence[str],
) -> "np.ndarray":
    """``cos(g(x), Delta)`` for each ``(masked_text, label_token)`` pair.

    ``label_token`` is the row's own factual completion (the word that fills
    the mask), exactly what CAA fills the template with for its own scoring --
    there is no counterfactual available at eval time, so this is the only
    well-defined per-row gradient. One backward pass per row via
    :func:`_score_one_row`; nothing is batched or retained.

    Strict: any per-row failure raises. Kept as the direct, hard-guarantee
    primitive for callers with known-well-formed labels; production scoring
    (``fair_cga_encoder_eval``) uses :func:`score_labeled_rows_tolerant`
    instead, since real cluster data has disproven "labels are always
    single-token" -- see that function's docstring for the confirmed
    real-world cases (multi-word / multi-token class completions, not just
    auto-picked neutral fragments). Do not route production scoring back
    through this strict version, or that same class of failure crashes the
    whole task again.
    """
    import numpy as np

    if len(masked_texts) != len(label_tokens):
        raise ValueError(
            f"masked_texts ({len(masked_texts)}) and label_tokens "
            f"({len(label_tokens)}) must be the same length"
        )
    device = model_with_gradiend._get_base_forward_device()
    out = np.zeros(len(masked_texts), dtype=np.float64)
    for i, (text, label) in enumerate(zip(masked_texts, label_tokens)):
        if i == 0:
            print(
                f"  cga eval: device state before first row: "
                f"{_describe_device_state(model_with_gradiend)}",
                flush=True,
            )
        try:
            out[i] = _score_one_row(model_with_gradiend, direction, text, label, device=device)
        except Exception:
            print(
                f"  cga eval: device state at crash (row {i}): "
                f"{_describe_device_state(model_with_gradiend)}",
                flush=True,
            )
            raise
    return out


def neutral_masked_pairs(
    texts: Sequence[str],
    tokenizer: Any,
    is_decoder_only_model: bool,
    *,
    excluded_tokens: Optional[Sequence[str]] = None,
    seed: int = 0,
) -> List[Tuple[str, str]]:
    """``(masked_text, target_token)`` per neutral text, via the package's own
    ``create_masked_pair_from_text`` -- the same auto-mask convention its own
    ``_encode_neutral_dataset_rows`` uses for GRADIEND/ACTIEND. A neutral
    sentence has no factual/alternative fill, so a token to predict has to be
    picked from the sentence itself; texts that yield no valid pair (too
    short, all tokens excluded) are dropped, not padded.
    """
    import random as _random

    from gradiend.trainer.text.prediction.dataset import create_masked_pair_from_text

    mask_token = None if is_decoder_only_model else getattr(tokenizer, "mask_token", None)
    _random.seed(int(seed))
    pairs: List[Tuple[str, str]] = []
    for text in texts:
        pair = create_masked_pair_from_text(
            str(text),
            tokenizer,
            is_decoder_only_model,
            excluded_tokens=list(excluded_tokens or []),
            mask_token=mask_token,
            min_prefix_tokens=5,
        )
        if pair is not None:
            pairs.append(pair)
    return pairs


def score_neutral_texts(
    model_with_gradiend: Any,
    direction: torch.Tensor,
    texts: Sequence[str],
    *,
    tokenizer: Any,
    is_decoder_only_model: bool,
    excluded_tokens: Optional[Sequence[str]] = None,
    seed: int = 0,
) -> Tuple["np.ndarray", int]:
    """Cosine scores for free-text neutral rows; returns ``(scores, n_used)``.

    Neutral rows have no fixed completion word -- ``neutral_masked_pairs``
    auto-picks a token directly from the sentence's own BPE tokenization
    (``create_masked_pair_from_text``), which can legitimately be a sub-word
    fragment (e.g. ``'t`` from "don't") that does not round-trip through
    ``create_inputs``'s own re-tokenization (``f' {label}'``) as a single
    token. Uses :func:`_score_rows_tolerant`, so such a row is dropped rather
    than crashing -- the same policy :func:`score_labeled_rows_tolerant` now
    also applies to labeled rows, since real cluster data showed labeled
    completions can hit this too (see that function's docstring).
    """
    import numpy as np

    pairs = neutral_masked_pairs(
        texts,
        tokenizer,
        is_decoder_only_model,
        excluded_tokens=excluded_tokens,
        seed=seed,
    )
    if not pairs:
        return np.zeros(0, dtype=np.float64), 0
    device = model_with_gradiend._get_base_forward_device()
    scores, kept = _score_rows_tolerant(model_with_gradiend, direction, pairs, device=device)
    return scores, int(kept.sum())


def _describe_device_state(model_with_gradiend: Any) -> str:
    """One-line device/sharding snapshot for diagnosing the create_gradients /
    forward() device crash (2026-08-26/27) without guessing further.

    Two theories were each tried as a fix and both failed on real GPU runs:
    switching primitives (create_gradients -> forward()) and a defensive
    `.to()` resync. Both attempts were guesses without ground truth on what
    the model's actual device/sharding state is at the moment of the crash.
    This prints that ground truth instead, so the next run is diagnostic, not
    another blind attempt.
    """
    parts: List[str] = []

    def _add(label: str, fn):
        try:
            parts.append(f"{label}={fn()!r}")
        except Exception as exc:  # noqa: BLE001 - diagnostic only, must not itself crash
            parts.append(f"{label}=ERR({exc!r})")

    _add("base_model_is_sharded", lambda: getattr(model_with_gradiend, "base_model_is_sharded", "<missing>"))
    _add("base.device", lambda: getattr(model_with_gradiend.base_model, "device", "<missing>"))
    _add("hf_device_map", lambda: getattr(model_with_gradiend.base_model, "hf_device_map", "<missing>"))
    _add("first_param_device", lambda: next(model_with_gradiend.base_model.parameters()).device)
    _add("wte_weight_device", lambda: model_with_gradiend.base_model.get_input_embeddings().weight.device)
    _add("gradiend.device_encoder", lambda: model_with_gradiend.gradiend.device_encoder)
    _add(
        "get_base_forward_device",
        lambda: model_with_gradiend._get_base_forward_device(),
    )
    return " ".join(parts)


def _resync_base_model_device(model_with_gradiend: Any) -> None:
    """Defensive re-sync before scoring, kept as cheap insurance.

    Added after CGA's first cluster/GPU run (2026-08-26) crashed 3/3 trainers
    identically at the first row scored here, right after a fit that ran
    hundreds of successful forward/backward passes through the same base
    model. This call alone did **not** fix that crash on the second real run,
    and switching the scoring primitive from ``create_gradients()`` to
    ``create_inputs()`` + ``forward()`` (see that function's docstring) did
    **not** fix it either on the third -- identical crash, identical location,
    through the literal primitive the fit itself just used successfully. Root
    cause is genuinely open. Retained anyway: it is a no-op when the model is
    already correctly placed (the package's own ``.to()`` already skips
    moving a deliberately-sharded base model) and costs nothing measurable,
    so there is no reason to remove insurance for a cause that is still
    unidentified -- see ``score_labeled_rows``'s device diagnostics for the
    next actual data point.
    """
    # CGA may deliberately keep its inert full-width encoder on CPU to leave
    # room for unpruned paired gradients. Calling wrapper.to(device) here would
    # move that ~26 GiB tensor back to CUDA even though scoring never uses it.
    if bool(getattr(model_with_gradiend, "_cga_encoder_offloaded", False)):
        return
    device = getattr(getattr(model_with_gradiend, "gradiend", None), "device_encoder", None)
    if device is None:
        return
    to_fn = getattr(model_with_gradiend, "to", None)
    if callable(to_fn):
        to_fn(device)


def fair_cga_encoder_eval(
    model_with_gradiend: Any,
    direction: torch.Tensor,
    eval_df: "pd.DataFrame",
    neutral_df: "pd.DataFrame",
    *,
    target_classes: Sequence[str],
    class_encoding_direction: Optional[Mapping[str, float]] = None,
    max_size: Optional[int] = None,
    max_neutral: Optional[int] = None,
    n_bootstrap: int = 100,
    excluded_words: Optional[Sequence[str]] = None,
    eval_split: str = "test",
    val_split: str = "validation",
    seed: int = 0,
    score_labeled_fn: Optional[Any] = None,
    score_neutral_fn: Optional[Any] = None,
    component_slices: Optional[Mapping[str, Tuple[int, int]]] = None,
    backend_name: str = "cga",
    prior_readouts: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """CGA's encoder-evaluation counterpart to ``caa_eval.encode_caa_scores``.

    Val-fit / test-report Youden threshold, one ``class_vs_neutral_metrics``
    call per class -- structurally identical to how CAA scores its own
    directions, differing only in how a per-row score is produced (a
    gradient's cosine to ``direction`` instead of an activation's cosine to a
    CAA vector). Returns the same shape ``study/stages/train.py`` already reads
    off ``_fair_encoder_eval`` (``encoder_metrics`` / ``readout_metrics`` /
    ``per_class_readouts``), so no downstream row-emission code needs to know
    CGA's evaluation path is different from GRADIEND/ACTIEND's.

    ``prior_readouts`` (test-only migration): the ``per_class_readouts`` /
    ``per_component_readouts`` of an earlier eval of the *same* direction and
    scoring version. When every (component, class) already carries a usable
    ``val_readout`` the validation split is **not scored at all**: the stored
    validation readout is deterministic given the direction, so its frozen
    Spec_n/Excl rules and its ``val_readout`` (which the site/layer lock ranks
    on) are reused verbatim and only the test split is scored. If any
    (component, class) lacks one, the whole validation pass runs as usual.
    """
    import numpy as np
    from caa_eval import _subset_df
    from gradiend.model.utils import is_decoder_only_model as is_decoder_only_model_from_obj
    from neutral_protocol import texts_for_split
    from sae_eval import (
        class_vs_neutral_metrics,
        frozen_decision_rules,
        require_frozen_rules,
    )

    empty = {
        "encoder_metrics": {},
        "readout_metrics": {},
        "per_class_readouts": {},
        "per_component_readouts": {},
        "component_keys": [],
    }
    classes = [str(c) for c in target_classes]
    if eval_df is None or eval_df.empty or "label_class" not in eval_df.columns:
        return {**empty, "error": "empty or unlabeled eval_df"}

    tokenizer = model_with_gradiend.tokenizer
    is_decoder_only = is_decoder_only_model_from_obj(tokenizer)
    print(f"  cga eval: device state before resync: {_describe_device_state(model_with_gradiend)}", flush=True)
    _resync_base_model_device(model_with_gradiend)
    print(f"  cga eval: device state after resync:  {_describe_device_state(model_with_gradiend)}", flush=True)

    eval_split_df = _subset_df(eval_df, eval_split, max_size, label_col="label_class")
    val_split_df = _subset_df(eval_df, val_split, max_size, label_col="label_class")
    neu_test = texts_for_split(neutral_df, eval_split, max_rows=max_neutral, fallback="all")
    neu_val = texts_for_split(neutral_df, val_split, max_rows=max_neutral, fallback="all")
    if eval_split_df.empty or not neu_test:
        return {**empty, "error": "empty eval or neutral set"}

    present = set(eval_split_df["label_class"].astype(str))
    classes = [c for c in classes if c in present]
    if not classes:
        return {**empty, "error": f"no target classes present (had {list(target_classes)})"}

    def _sign_for(cls: str) -> float:
        if class_encoding_direction and cls in class_encoding_direction:
            return 1.0 if float(class_encoding_direction[cls]) >= 0 else -1.0
        return 1.0

    slices = {
        str(part): (int(bounds[0]), int(bounds[1]))
        for part, bounds in (component_slices or {}).items()
    }
    direction_size = int(direction.detach().numel())
    for part, (lo, hi) in slices.items():
        if lo < 0 or hi <= lo or hi > direction_size:
            raise ValueError(
                f"invalid {part} slice [{lo}:{hi}] for direction dim {direction_size}"
            )

    def _score_split(df: "pd.DataFrame", neu_texts: Sequence[str]):
        if df is None or df.empty or not neu_texts:
            return None
        labels_arr = df["label_class"].astype(str).to_numpy()
        # Real task vocabularies include multi-token completions. Decoder-only
        # CGA scores them with the same prefix + first-target-token objective as
        # its fit; the tolerant wrapper remains for other legacy objectives.
        # Pluggable per-row scorer: default is CGA's weight-gradient cosine via
        # model.forward; CAGA passes an ActivationSignalExtractor-based dL/dh scorer
        # (fair_caga_encoder_eval). Everything else -- val-fit/test-report Youden,
        # class_vs_neutral_metrics -- is shared so the two are comparable.
        if score_labeled_fn is not None:
            labeled_result = score_labeled_fn(
                df["masked"].tolist(), df["label"].tolist()
            )
            raw, kept = labeled_result[:2]
            labeled_parts = labeled_result[2] if len(labeled_result) >= 3 else {}
        elif slices:
            raw, kept, labeled_parts = _score_rows_tolerant_with_components(
                model_with_gradiend,
                direction,
                slices,
                list(zip(df["masked"].tolist(), df["label"].tolist())),
                device=model_with_gradiend._get_base_forward_device(),
            )
        else:
            raw, kept = score_labeled_rows_tolerant(
                model_with_gradiend, direction, df["masked"].tolist(), df["label"].tolist()
            )
            labeled_parts = {}
        labels_arr = labels_arr[kept]
        if score_neutral_fn is not None:
            neutral_result = score_neutral_fn(list(neu_texts))
            neu_scores, n_neu_used = neutral_result[:2]
            neutral_parts = neutral_result[2] if len(neutral_result) >= 3 else {}
        elif slices:
            pairs = neutral_masked_pairs(
                neu_texts,
                tokenizer,
                is_decoder_only,
                excluded_tokens=excluded_words,
                seed=seed,
            )
            if pairs:
                neu_scores, neu_kept, neutral_parts = _score_rows_tolerant_with_components(
                    model_with_gradiend,
                    direction,
                    slices,
                    pairs,
                    device=model_with_gradiend._get_base_forward_device(),
                )
                n_neu_used = int(neu_kept.sum())
            else:
                neu_scores = np.zeros(0, dtype=np.float64)
                n_neu_used = 0
                neutral_parts = {part: np.zeros(0, dtype=np.float64) for part in slices}
        else:
            neu_scores, n_neu_used = score_neutral_texts(
                model_with_gradiend,
                direction,
                neu_texts,
                tokenizer=tokenizer,
                is_decoder_only_model=is_decoder_only,
                excluded_tokens=excluded_words,
                seed=seed,
            )
            neutral_parts = {}
        return raw, labels_arr, neu_scores, n_neu_used, labeled_parts, neutral_parts

    def _prior_val_readout(
        part: Optional[str], cls: str
    ) -> Optional[Mapping[str, Any]]:
        if not prior_readouts:
            return None
        if part is None:
            blob = (prior_readouts.get("per_class_readouts") or {}).get(cls)
        else:
            blob = (
                (prior_readouts.get("per_component_readouts") or {}).get(part) or {}
            ).get(cls)
        vr = blob.get("val_readout") if isinstance(blob, Mapping) else None
        # Usable = carries a frozen neutral rule; anything else must be rescored.
        return vr if isinstance(vr, Mapping) and frozen_decision_rules(vr) else None

    use_prior_val = bool(prior_readouts) and all(
        _prior_val_readout(part, cls) is not None
        for part in (None, *slices)
        for cls in classes
    )
    test_pack = _score_split(eval_split_df, neu_test)
    if test_pack is None:
        return {**empty, "error": "empty eval or neutral set"}
    if use_prior_val:
        val_pack = None
        print(
            f"  {backend_name} eval: reusing prior validation readouts for "
            f"{len(classes)} class(es) x {1 + len(slices)} component(s); "
            "validation split not scored",
            flush=True,
        )
    else:
        val_pack = _score_split(val_split_df, neu_val)
    raw_test, labels_test, neu_test_scores, n_neu_test, test_parts, neu_test_parts = test_pack

    def _readouts(
        test_scores: "np.ndarray",
        test_neutral: "np.ndarray",
        val_scores: Optional["np.ndarray"],
        val_neutral: Optional["np.ndarray"],
        *,
        component_part: Optional[str] = None,
    ) -> Dict[str, Any]:
        per_class: Dict[str, Any] = {}
        labels_val = val_pack[1] if val_pack is not None else None
        for cls in classes:
            sign = _sign_for(cls)
            pos_test = sign * test_scores[labels_test == cls]
            other_test = {
                other: sign * test_scores[labels_test == other]
                for other in classes
                if other != cls
            }
            tau = None
            frozen: Dict[str, Any] = {}
            val_readout = None
            prior_vr = _prior_val_readout(component_part, cls) if use_prior_val else None
            if prior_vr is not None:
                tau = prior_vr.get("youden_threshold")
                frozen = frozen_decision_rules(prior_vr)
                val_readout = dict(prior_vr)
            elif (
                val_scores is not None
                and val_neutral is not None
                and labels_val is not None
            ):
                pos_val = sign * val_scores[labels_val == cls]
                if pos_val.size and val_neutral.size:
                    other_val = {
                        other: sign * val_scores[labels_val == other]
                        for other in classes
                        if other != cls
                    }
                    val_other = next(iter(other_val), None)
                    v_rd = class_vs_neutral_metrics(
                        pos_val,
                        sign * val_neutral,
                        target_class=cls,
                        scores_other=other_val.get(val_other) if val_other else None,
                        other_class=val_other,
                        scores_other_by_class=other_val,
                        n_bootstrap=0,
                    )
                    tau = v_rd.get("youden_threshold")
                    # Frozen Spec_n / Excl rules (threshold + sign) for the test
                    # report; ``tau`` is only the pooled diagnostic.
                    frozen = frozen_decision_rules(v_rd)
                    val_readout = dict(v_rd)
            other = next(iter(other_test), None)
            rd = class_vs_neutral_metrics(
                pos_test,
                sign * test_neutral,
                target_class=cls,
                scores_other=other_test.get(other) if other else None,
                other_class=other,
                scores_other_by_class=other_test,
                n_bootstrap=n_bootstrap,
                youden_threshold=tau,
                **frozen,
                extras={
                    "backend": str(backend_name),
                    "score": "cosine",
                    "component_part": component_part,
                },
            )
            rd["score"] = "cosine"
            rd["val_readout"] = val_readout
            rd["val_readout_source"] = "prior" if prior_vr is not None else "scored"
            if val_readout is not None:
                from suitability import encoding_e

                rd["val_encoding_E"] = encoding_e(val_readout)
            rd["n_neutral_texts"] = int(n_neu_test)
            if component_part is not None:
                rd["component_part"] = str(component_part)
            per_class[cls] = rd
        return per_class

    raw_val = val_pack[0] if val_pack is not None else None
    neu_val_scores = val_pack[2] if val_pack is not None else None
    per_class = _readouts(raw_test, neu_test_scores, raw_val, neu_val_scores)
    per_component: Dict[str, Dict[str, Any]] = {}
    for part in slices:
        if part not in test_parts or part not in neu_test_parts:
            raise ValueError(f"scorer did not return requested component {part!r}")
        val_parts = val_pack[4] if val_pack is not None else {}
        neu_val_parts = val_pack[5] if val_pack is not None else {}
        per_component[part] = _readouts(
            test_parts[part],
            neu_test_parts[part],
            val_parts.get(part),
            neu_val_parts.get(part),
            component_part=part,
        )

    # A class without a usable validation readout would silently get a test-fit
    # (oracle) Spec_n/Excl above; that is an error, not a fallback.
    require_frozen_rules(per_class, context=f"{backend_name} encoder eval (test)")
    for _part, _comp in per_component.items():
        require_frozen_rules(
            _comp, context=f"{backend_name} encoder eval component {_part!r} (test)"
        )

    numeric_label = np.array(
        [_sign_for(cls) if cls in classes else 0.0 for cls in labels_test], dtype=np.float64
    )
    in_classes = np.isin(labels_test, classes)
    corr = None
    if in_classes.sum() >= 2:
        raw_sub = raw_test[in_classes]
        lab_sub = numeric_label[in_classes]
        if float(np.std(raw_sub)) > 0.0 and float(np.std(lab_sub)) > 0.0:
            corr = float(np.corrcoef(raw_sub, lab_sub)[0, 1])
    mean_by_class = {
        cls: float(np.mean(raw_test[labels_test == cls])) for cls in classes
        if int((labels_test == cls).sum()) > 0
    }

    return {
        "encoder_metrics": {"correlation": corr, "mean_by_class": mean_by_class},
        "readout_metrics": {},
        "per_class_readouts": per_class,
        "per_component_readouts": per_component,
        "component_keys": list(slices),
    }
