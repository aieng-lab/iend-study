"""Wall-clock + CUDA peak-memory cost timers for study runs.

Records land in ``results.json`` → ``raw.cost`` (per-stage) and
``raw.cost_summary`` (aggregates). Prefer passing ``phase=`` so summaries
group cleanly even when timer names vary.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence


# Canonical phases for ``cost_summary`` (order used in reports).
COST_PHASES: tuple = (
    "data",
    "train",
    "feature_select",
    "encode",
    "causal",
    "localization",
    "metrics",
    "other",
)


@dataclass
class CostRecord:
    name: str
    seconds: float
    peak_cuda_bytes: Optional[int] = None
    peak_cuda_reserved_bytes: Optional[int] = None
    allocated_before_bytes: Optional[int] = None
    allocated_after_bytes: Optional[int] = None
    peak_delta_bytes: Optional[int] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def phase(self) -> str:
        p = self.meta.get("phase")
        if isinstance(p, str) and p.strip():
            return str(p).strip()
        return _infer_phase(self.name)

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "name": self.name,
            "phase": self.phase,
            "seconds": float(self.seconds),
            **{k: v for k, v in self.meta.items() if k != "phase"},
        }
        if self.peak_cuda_bytes is not None:
            out["peak_cuda_bytes"] = int(self.peak_cuda_bytes)
            out["peak_cuda_gb"] = self.peak_cuda_bytes / (1024**3)
        if self.peak_cuda_reserved_bytes is not None:
            out["peak_cuda_reserved_bytes"] = int(self.peak_cuda_reserved_bytes)
            out["peak_cuda_reserved_gb"] = self.peak_cuda_reserved_bytes / (1024**3)
        if self.allocated_before_bytes is not None:
            out["allocated_before_bytes"] = int(self.allocated_before_bytes)
            out["allocated_before_gb"] = self.allocated_before_bytes / (1024**3)
        if self.allocated_after_bytes is not None:
            out["allocated_after_bytes"] = int(self.allocated_after_bytes)
            out["allocated_after_gb"] = self.allocated_after_bytes / (1024**3)
        if self.peak_delta_bytes is not None:
            out["peak_delta_bytes"] = int(self.peak_delta_bytes)
            out["peak_delta_gb"] = self.peak_delta_bytes / (1024**3)
        return out


_RECORDS: List[CostRecord] = []
_RUN_META: Dict[str, Any] = {}


def reset_cost_records() -> None:
    _RECORDS.clear()
    _RUN_META.clear()


def get_cost_records() -> List[Dict[str, Any]]:
    # Copy immutable run provenance onto every row.  ``results.json`` can be
    # incrementally refreshed by method family; a durable per-method ledger
    # therefore cannot rely on the one run-level summary that happens to be
    # current when the file is read later.
    run_meta = dict(_RUN_META)
    return [
        {
            **r.to_dict(),
            **({"run": run_meta} if run_meta else {}),
        }
        for r in _RECORDS
    ]


def _ledger_owner(record: Mapping[str, Any]) -> Optional[str]:
    """Map timer variants to their method family for durable cost retention."""
    backend = str(record.get("backend") or "").strip().lower()
    if not backend:
        return None
    for family in ("gradiend", "actiend", "agiend", "caga", "cga", "caa", "sae"):
        if backend == family or backend.startswith(f"{family}_"):
            return family
    return None


def merge_cost_ledger(
    previous: Sequence[Mapping[str, Any]] | None,
    current: Sequence[Mapping[str, Any]] | None,
) -> List[Dict[str, Any]]:
    """Retain one current timer batch per method family across refreshes.

    ``raw.cost`` intentionally describes only the active invocation.  The
    separate ``raw.cost_ledger`` keeps the most recent complete batch for each
    method family, so a later ``METHODS=gradiend`` refresh cannot erase CAA or
    SAE timing/memory evidence.  Unattributed process-wide timers are excluded
    because assigning them to a family would make the ledger misleading.
    """
    old = [dict(row) for row in (previous or []) if isinstance(row, Mapping)]
    new = [dict(row) for row in (current or []) if isinstance(row, Mapping)]
    refreshed = {_ledger_owner(row) for row in new}
    refreshed.discard(None)
    retained = [row for row in old if _ledger_owner(row) not in refreshed]
    return retained + [row for row in new if _ledger_owner(row) is not None]


def set_run_compute_meta(**meta: Any) -> None:
    """Attach run-level device / model / workload facts to ``cost_summary``."""
    for k, v in meta.items():
        if v is not None:
            _RUN_META[str(k)] = v


def record_cost(
    name: str,
    seconds: float,
    *,
    peak_cuda_bytes: Optional[int] = None,
    phase: Optional[str] = None,
    **meta: Any,
) -> None:
    """Append a synthetic / cache-hit cost row without running a block."""
    if phase is not None:
        meta = {**meta, "phase": phase}
    _RECORDS.append(
        CostRecord(
            name=name,
            seconds=float(seconds),
            peak_cuda_bytes=peak_cuda_bytes,
            meta=dict(meta),
        )
    )


def _infer_phase(name: str) -> str:
    n = str(name).lower()
    if n.startswith("train:") or n == "train":
        return "train"
    if n.startswith("sae_select") or n.startswith("caa_fit") or "feature_select" in n:
        return "feature_select"
    if (
        n.startswith("encoder_eval")
        or n.startswith("sae_encode")
        or n.startswith("caa_encode")
        or n.startswith("encode")
    ):
        return "encode"
    if n.startswith("causal"):
        return "causal"
    if n.startswith("localization"):
        return "localization"
    if n.startswith("encoder_component") or n.startswith("sae_metrics") or n.startswith("metrics"):
        return "metrics"
    if n.startswith("data") or n.startswith("build_"):
        return "data"
    if n.startswith("sae_identify"):
        return "feature_select"
    if n.startswith("caa_fit_encode"):
        return "feature_select"
    return "other"


def _cuda_snapshot() -> Dict[str, Any]:
    out: Dict[str, Any] = {"cuda_available": False}
    try:
        import torch

        out["cuda_available"] = bool(torch.cuda.is_available())
        if not out["cuda_available"]:
            out["device"] = "cpu"
            return out
        idx = int(torch.cuda.current_device())
        props = torch.cuda.get_device_properties(idx)
        out.update(
            {
                "device": f"cuda:{idx}",
                "gpu_name": getattr(props, "name", None),
                "gpu_total_memory_bytes": int(getattr(props, "total_memory", 0) or 0),
                "gpu_total_memory_gb": float(getattr(props, "total_memory", 0) or 0) / (1024**3),
                "gpu_capability": (
                    f"{props.major}.{props.minor}"
                    if hasattr(props, "major")
                    else None
                ),
                "gpu_count": int(torch.cuda.device_count()),
            }
        )
    except Exception as exc:
        out["device_error"] = str(exc)
    return out


def snapshot_device() -> Dict[str, Any]:
    """Public helper: current CUDA/CPU device facts."""
    return _cuda_snapshot()




def flops_proxy_forward(*, n_params: Optional[int], n_tokens: Optional[int]) -> Optional[float]:
    """Rough LM forward FLOPs ≈ ``2 * n_params * n_tokens`` (Kaplan-style proxy)."""
    if n_params is None or n_tokens is None:
        return None
    if n_params <= 0 or n_tokens <= 0:
        return None
    return float(2.0 * float(n_params) * float(n_tokens))


@contextmanager
def cost_timer(name: str, **meta: Any) -> Iterator[None]:
    """Record wall time; if CUDA is available, also peak allocated / reserved bytes."""
    section = CostSection(name, **meta)
    section.start()
    try:
        yield
    finally:
        section.stop()


class CostSection:
    """Manual start/stop timer (avoids reindenting large blocks under ``with``)."""

    def __init__(self, name: str, **meta: Any) -> None:
        self.name = name
        self.meta = dict(meta)
        self._t0: Optional[float] = None
        self._alloc_before: Optional[int] = None

    def start(self) -> "CostSection":
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                self._alloc_before = int(torch.cuda.memory_allocated())
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            self._alloc_before = None
        self._t0 = time.perf_counter()
        return self

    def stop(self) -> None:
        peak = None
        peak_reserved = None
        alloc_after = None
        peak_delta = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                peak = int(torch.cuda.max_memory_allocated())
                try:
                    peak_reserved = int(torch.cuda.max_memory_reserved())
                except Exception:
                    peak_reserved = None
                alloc_after = int(torch.cuda.memory_allocated())
                if self._alloc_before is not None and peak is not None:
                    peak_delta = max(0, int(peak) - int(self._alloc_before))
        except Exception:
            peak = None
        dt = time.perf_counter() - float(self._t0 or time.perf_counter())
        _RECORDS.append(
            CostRecord(
                name=self.name,
                seconds=float(dt),
                peak_cuda_bytes=peak,
                peak_cuda_reserved_bytes=peak_reserved,
                allocated_before_bytes=self._alloc_before,
                allocated_after_bytes=alloc_after,
                peak_delta_bytes=peak_delta,
                meta=dict(self.meta),
            )
        )
        peak_txt = ""
        if peak is not None:
            peak_txt = f" peak_cuda={peak / 1024**3:.2f}GB"
            if peak_delta is not None:
                peak_txt += f" (+{peak_delta / 1024**3:.2f}GB)"
        print(f"  [cost] {self.name}: {dt:.2f}s{peak_txt}", flush=True)


def _max_optional(vals: Iterable[Optional[float]]) -> Optional[float]:
    nums = [float(v) for v in vals if v is not None]
    return max(nums) if nums else None


def _sum_seconds(recs: Sequence[CostRecord]) -> float:
    return float(sum(float(r.seconds) for r in recs))


def summarize_cost_records(
    records: Optional[Sequence[Mapping[str, Any] | CostRecord]] = None,
    *,
    run_meta: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Aggregate per-stage cost into headline compute metrics.

    Peak GPU fields:
    - ``peak_gpu_*_gb``: max absolute ``peak_cuda_gb`` in that phase (includes resident
      tensors from earlier stages — useful as "high-water mark while doing X").
    - ``peak_delta_*_gb``: max within-block growth vs allocated-at-entry (cleaner
      attribution of what the stage itself allocated).
    """
    if records is None:
        recs = list(_RECORDS)
        dicts = [r.to_dict() for r in recs]
    else:
        dicts = []
        recs = []
        for r in records:
            if isinstance(r, CostRecord):
                recs.append(r)
                dicts.append(r.to_dict())
            else:
                d = dict(r)
                dicts.append(d)
                meta = {
                    k: v
                    for k, v in d.items()
                    if k
                    not in {
                        "name",
                        "seconds",
                        "phase",
                        "peak_cuda_bytes",
                        "peak_cuda_gb",
                        "peak_cuda_reserved_bytes",
                        "peak_cuda_reserved_gb",
                        "allocated_before_bytes",
                        "allocated_before_gb",
                        "allocated_after_bytes",
                        "allocated_after_gb",
                        "peak_delta_bytes",
                        "peak_delta_gb",
                    }
                }
                if d.get("phase"):
                    meta["phase"] = d["phase"]
                recs.append(
                    CostRecord(
                        name=str(d.get("name") or ""),
                        seconds=float(d.get("seconds") or 0.0),
                        peak_cuda_bytes=d.get("peak_cuda_bytes"),
                        peak_cuda_reserved_bytes=d.get("peak_cuda_reserved_bytes"),
                        allocated_before_bytes=d.get("allocated_before_bytes"),
                        allocated_after_bytes=d.get("allocated_after_bytes"),
                        peak_delta_bytes=d.get("peak_delta_bytes"),
                        meta=meta,
                    )
                )

    by_phase: Dict[str, List[CostRecord]] = {p: [] for p in COST_PHASES}
    for r in recs:
        phase = r.phase if r.phase in by_phase else "other"
        by_phase.setdefault(phase, []).append(r)

    phase_seconds = {p: _sum_seconds(rs) for p, rs in by_phase.items()}
    phase_peak_gb = {
        p: _max_optional(
            (r.peak_cuda_bytes / (1024**3)) if r.peak_cuda_bytes is not None else None
            for r in rs
        )
        for p, rs in by_phase.items()
    }
    phase_delta_gb = {
        p: _max_optional(
            (r.peak_delta_bytes / (1024**3)) if r.peak_delta_bytes is not None else None
            for r in rs
        )
        for p, rs in by_phase.items()
    }
    phase_reserved_gb = {
        p: _max_optional(
            (r.peak_cuda_reserved_bytes / (1024**3))
            if r.peak_cuda_reserved_bytes is not None
            else None
            for r in rs
        )
        for p, rs in by_phase.items()
    }

    wall_total = _sum_seconds(recs)
    overall_peak = _max_optional(
        (r.peak_cuda_bytes / (1024**3)) if r.peak_cuda_bytes is not None else None
        for r in recs
    )

    summary: Dict[str, Any] = {
        "wall_total_s": wall_total,
        "wall_train_s": phase_seconds.get("train", 0.0),
        "wall_feature_select_s": phase_seconds.get("feature_select", 0.0),
        "wall_encode_s": phase_seconds.get("encode", 0.0),
        "wall_causal_s": phase_seconds.get("causal", 0.0),
        "wall_localization_s": phase_seconds.get("localization", 0.0),
        "wall_data_s": phase_seconds.get("data", 0.0),
        "wall_metrics_s": phase_seconds.get("metrics", 0.0),
        "peak_gpu_train_gb": phase_peak_gb.get("train"),
        "peak_gpu_feature_select_gb": phase_peak_gb.get("feature_select"),
        "peak_gpu_encode_gb": phase_peak_gb.get("encode"),
        "peak_gpu_causal_gb": phase_peak_gb.get("causal"),
        "peak_gpu_overall_gb": overall_peak,
        "peak_delta_train_gb": phase_delta_gb.get("train"),
        "peak_delta_feature_select_gb": phase_delta_gb.get("feature_select"),
        "peak_delta_encode_gb": phase_delta_gb.get("encode"),
        "peak_delta_causal_gb": phase_delta_gb.get("causal"),
        "peak_reserved_train_gb": phase_reserved_gb.get("train"),
        "peak_reserved_encode_gb": phase_reserved_gb.get("encode"),
        "by_phase": {
            p: {
                "seconds": phase_seconds.get(p, 0.0),
                "peak_cuda_gb": phase_peak_gb.get(p),
                "peak_delta_gb": phase_delta_gb.get(p),
                "peak_reserved_gb": phase_reserved_gb.get(p),
                "n_timers": len(by_phase.get(p) or []),
            }
            for p in COST_PHASES
            if (by_phase.get(p) or [])
        },
        "n_timers": len(recs),
        "cache_hits": sum(1 for r in recs if r.meta.get("cache_hit")),
    }

    causal_backend: Dict[str, List[CostRecord]] = {}
    for r in recs:
        if r.phase != "causal":
            continue
        backend = r.meta.get("backend")
        if not backend:
            name = str(r.name)
            if name.startswith("causal:") and ":" in name[7:]:
                backend = name.split(":")[1]
            elif name.startswith("causal:"):
                backend = name.split(":", 1)[1] or "all"
            else:
                backend = "all"
        causal_backend.setdefault(str(backend), []).append(r)
    if causal_backend:
        summary["causal_by_backend"] = {
            b: {
                "seconds": _sum_seconds(rs),
                "peak_cuda_gb": _max_optional(
                    (r.peak_cuda_bytes / (1024**3)) if r.peak_cuda_bytes is not None else None
                    for r in rs
                ),
                "peak_delta_gb": _max_optional(
                    (r.peak_delta_bytes / (1024**3)) if r.peak_delta_bytes is not None else None
                    for r in rs
                ),
                "n_timers": len(rs),
            }
            for b, rs in sorted(causal_backend.items())
        }

    meta = dict(_RUN_META)
    if run_meta:
        meta.update(dict(run_meta))
    if meta:
        summary["run"] = meta

    n_params = meta.get("n_params")
    train_tokens = meta.get("train_tokens")
    encode_tokens = meta.get("encode_tokens")
    if n_params and train_tokens:
        fwd = flops_proxy_forward(n_params=int(n_params), n_tokens=int(train_tokens))
        if fwd is not None:
            summary["flops_proxy_train"] = float(fwd) * 3.0
            summary["flops_proxy_train_note"] = "3 * 2 * n_params * train_tokens"
    if n_params and encode_tokens:
        fwd = flops_proxy_forward(n_params=int(n_params), n_tokens=int(encode_tokens))
        if fwd is not None:
            summary["flops_proxy_encode"] = float(fwd)
            summary["flops_proxy_encode_note"] = "2 * n_params * encode_tokens"

    summary["stages"] = dicts
    return summary


def format_cost_summary_md(summary: Mapping[str, Any]) -> str:
    """Markdown section for REPORT.md."""
    lines = [
        "## Compute",
        "",
        "Wall-clock and CUDA peak memory by pipeline phase. "
        "`peak_gpu_*` is the absolute high-water mark while that phase ran "
        "(may include tensors still resident from earlier stages). "
        "`peak_delta_*` is growth vs allocated-at-entry for that phase.",
        "",
        "| phase | wall_s | peak_gpu_gb | peak_delta_gb |",
        "| --- | ---: | ---: | ---: |",
    ]
    rows = [
        ("train", "wall_train_s", "peak_gpu_train_gb", "peak_delta_train_gb"),
        (
            "feature_select",
            "wall_feature_select_s",
            "peak_gpu_feature_select_gb",
            "peak_delta_feature_select_gb",
        ),
        ("encode", "wall_encode_s", "peak_gpu_encode_gb", "peak_delta_encode_gb"),
        ("causal", "wall_causal_s", "peak_gpu_causal_gb", "peak_delta_causal_gb"),
        ("localization", "wall_localization_s", None, None),
        ("total", "wall_total_s", "peak_gpu_overall_gb", None),
    ]
    for label, sk, pk, dk in rows:
        s = summary.get(sk)
        p = summary.get(pk) if pk else None
        d = summary.get(dk) if dk else None
        s_txt = f"{float(s):.2f}" if isinstance(s, (int, float)) else "—"
        p_txt = f"{float(p):.2f}" if isinstance(p, (int, float)) else "—"
        d_txt = f"{float(d):.2f}" if isinstance(d, (int, float)) else "—"
        lines.append(f"| {label} | {s_txt} | {p_txt} | {d_txt} |")

    by_backend = summary.get("causal_by_backend") or {}
    if by_backend:
        lines.extend(
            [
                "",
                "### Causal by backend",
                "",
                "| backend | wall_s | peak_gpu_gb | peak_delta_gb |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for backend, blob in sorted(by_backend.items()):
            s = blob.get("seconds")
            p = blob.get("peak_cuda_gb")
            d = blob.get("peak_delta_gb")
            s_txt = f"{float(s):.2f}" if isinstance(s, (int, float)) else "—"
            p_txt = f"{float(p):.2f}" if isinstance(p, (int, float)) else "—"
            d_txt = f"{float(d):.2f}" if isinstance(d, (int, float)) else "—"
            lines.append(f"| {backend} | {s_txt} | {p_txt} | {d_txt} |")

    disk = summary.get("disk") or {}
    if disk:
        lines.extend(["", "### Disk", ""])
        for k in (
            "base_model_bytes_est",
            "base_model_gb_est",
            "modified_models_bytes",
            "modified_models_gb",
            "artifacts_bytes",
            "artifacts_gb",
            "run_dir_bytes",
            "run_dir_gb",
            "modified_over_base",
            "bytes_per_param_assumed",
        ):
            if k in disk and disk[k] is not None:
                lines.append(f"- **{k}**: `{disk[k]}`")

    run = summary.get("run") or {}
    if run:
        lines.extend(["", "### Device / workload", ""])
        for k in (
            "device",
            "gpu_name",
            "gpu_total_memory_gb",
            "n_params",
            "n_params_trainable",
            "hf_model",
            "max_steps",
            "train_batch_size",
            "train_texts",
            "train_tokens",
            "encode_tokens",
            "sae_layers",
            "causal_n_per_group",
            "causal_strengths",
        ):
            if k in run and run[k] is not None:
                lines.append(f"- **{k}**: `{run[k]}`")
        if summary.get("flops_proxy_train") is not None:
            lines.append(
                f"- **flops_proxy_train**: `{summary['flops_proxy_train']:.3e}` "
                f"({summary.get('flops_proxy_train_note')})"
            )
        if summary.get("flops_proxy_encode") is not None:
            lines.append(
                f"- **flops_proxy_encode**: `{summary['flops_proxy_encode']:.3e}` "
                f"({summary.get('flops_proxy_encode_note')})"
            )
    lines.append("")
    return "\n".join(lines)




