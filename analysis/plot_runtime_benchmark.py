"""Render appendix figures from one fixed-hardware runtime benchmark.

The input is a ``results.json`` with ``raw.cost_ledger`` rows.  Runtime is
additive within a method-stage, whereas GPU memory is a peak: this distinction
is deliberately encoded in the two figures produced by this script.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

from analysis.plot_style import GPU_COLORS, strip_figure_titles


DISPLAY = {
    "gradiend": "GRADIEND",
    "actiend": "ACTIEND",
    "agiend": "AGIEND",
    "cga": "CGA",
    "caga": "CAGA",
    "caa": "CAA",
    "sae": "SAE",
}
# Exact order used by the rendered paper method tables (after collapsing the
# pairwise/one-sided rows to the runtime benchmark's method families).
ORDER = ("cga", "gradiend", "caga", "agiend", "caa", "actiend", "sae")
STAGES = ("fit_train", "encode", "causal")
STAGE_LABELS = {
    "fit_train": "Fit / train",
    "encode": "Detection",
    "causal": "Intervention",
}
STAGE_COLORS = {
    "fit_train": GPU_COLORS["blue"],
    "encode": GPU_COLORS["teal"],
    "causal": GPU_COLORS["red"],
}


def configure_paper_font(font_size: int = 16) -> None:
    """Use the bundled Times face that the LaTeX paper figures use."""
    paper_font = Path(__file__).resolve().parents[1] / "times.ttf"
    if paper_font.is_file():
        font_manager.fontManager.addfont(str(paper_font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(paper_font)).get_name()
    plt.rcParams.update({
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": font_size,
        "axes.labelsize": font_size + 1,
        "xtick.labelsize": font_size - 1,
        "ytick.labelsize": font_size,
        "legend.fontsize": font_size - 1,
    })


def _records(payload: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    raw = payload.get("raw") if isinstance(payload.get("raw"), Mapping) else {}
    rows = raw.get("cost_ledger") or raw.get("cost") or []
    return [r for r in rows if isinstance(r, Mapping) and r.get("backend")]


def with_sae_companion(payload: Mapping[str, Any], sae_payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Replace the SAE cost rows of ``payload`` by those of the k=1-only companion run.

    ``runtime_benchmark_sae_k1`` reruns just the SAE family with the headline k=1
    selection (four causal evaluations instead of the k1+kstar eight), on the same
    hardware; every other backend keeps its rows from the main benchmark.
    """
    raw = dict(payload.get("raw") or {})
    key = "cost_ledger" if raw.get("cost_ledger") else "cost"
    kept = [r for r in raw.get(key) or [] if not (isinstance(r, Mapping) and str(r.get("backend")).lower() == "sae")]
    sae_rows = [r for r in _records(sae_payload) if str(r.get("backend")).lower() == "sae"]
    if not sae_rows:
        raise SystemExit("SAE companion run has no SAE cost rows.")
    raw[key] = kept + sae_rows
    return {**payload, "raw": raw}


def summarize(payload: Mapping[str, Any]) -> tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    """Return additive seconds and max GPU peaks for each backend-stage."""
    seconds: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    peaks: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for row in _records(payload):
        backend = str(row["backend"]).lower()
        if backend not in DISPLAY:
            continue
        phase = str(row.get("phase") or "other")
        # An SAE is pre-trained externally: selecting a latent from that fixed
        # dictionary is discovery/detection, not representation fitting.  The
        # same is true for CAA's direct direction selection.  Keeping these
        # costs in ``encode`` makes a zero Fit/train segment mean exactly what
        # the figure claims for methods with no concept-specific optimisation.
        stage = "fit_train" if phase == "train" else "encode" if phase == "feature_select" else phase
        if stage not in STAGES:
            continue
        seconds[backend][stage] += float(row.get("seconds") or 0.0)
        for metric, key in (
            ("allocated", "peak_cuda_gb"),
            ("reserved", "peak_cuda_reserved_gb"),
            ("delta", "peak_delta_gb"),
        ):
            peaks[backend][f"{stage}:{metric}"] = max(
                peaks[backend][f"{stage}:{metric}"], float(row.get(key) or 0.0)
            )
    return seconds, peaks


def plot_runtime(seconds: Mapping[str, Mapping[str, float]], out_dir: Path) -> list[Path]:
    backends = [b for b in ORDER if b in seconds]
    y = np.arange(len(backends))
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    left = np.zeros(len(backends))
    for stage in STAGES:
        values = np.array([seconds[b].get(stage, 0.0) / 60.0 for b in backends])
        bars = ax.barh(y, values, left=left, color=STAGE_COLORS[stage], label=STAGE_LABELS[stage])
        for bar, value in zip(bars, values):
            if value >= 2.5:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_y() + bar.get_height() / 2,
                    f"{value:.1f}", ha="center", va="center", fontsize=plt.rcParams["font.size"] - 3, color="white",
                )
        left += values
    ax.set_yticks(y, [DISPLAY[b] for b in backends])
    ax.invert_yaxis()
    ax.set_xlabel("Wall-clock time (minutes)")
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(
        loc="lower right", ncol=3, frameon=True, fancybox=False,
        framealpha=1, edgecolor="#555555", borderpad=0.55,
    )
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    strip_figure_titles(fig)
    paths = [out_dir / "runtime_benchmark_stage_runtime.pdf", out_dir / "runtime_benchmark_stage_runtime.png"]
    for path in paths:
        fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return paths


def plot_memory(peaks: Mapping[str, Mapping[str, float]], out_dir: Path) -> list[Path]:
    backends = [b for b in ORDER if b in peaks]
    y = np.arange(len(backends))
    allocated = np.array([max((peaks[b].get(f"{s}:allocated", 0.0) for s in STAGES), default=0.0) for b in backends])
    reserved = np.array([max((peaks[b].get(f"{s}:reserved", 0.0) for s in STAGES), default=0.0) for b in backends])

    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    # Reserved CUDA memory contains allocated memory; show it as the wider
    # background bar instead of stacking the two and double-counting bytes.
    ax.barh(y, reserved, 0.62, label="Peak reserved", color=GPU_COLORS["red"])
    ax.barh(y, allocated, 0.38, label="Peak allocated", color=GPU_COLORS["blue"])
    observed_max = float(max(np.max(allocated), np.max(reserved)))
    # The L40S capacity is much larger than every observed peak.  Plotting it
    # as a reference line would compress the informative 0--6 GiB region.
    ax.set_xlim(0, observed_max * 1.32)
    ax.set_yticks(y, [DISPLAY[b] for b in backends])
    ax.invert_yaxis()
    ax.set_xlabel("Peak GPU memory (GiB)")
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.legend(
        loc="lower right", frameon=True, fancybox=False,
        framealpha=1, edgecolor="#555555", borderpad=0.55,
    )
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    strip_figure_titles(fig)
    paths = [out_dir / "runtime_benchmark_gpu_memory.pdf", out_dir / "runtime_benchmark_gpu_memory.png"]
    for path in paths:
        fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return paths


def _panel_label(payload: Mapping[str, Any]) -> str:
    """Compact column header for the joint GPT-2/Llama appendix figure."""
    rows = list(_records(payload))
    run = next((row.get("run") for row in rows if isinstance(row.get("run"), Mapping)), {})
    model = str((run or {}).get("model_key") or "model")
    model_display = {
        "gpt2-small": "GPT-2",
        "llama-3.1-8b": "Llama-3.1-8B",
    }.get(model, model)
    gpu = str((run or {}).get("gpu_name") or "GPU")
    gpu_display = "A100" if "A100" in gpu else "L40S" if "L40" in gpu else gpu
    return f"{model_display} / {gpu_display}"


def plot_joint(
    panels: list[tuple[Mapping[str, Any], Mapping[str, Mapping[str, float]], Mapping[str, Mapping[str, float]]]],
    out_dir: Path,
) -> list[Path]:
    """Write a 2×2 runtime/VRAM comparison while retaining readable per-model scales."""
    fig, axes = plt.subplots(2, 2, figsize=(20, 13))
    runtime_handles = []
    memory_handles = []
    for row, (payload, seconds, peaks) in enumerate(panels):
        backends = [backend for backend in ORDER if backend in seconds or backend in peaks]
        y = np.arange(len(backends))
        runtime_ax = axes[row, 0]
        left = np.zeros(len(backends))
        for stage in STAGES:
            values = np.array([seconds[backend].get(stage, 0.0) / 60.0 for backend in backends])
            bars = runtime_ax.barh(y, values, left=left, color=STAGE_COLORS[stage], label=STAGE_LABELS[stage])
            if row == 0:
                runtime_handles.append(bars[0])
            left += values
        runtime_ax.set_yticks(y, [DISPLAY[backend] for backend in backends])
        runtime_ax.invert_yaxis()
        runtime_ax.set_xlabel("Wall-clock time (minutes)")
        runtime_ax.grid(axis="x", color="#D9D9D9", linewidth=0.7)
        runtime_ax.set_axisbelow(True)
        for spine in ("top", "right", "left"):
            runtime_ax.spines[spine].set_visible(False)

        memory_ax = axes[row, 1]
        allocated = np.array([max((peaks[backend].get(f"{stage}:allocated", 0.0) for stage in STAGES), default=0.0) for backend in backends])
        reserved = np.array([max((peaks[backend].get(f"{stage}:reserved", 0.0) for stage in STAGES), default=0.0) for backend in backends])
        reserved_bars = memory_ax.barh(y, reserved, 0.62, label="Peak reserved", color=GPU_COLORS["red"])
        allocated_bars = memory_ax.barh(y, allocated, 0.38, label="Peak allocated", color=GPU_COLORS["blue"])
        if row == 0:
            memory_handles.extend((reserved_bars[0], allocated_bars[0]))
        # The adjacent runtime panel carries the shared method labels in the
        # same canonical order; repeating them here only consumes horizontal
        # space in the compact joint figure.
        memory_ax.set_yticks(y)
        memory_ax.set_yticklabels([])
        memory_ax.invert_yaxis()
        memory_ax.set_xlabel("Peak GPU memory (GiB)")
        memory_ax.grid(axis="x", color="#D9D9D9", linewidth=0.7)
        memory_ax.set_axisbelow(True)
        for spine in ("top", "right", "left"):
            memory_ax.spines[spine].set_visible(False)
        fig.text(0.008, 0.70 - 0.41 * row, _panel_label(payload), ha="center", va="center", rotation=90, fontsize=25)

    fig.legend(runtime_handles, [STAGE_LABELS[stage] for stage in STAGES], loc="upper center", ncol=3,
               bbox_to_anchor=(0.29, 0.915), frameon=True, fancybox=False, framealpha=1, edgecolor="#555555", borderpad=0.45)
    fig.legend(memory_handles, ["Peak reserved", "Peak allocated"], loc="upper center", ncol=2,
               bbox_to_anchor=(0.76, 0.915), frameon=True, fancybox=False, framealpha=1, edgecolor="#555555", borderpad=0.45)
    fig.subplots_adjust(left=0.10, right=0.99, top=0.855, bottom=0.08, wspace=0.10, hspace=0.30)
    strip_figure_titles(fig)
    paths = [out_dir / "runtime_benchmark_gpt2_llama_joint.pdf", out_dir / "runtime_benchmark_gpt2_llama_joint.png"]
    for path in paths:
        fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return paths


def plot_stage_memory(peaks: Mapping[str, Mapping[str, float]], out_dir: Path) -> list[Path]:
    """Appendix view: stage-wise peak allocated/reserved GPU RAM.

    Colour encodes reserved CUDA RAM and each cell prints allocated/reserved
    GiB.  This retains the diagnostic detail omitted by the compact main bar
    figure without confusing maxima with additive quantities.
    """
    backends = [b for b in ORDER if b in peaks]
    reserved = np.array([[peaks[b].get(f"{s}:reserved", 0.0) for s in STAGES] for b in backends])
    allocated = np.array([[peaks[b].get(f"{s}:allocated", 0.0) for s in STAGES] for b in backends])
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    image = ax.imshow(reserved, cmap="Reds", aspect="auto")
    threshold = reserved.max() * 0.55 if reserved.size else 0.0
    for row in range(len(backends)):
        for col in range(len(STAGES)):
            value = reserved[row, col]
            ax.text(
                col, row, f"{allocated[row, col]:.1f} / {value:.1f}",
                ha="center", va="center", fontsize=8,
                color="white" if value > threshold else "#222222",
            )
    ax.set_xticks(range(len(STAGES)), [STAGE_LABELS[s] for s in STAGES])
    ax.set_yticks(range(len(backends)), [DISPLAY[b] for b in backends])
    ax.set_xlabel("Stage (cell: peak allocated / peak reserved GiB)")
    colourbar = fig.colorbar(image, ax=ax, pad=0.03)
    colourbar.set_label("Peak reserved GPU RAM (GiB)")
    fig.tight_layout()
    strip_figure_titles(fig)
    paths = [out_dir / "runtime_benchmark_gpu_memory_stages.pdf", out_dir / "runtime_benchmark_gpu_memory_stages.png"]
    for path in paths:
        fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="Benchmark results.json")
    parser.add_argument("--out", type=Path, default=Path("analysis/plots/runtime_benchmark_l40s"))
    parser.add_argument(
        "--joint-with", type=Path,
        help="Second benchmark results.json; writes a 2×2 runtime/VRAM comparison.",
    )
    parser.add_argument(
        "--joint-out", type=Path,
        help="Output directory for --joint-with (defaults beside --out).",
    )
    parser.add_argument(
        "--sae-from", type=Path,
        help="SAE k=1 companion results.json (runtime_benchmark_sae_k1) replacing the SAE rows of RESULTS.",
    )
    parser.add_argument(
        "--joint-sae-from", type=Path,
        help="SAE k=1 companion results.json replacing the SAE rows of --joint-with.",
    )
    parser.add_argument(
        "--stage-memory-diagnostic", action="store_true",
        help="Also write the stage-wise allocated/reserved RAM heatmap (diagnostic, not paper figure).",
    )
    args = parser.parse_args()
    payload = json.loads(args.results.read_text(encoding="utf-8"))
    if args.sae_from:
        payload = with_sae_companion(payload, json.loads(args.sae_from.read_text(encoding="utf-8")))
    configure_paper_font(16)
    seconds, peaks = summarize(payload)
    if not seconds or not peaks:
        raise SystemExit("No method-attributed cost ledger found.")
    args.out.mkdir(parents=True, exist_ok=True)
    written = [*plot_runtime(seconds, args.out), *plot_memory(peaks, args.out)]
    if args.stage_memory_diagnostic:
        written.extend(plot_stage_memory(peaks, args.out))
    for path in written:
        print(path)
    if args.joint_with:
        comparison = json.loads(args.joint_with.read_text(encoding="utf-8"))
        if args.joint_sae_from:
            comparison = with_sae_companion(comparison, json.loads(args.joint_sae_from.read_text(encoding="utf-8")))
        comparison_seconds, comparison_peaks = summarize(comparison)
        joint_out = args.joint_out or args.out.parent / "runtime_benchmark_joint"
        joint_out.mkdir(parents=True, exist_ok=True)
        configure_paper_font(22)
        for path in plot_joint(
            [(comparison, comparison_seconds, comparison_peaks), (payload, seconds, peaks)],
            joint_out,
        ):
            print(path)


if __name__ == "__main__":
    main()
