"""Global paper-figure colors and marker semantics.

Keep scientific encodings here so every plotting module uses the same visual
language.  Signal owns hue and marker shape; estimator owns shade; construction
owns fill.  Changing the three base palettes here updates every consumer.
"""

from __future__ import annotations

import os
from typing import Dict, Tuple

SIGNAL_LABELS: Dict[str, str] = {
    "activation_value": "Activation value",
    "activation_gradient": "Activation gradient",
    "parameter_gradient": "Parameter gradient",
}

# One restrained palette shared by the compute and factor figures.  The three
# anchor hues are the allocated/reserved/incremental GPU-memory bars; a light
# and dark shade distinguish contrastive from learned estimators without
# introducing a second unrelated colour language.
GPU_COLORS: Dict[str, str] = {
    "blue": "#4C78A8",
    "red": "#E45756",
    "teal": "#72B7B2",
}
SIGNAL_PALETTE: Dict[str, Dict[str, str]] = {
    "activation_value": {"light": "#A8C1D8", "dark": GPU_COLORS["blue"], "base": GPU_COLORS["blue"]},
    "activation_gradient": {"light": "#B9DFD9", "dark": GPU_COLORS["teal"], "base": GPU_COLORS["teal"]},
    "parameter_gradient": {"light": "#F3B8B0", "dark": GPU_COLORS["red"], "base": GPU_COLORS["red"]},
}

SIGNAL_MARKERS: Dict[str, str] = {
    "activation_value": "o",
    "activation_gradient": "s",
    "parameter_gradient": "^",
}

ESTIMATOR_LABELS: Dict[str, str] = {
    "contrastive_mean": "Contrastive mean",
    "iend": "IEND",
}

# SAE is a neutral reference rather than a fourth categorical hue.
SAE_VARIANT_COLORS: Dict[str, str] = {
    "k1": "#666666",
    "kstar": "#A6A6A6",
}
SAE_COLOR = SAE_VARIANT_COLORS["k1"]
SAE_MARKER = "D"

BACKEND_SCIENCE: Dict[str, Tuple[str, str]] = {
    "caa": ("activation_value", "contrastive_mean"),
    "actiend": ("activation_value", "iend"),
    "actiend_pre": ("activation_value", "iend"),
    "actiend_ridge": ("activation_value", "iend"),
    "caga": ("activation_gradient", "contrastive_mean"),
    "agiend": ("activation_gradient", "iend"),
    "cga": ("parameter_gradient", "contrastive_mean"),
    "cga_tensor_norm": ("parameter_gradient", "contrastive_mean"),
    "gradiend": ("parameter_gradient", "iend"),
}


def method_science(method: str) -> Tuple[str, str]:
    """Return ``(signal, estimator)``; SAE is a neutral special case."""
    backend = str(method).partition(":")[0]
    return BACKEND_SCIENCE.get(backend, ("sae", "sae"))


def method_color(method: str) -> str:
    signal, estimator = method_science(method)
    if signal == "sae":
        variant = str(method).partition(":")[2].partition(":")[0]
        return SAE_VARIANT_COLORS.get(variant, SAE_COLOR)
    shade = "dark" if estimator == "iend" else "light"
    return SIGNAL_PALETTE[signal][shade]


def method_marker(method: str) -> str:
    signal, _estimator = method_science(method)
    return SAE_MARKER if signal == "sae" else SIGNAL_MARKERS[signal]


# A hollow marker carries its color only in the edge, so the edge is thicker.
HOLLOW_EDGE_WIDTH = 2.0


def scatter_style(method: str, construction: str) -> Dict[str, object]:
    """Matplotlib kwargs implementing hue/shape/shade/fill semantics."""
    color = method_color(method)
    # SAE is intrinsically one-sided and is always drawn hollow.
    filled = construction == "pairwise" and method_science(method)[0] != "sae"
    return {
        "marker": method_marker(method),
        "facecolors": color if filled else "none",
        "edgecolors": color,
        "linewidth": 1.15 if filled else HOLLOW_EDGE_WIDTH,
    }


BACKEND_COLORS: Dict[str, str] = {
    backend: method_color(backend) for backend in BACKEND_SCIENCE
}
BACKEND_COLORS.update({"sae": SAE_COLOR, "sae_pre": SAE_COLOR})


def strip_figure_titles(figure) -> None:
    """Remove Matplotlib titles; paper prose belongs in LaTeX captions."""
    suptitle = getattr(figure, "_suptitle", None)
    if suptitle is not None:
        suptitle.set_text("")
    for axis in figure.axes:
        axis.set_title("")


def show_plot_if_requested(pyplot) -> None:
    """Show a figure only when interactive inspection was explicitly enabled.

    Set ``GRADIEND_SHOW_PLOTS=1`` in an IDE run configuration, or use a
    script-specific ``--show`` flag where provided.  ``auto`` is available
    for IDE-driven runs, but explicit opt-in is the default so report/batch
    commands remain non-interactive.
    """
    setting = os.environ.get("GRADIEND_SHOW_PLOTS", "").strip().lower()
    enabled = setting in {"1", "true", "yes", "on"}
    if setting == "auto":
        enabled = any(
            os.environ.get(name)
            for name in ("PYCHARM_HOSTED", "VSCODE_PID", "SPYDER_ARGS")
        )
    if enabled:
        pyplot.show()
