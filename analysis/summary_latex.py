#!/usr/bin/env python
"""LaTeX summary tables for headline method groups (paper tables).

One **methods × tasks** display per headline metric (encoder + causal), backed
by a numeric tasks × methods CSV. Separate mean and median **methods × metrics**
summary tables aggregate over the same task set as the matrix tables (all tasks
in data by default). The reducer is selected generically via ``--aggregation``;
the standard PDF launcher emits both variants. ``SUMMARY_TASKS`` /
``resolve_summary_tasks`` remain available but are no longer used by
``main()`` — averaging over only 4 "reliable" tasks made the Summary page
look deceptively complete on a sparse run (see CLAUDE.md, 2026-08-19).

A companion ``Summary (rank)`` table (``summary_methods_rank_*``) reports,
per method/metric, the method's **mean proportional placement among methods
with data** on each task — 1 = always best of its competing field that task,
0 = always worst, ties averaged — rather than the mean of the raw metric. This
is deliberately robust to the outlier-driven means the plain ``Summary``
table can show on the causal metrics, where a handful of tasks with
unusually large/small effects can dominate a raw mean even though the
method rarely ranks best or worst elsewhere. The rank is normalized by how
many methods actually had data on that task (not a raw 1..N position)
because one-pole and two-pole methods, and the various SAE/CAA baselines,
don't all compete against the same field size from task to task.

Also automatically generated for every ``--model`` (no separate command): a
GRADIEND/ACTIEND **training-convergence** overview, from
``analysis/convergence_overview.py`` -- ``summary_convergence_counts_*``
(compact backend x status counts) and ``summary_convergence_problems_*`` (a
longtable of every collapsed / not_converged / unknown_low_quality run, for
debugging). Both are read straight from ``runs/{model}/[.../]{task}/artifacts/
{backend}__.../done.json``, not from ``results.json`` -- that's the only
place per-run convergence detail lives at all. Folded into the same
``--write-book`` / ``scripts/summary_tables_pdf.sh`` output as everything
else.

Data policy: ``runs/{model}/`` only. Archived ``runs/{model}_old/`` is never merged.

Examples::

  python analysis/summary_latex.py --model gpt2-small
  python analysis/summary_latex.py --model gpt2-small --tasks gender_en,emotion,ioi
  python analysis/summary_latex.py --model gpt2-small --list-tasks
  bash scripts/summary_tables_pdf.sh
  MODELS=gpt2-small,pythia-70m-deduped bash scripts/summary_tables_pdf.sh
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.convergence_overview import (  # noqa: E402
    collect_convergence_rows,
    format_convergence_counts_latex,
    format_convergence_problems_latex,
    PROBLEM_STATUSES as CONVERGENCE_PROBLEM_STATUSES,
)
from analysis.layer_aggregations import (  # noqa: E402
    canonical_layer_aggregations,
    format_layer_aggregation_latex,
)
from analysis.method_groups import (  # noqa: E402
    METHOD_GROUP_ORDER,
    _load_results,
    collect_group_rows_for_results,
)
from analysis.task_order import PAPER_TASK_ORDER as CANONICAL_PAPER_TASK_ORDER, order_tasks  # noqa: E402
from analysis.task_specs import (  # noqa: E402
    GAP_DISPLAY,
    MISSING_DISPLAY,
    collect_task_specs,
    group_is_applicable,
    resolve_spec,
    spec_for_task,
)

DEFAULT_RUNS = ROOT / "runs"
DEFAULT_OUT = ROOT / "analysis" / "tables" / "latex"
DEFAULT_HEADLINE_SCATTER_DIR = ROOT / "analysis" / "figures"

# The paper-wide comparison deliberately mixes OUTPUT_SUBDIR layouts: the two
# full small-model studies live in suite_full2, while the later lean large-model
# replications were written directly below runs/{model}.  Keep that provenance
# explicit instead of making callers remember four separate invocations.
from analysis.report_model_policy import PAPER_MODELS as DEFAULT_CROSS_MODEL_SOURCES  # noqa: E402


def parse_cross_model_sources(values: Optional[Sequence[str]]) -> Tuple[Tuple[str, str], ...]:
    """Parse repeatable ``MODEL[=OUTPUT_SUBDIR]`` source arguments."""
    if not values:
        return DEFAULT_CROSS_MODEL_SOURCES
    parsed: List[Tuple[str, str]] = []
    seen: set[str] = set()
    for raw in values:
        model, separator, subdir = str(raw).partition("=")
        model = model.strip()
        subdir = subdir.strip() if separator else ""
        if not model:
            raise ValueError(f"invalid empty model in --source {raw!r}")
        if model in seen:
            raise ValueError(f"duplicate --source model {model!r}")
        if subdir and (Path(subdir).is_absolute() or ".." in Path(subdir).parts):
            raise ValueError(f"OUTPUT_SUBDIR must be a relative child name: {subdir!r}")
        seen.add(model)
        parsed.append((model, subdir))
    return tuple(parsed)


def parse_summary_csv_sources(
    values: Sequence[str],
) -> Tuple[Tuple[str, Path], ...]:
    """Parse compact canonical-summary inputs for a local headline rebuild.

    This is intentionally separate from ``--source``: the latter names run
    trees and therefore reads every task's ``results.json``. A summary CSV is
    the normalized lightweight representation needed by headline aggregates.
    """
    parsed: List[Tuple[str, Path]] = []
    seen: set[str] = set()
    for raw in values:
        model, separator, raw_path = str(raw).partition("=")
        model, raw_path = model.strip(), raw_path.strip()
        if not separator or not model or not raw_path:
            raise ValueError("--summary-csv must be MODEL=PATH_TO_summary_merged.csv")
        if model in seen:
            raise ValueError(f"duplicate --summary-csv model {model!r}")
        path = Path(raw_path)
        if not path.is_file():
            raise ValueError(f"missing --summary-csv input: {path}")
        seen.add(model)
        parsed.append((model, path))
    return tuple(parsed)


def load_cross_model_summary_csvs(
    csv_sources: Sequence[Tuple[str, Path]],
) -> Tuple[
    Dict[str, pd.DataFrame],
    Dict[str, Dict[Tuple[str, str], Mapping[str, Any]]],
    Dict[str, str],
]:
    """Load normalized local CSVs without reopening any results artifacts."""
    frames: Dict[str, pd.DataFrame] = {}
    suites: Dict[str, str] = {}
    for model, path in csv_sources:
        frame = pd.read_csv(path)
        if "model" not in frame.columns:
            frame["model"] = model
        frame["model"] = model
        frames[model] = frame
        values = (
            [str(value) for value in frame.get("suite", pd.Series(dtype=str)).dropna().unique()]
            if not frame.empty
            else []
        )
        suites[model] = values[0] if values else "core"
    # Kept for the existing completeness API; declared task suites need no
    # artifact scan once the canonical rows are already in the CSV.
    return frames, {model: {} for model, _path in csv_sources}, suites

MODEL_LATEX: Dict[str, str] = {
    "gpt2-small": "GPT-2",
    "pythia-70m-deduped": "Pythia-70M",
    "llama-3.1-8b": "Llama-3.1-8B",
    "qwen3.5-9b-base": "Qwen-3.5-9B",
    "qwen3.5-2b-base": "Qwen-3.5-2B",
    "gemma-2-2b": "Gemma-2-2B",
}

# Parameter counts, used to order cross-model figures small -> large.
MODEL_PARAMS: Dict[str, float] = {
    "pythia-70m-deduped": 70e6,
    "gpt2-small": 124e6,
    "llama-3.1-8b": 8.0e9,
    "qwen3.5-9b-base": 9.0e9,
    "qwen3.5-2b-base": 2.0e9,
    "gemma-2-2b": 2.0e9,
}

TASK_COMMAND_DEFINITIONS: Tuple[str, ...] = (
    r"\newcommand{\taskicon}[2][0.85]{\scalebox{#1}{\faIcon{#2}}}",
    r"\newcommand{\taskGender}{\taskicon{mars-and-venus}\hspace{1pt}\textsc{Gender}}",
    r"\newcommand{\taskEmotion}{\taskicon{face-smile}\hspace{1pt}\textsc{Emotion}}",
    r"\newcommand{\taskRace}{\taskicon{person}\hspace{0.5pt}\textsc{Race}}",
    r"\newcommand{\taskReligion}{\taskicon{hands-praying}\hspace{1pt}\textsc{Religion}}",
    r"\newcommand{\taskPronNum}{\taskicon{people-group}\hspace{1pt}\textsc{PronNum}}",
    r"\newcommand{\taskPronPers}{\taskicon{people-arrows}\hspace{1pt}\textsc{PronPers}}",
    r"\newcommand{\taskRavelCont}{\taskicon{earth-africa}\hspace{1pt}\textsc{RavelCont}}",
    r"\newcommand{\taskRavelCountry}{\taskicon{flag}\hspace{1pt}\textsc{RavelCntry}}",
    r"\newcommand{\taskRavelLang}{\taskicon{language}\hspace{1pt}\textsc{RavelLang}}",
    r"\newcommand{\taskLangID}{\taskicon{fingerprint}\hspace{1pt}\textsc{LangID}}",
    r"\newcommand{\taskIOI}{\taskicon{people-arrows}\hspace{1pt}\textsc{IOIOwn}}",
    r"\newcommand{\taskMIBIOI}{\taskicon{arrow-right-arrow-left}\hspace{1pt}\textsc{IOI}}",
    r"\newcommand{\taskKeyValue}{\taskicon{key}\hspace{1pt}\textsc{KeyValue}}",
    r"\newcommand{\taskInduction}{\taskicon{forward-step}\hspace{1pt}\textsc{Induction}}",
    r"\newcommand{\taskRepetition}{\taskicon{arrows-rotate}\hspace{1pt}\textsc{Repetition}}",
    r"\newcommand{\taskFuncComp}{\taskicon{link}\hspace{1pt}\textsc{FuncComp}}",
)

# Summary-table books are standalone documents, so they must load the same
# Font Awesome 7 package that supplies the \faIcon task-label macros used by
# the paper and the LaTeX-rendered figures.
FA_ICON_PACKAGE = r"\usepackage{fontawesome7}"

MEAN_LABEL = "Mean"
AGGREGATIONS: Tuple[str, ...] = ("mean", "median")

PAPER_METHOD_GROUPS: Tuple[str, ...] = (
    # Strongest-to-weakest supervision signal. Within each signal family,
    # contrastive mean precedes IEND; variants stay beside their parent.
    "cga:two_pole",
    "cga:one_pole",
    "cga_tensor_norm:two_pole",
    "cga_tensor_norm:one_pole",
    "gradiend:two_pole",
    "gradiend:one_pole",
    # Activation-gradient signal.
    "caga:two_pole",
    "caga:one_pole",
    "agiend:two_pole",
    "agiend:one_pole",
    # Activation signal, ending with the untargeted SAE reference.
    "caa:two_pole",
    "caa:one_pole",
    "actiend:two_pole",
    "actiend:one_pole",
    "sae:k1",
    "sae:kstar",
    "sae_pre:k1",
    "sae_pre:kstar",
)

# The selected headline view intentionally does not treat the pre-prediction
# SAE or tensor-norm CGA arm as default headline methods.  The full view is a
# genuine all-canonical-methods view, including those arms.
# The headline uses SAE k=1 only: k* and the pre-prediction site are
# explicitly appendix controls.  Keeping the exclusion here makes every
# selected headline table and figure agree without relying on per-script
# filtering.
TABLE_EXCLUDED_FAMILIES: Tuple[str, ...] = ("actiend_ridge",)
HEADLINE_EXCLUDED_FAMILIES: Tuple[str, ...] = TABLE_EXCLUDED_FAMILIES + (
    "sae_pre",
    "cga_tensor_norm",
)
HEADLINE_EXCLUDED_GROUPS: Tuple[str, ...] = ("sae:kstar",)


def _headline_family(group: str) -> str:
    return str(group).partition(":")[0]


def _configured_method_list(name: str) -> Optional[Tuple[str, ...]]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def headline_method_groups(
    method_set: str = "selected", *, frame: Optional[pd.DataFrame] = None
) -> Tuple[str, ...]:
    """Return the configurable methods used by headline tables/figures.

    ``selected`` is the paper/default subset.  ``full`` uses every canonical
    group present in the supplied frame. ``HEADLINE_METHODS`` and
    ``HEADLINE_FULL_METHODS`` are comma-separated overrides for reproducible
    custom builds.
    """
    if method_set not in {"selected", "full"}:
        raise ValueError(f"unknown headline method set: {method_set!r}")
    override = _configured_method_list(
        "HEADLINE_METHODS" if method_set == "selected" else "HEADLINE_FULL_METHODS"
    )
    if override is not None:
        return tuple(override)
    selected = tuple(
        group
        for group in PAPER_METHOD_GROUPS
        if (
            _headline_family(group) not in HEADLINE_EXCLUDED_FAMILIES
            and group not in HEADLINE_EXCLUDED_GROUPS
        )
    )
    if method_set == "selected" or frame is None or frame.empty or "method_group" not in frame:
        return selected
    discovered = {
        str(group)
        for group in frame["method_group"].dropna().astype(str)
        if _headline_family(str(group)) not in TABLE_EXCLUDED_FAMILIES
    }
    return tuple(_order_methods(sorted(discovered | set(selected))))

METHOD_LATEX: Dict[str, str] = {
    "gradiend:two_pole": r"GRADIEND$_{\mathrm{pw}}$",
    "gradiend:one_pole": r"GRADIEND$_{\mathrm{1s}}$",
    "actiend:two_pole": r"ACTIEND$_{\mathrm{pw}}$",
    "actiend:one_pole": r"ACTIEND$_{\mathrm{1s}}$",
    "actiend_ridge:two_pole": r"ACTIEND$_{\mathrm{ridge,pw}}$",
    "actiend_ridge:one_pole": r"ACTIEND$_{\mathrm{ridge,1s}}$",
    "sae:k1": r"\MethodSAEkOne",
    "sae:kstar": r"\MethodSAEkStar",
    "sae_pre:k1": r"\MethodSAEPrekOne",
    "sae_pre:kstar": r"\MethodSAEPrekStar",
    "caa:two_pole": r"CAA$_{\mathrm{pw}}$",
    "caa:one_pole": r"CAA$_{\mathrm{1s}}$",
    "cga:two_pole": r"CGA$_{\mathrm{pw}}$",
    "cga:one_pole": r"CGA$_{\mathrm{1s}}$",
    "cga_tensor_norm:two_pole": r"CGA$_{\mathrm{tn,pw}}$",
    "cga_tensor_norm:one_pole": r"CGA$_{\mathrm{tn,1s}}$",
    "caga:two_pole": r"CAGA$_{\mathrm{pw}}$",
    "caga:one_pole": r"CAGA$_{\mathrm{1s}}$",
    "agiend:two_pole": r"AGIEND$_{\mathrm{pw}}$",
    "agiend:one_pole": r"AGIEND$_{\mathrm{1s}}$",
}

# ``sae:k1`` is the single SAE method in the headline comparison.  The main
# paper therefore calls it simply SAE; the explicit recipe belongs to the
# appendix/full table set, where it distinguishes the ablation baseline.
HEADLINE_METHOD_LATEX: Dict[str, str] = {
    **METHOD_LATEX,
    "sae:k1": r"\MethodSAE",
}

# Multi-class, binary, then one-pole / single-claim. Hidden/legacy
# overlays such as ``gender_en_pre`` are never included.
PAPER_MULTICLASS_TASKS: Tuple[str, ...] = (
    "race",
    "religion",
    "ravel_continent",
    "ravel_country",
    "ravel_language",
    "language",
    "pronoun_person",
)

PAPER_BINARY_TASKS: Tuple[str, ...] = (
    "pronoun_number",
    "gender_en",
    "emotion",
)

PAPER_PAIRWISE_TASKS: Tuple[str, ...] = PAPER_MULTICLASS_TASKS + PAPER_BINARY_TASKS

PAPER_ONE_POLE_TASKS: Tuple[str, ...] = (
    "induction",
    "ioi_mib",
    "repetition",
    "function_composition",
    "key_value",
    "race_one_pole",
    "religion_one_pole",
)

PAPER_TASK_ORDER: Tuple[str, ...] = CANONICAL_PAPER_TASK_ORDER

# race_one_pole / religion_one_pole are the deprecated source=alternative one-pole
# check-tasks; since one-pole training is source=both for every task (2026-08-31),
# the base race/religion one-pole ablations subsume them, so they are hidden from
# the paper tables (configs kept, still loadable via --task). Matches their
# ``study_hidden: true`` in list_tasks()/TASKS=all.
HIDDEN_MATRIX_TASKS: frozenset[str] = frozenset(
    {
        "gender_en_pre",
        "ioi",
        "race_one_pole",
        "religion_one_pole",
    }
)

# Tasks used for the methods × metrics summary table (reliable headline coverage).
SUMMARY_TASKS: Tuple[str, ...] = (
    "emotion",
    "gender_en",
    "pronoun_number",
    "pronoun_person",
)

# Back-compat alias (tests / docs).

TASK_LATEX: Dict[str, str] = {
    "emotion": r"\taskEmotion",
    "function_composition": r"\taskFuncComp",
    "gender_en": r"\taskGender",
    "induction": r"\taskInduction",
    "ioi_mib": r"\taskMIBIOI",
    "key_value": r"\taskKeyValue",
    "language": r"\taskLangID",
    "pronoun_number": r"\taskPronNum",
    "pronoun_person": r"\taskPronPers",
    "race": r"\taskRace",
    "race_one_pole": r"\taskRace",
    "ravel_continent": r"\taskRavelCont",
    "ravel_country": r"\taskRavelCountry",
    "ravel_language": r"\taskRavelLang",
    "religion": r"\taskReligion",
    "religion_one_pole": r"\taskReligion",
    "repetition": r"\taskRepetition",
}

MATRIX_METRICS: Tuple[str, ...] = (
    "roc_auc_neutral",
    "roc_auc_other",
    "neutral_specificity",
    "class_exclusivity",
    "detection_score",
    "causal_signed_effect",
    "causal_signed_effect_weaken",
    "intervention_score",
    "causal_lms",
    "causal_weaken_lms",
)

SHARED_SAE_METHOD_GROUPS: frozenset[str] = frozenset(
    {"sae:k1", "sae:kstar", "sae_pre:k1", "sae_pre:kstar"}
)
RESULT_TABLE_VIEWS: Tuple[str, ...] = ("combined", "pairwise", "one_pole")

HEADLINE_METRICS: Tuple[str, ...] = ("detection_score", "intervention_score")
DETECTION_FIELDS: Tuple[str, ...] = (
    "roc_auc_neutral",
    "roc_auc_other",
    "neutral_specificity",
    "class_exclusivity",
    "detection_score",
)
INTERVENTION_FIELDS: Tuple[str, ...] = (
    "causal_signed_effect",
    "causal_signed_effect_weaken",
    "intervention_score",
    "causal_lms",
    "causal_weaken_lms",
)

METRIC_LATEX: Dict[str, str] = {
    "roc_auc_neutral": r"AUC$_n$",
    "roc_auc_other": r"AUC$_o$",
    "neutral_specificity": r"Spec$_n$",
    "class_exclusivity": "Excl.",
    "detection_score": r"$\mathrm{Det.}$",
    "causal_signed_effect": r"$\Delta P^{+}$",
    "causal_signed_effect_weaken": r"$\Delta P^{-}$",
    "intervention_score": r"$\mathrm{Int.}$",
    "causal_lms": r"LMS$^{+}$",
    "causal_weaken_lms": r"LMS$^{-}$",
}

CAUSAL_FIELDS: Tuple[str, ...] = (
    "causal_signed_effect",
    "causal_signed_effect_weaken",
    "intervention_score",
    "causal_lms",
    "causal_weaken_lms",
)
ENCODER_SUMMARY_FIELDS: Tuple[str, ...] = tuple(
    ["encoding_E"]
    + [metric for metric in MATRIX_METRICS if metric not in CAUSAL_FIELDS]
)
PATCH_FIELDS: Tuple[str, ...] = CAUSAL_FIELDS + MATRIX_METRICS

# Removed output names from earlier script versions (deleted on regenerate).
DEPRECATED_OUTPUTS: Tuple[str, ...] = (
    "summary_metrics_{model}.tex",
    "summary_metrics_{model}.csv",
    "summary_deltap_{model}.tex",
    "summary_deltap_{model}.csv",
    "summary_matrix_encoding_e_{model}.tex",
    "summary_matrix_encoding_e_{model}.csv",
)

WIN_EPS = 1e-6


def _task_label(task: str) -> str:
    return TASK_LATEX.get(task, task.replace("_", " "))


_TASK_COMMAND_DEF_RE = re.compile(
    r"\\newcommand\{\\(\w+)\}\{(?:\\faIcon\{[^}]+\})?\\textsc\{([^}]+)\}\}"
)


def _rendered_task_command_text() -> Dict[str, str]:
    """``\\taskGender`` -> ``Gender``, parsed from ``TASK_COMMAND_DEFINITIONS``
    itself so the rendered text can never drift from what the generated LaTeX
    tables actually typeset."""
    out: Dict[str, str] = {}
    for defn in TASK_COMMAND_DEFINITIONS:
        m = _TASK_COMMAND_DEF_RE.match(defn)
        if m:
            out[f"\\{m.group(1)}"] = m.group(2)
    return out


TASK_COMMAND_RENDERED: Dict[str, str] = _rendered_task_command_text()


def task_label_rendered(task: str) -> str:
    """The literal rendered text of this task's paper LaTeX command (``TASK_LATEX``).

    e.g. ``ravel_country`` -> ``RavelCntry`` (from ``\\taskRavelCountry`` ->
    ``\\textsc{RavelCntry}``) -- not a separately invented human-readable label,
    so a figure titled this way reads identically to how the task is named in
    the generated LaTeX tables.
    """
    macro = TASK_LATEX.get(task)
    if macro is None:
        return task.replace("_", " ")
    return TASK_COMMAND_RENDERED.get(macro, macro.lstrip("\\"))


def _method_label(group: str, *, method_set: str = "selected") -> str:
    labels = METHOD_LATEX if method_set == "full" else HEADLINE_METHOD_LATEX
    return labels.get(group, group.replace("_", r"\_"))


METHOD_SIGNAL: Dict[str, str] = {
    "cga": "parameter_gradient",
    "cga_tensor_norm": "parameter_gradient",
    "gradiend": "parameter_gradient",
    "caga": "activation_gradient",
    "agiend": "activation_gradient",
    "caa": "activation",
    "actiend": "activation",
    "actiend_ridge": "activation",
    "actiend_pre": "activation",
    "sae": "activation",
    "sae_pre": "activation",
}

SUMMARY_TABLE_COMMAND_DEFINITIONS: Tuple[str, ...] = (
    r"\providecommand{\MethodSAE}{$\mathrm{SAE}$}",
    r"\providecommand{\MethodSAEkOne}{$\mathrm{SAE}^{k=1}$}",
    r"\providecommand{\MethodSAEkStar}{$\mathrm{SAE}^{k^{*}}$}",
    r"\providecommand{\MethodSAEPrekOne}{$\mathrm{SAE}_{\mathrm{pre}}^{k=1}$}",
    r"\providecommand{\MethodSAEPrekStar}{$\mathrm{SAE}_{\mathrm{pre}}^{k^{*}}$}",
    r"\providecommand{\SummarySignalRule}[1]{\midrule}",
    r"\providecommand{\SummaryMethodRule}[1]{\cmidrule(lr){1-#1}}",
)

METHOD_FAMILY_ALIASES: Dict[str, str] = {
    "cga_tensor_norm": "cga",
    "actiend_ridge": "actiend",
    "actiend_pre": "actiend",
    "sae_pre": "sae",
}


def _method_family(group: str) -> str:
    family = str(group).partition(":")[0]
    return METHOD_FAMILY_ALIASES.get(family, family)


def _method_signal(group: str) -> str:
    family = _method_family(group)
    return METHOD_SIGNAL.get(family, family)


def _method_separator(methods: Sequence[str], index: int) -> Optional[str]:
    if index + 1 >= len(methods):
        return None
    current, following = methods[index], methods[index + 1]
    if _method_signal(current) != _method_signal(following):
        return "signal"
    if _method_family(current) != _method_family(following):
        return "method"
    return None


def result_view_methods(methods: Sequence[str], view: str) -> List[str]:
    """Ordered methods for a combined, pairwise, or one-sided result view."""
    if view == "combined":
        return list(methods)
    if view not in RESULT_TABLE_VIEWS:
        raise ValueError(f"unknown result table view: {view!r}")
    pole = "two_pole" if view == "pairwise" else "one_pole"
    return [
        method
        for method in methods
        if method.endswith(f":{pole}")
        or (view in {"pairwise", "one_pole"} and method in SHARED_SAE_METHOD_GROUPS)
    ]


def result_view_tasks(tasks: Sequence[str], view: str) -> List[str]:
    """Tasks that admit the construction used by a result-table view."""
    if view in {"combined", "one_pole"}:
        return list(tasks)
    if view == "pairwise":
        pairwise_tasks = set(PAPER_PAIRWISE_TASKS)
        return [task for task in tasks if task in pairwise_tasks]
    raise ValueError(f"unknown result table view: {view!r}")


def _view_suffix(view: str) -> str:
    return "" if view == "combined" else f"_{view}"


def _view_caption_prefix(view: str) -> str:
    return {
        "combined": "",
        "pairwise": "Pairwise-task ",
        "one_pole": "One-sided ",
    }[view]


def _metric_label(metric: str) -> str:
    return METRIC_LATEX.get(metric, metric.replace("_", r"\_"))


def _metric_slug(metric: str) -> str:
    if metric == "detection_score":
        return "detection"
    if metric == "intervention_score":
        return "intervention"
    return re.sub(r"[^a-z0-9]+", "_", metric.lower()).strip("_")


def _order_tasks(tasks: Sequence[str]) -> List[str]:
    return order_tasks(tasks)


def _is_hidden_task(task: str) -> bool:
    return str(task) in HIDDEN_MATRIX_TASKS




def _order_methods(groups: Sequence[str]) -> List[str]:
    rank = {g: i for i, g in enumerate(METHOD_GROUP_ORDER)}
    paper = {g: i for i, g in enumerate(PAPER_METHOD_GROUPS)}
    return sorted(groups, key=lambda g: (paper.get(g, 1_000 + rank.get(g, 2_000)), g))


def _is_num(value: Any) -> bool:
    if value is None or isinstance(value, bool):
        return False
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False
    return not math.isnan(v)


def _mean(nums: Sequence[float]) -> float:
    vals = [float(x) for x in nums if _is_num(x)]
    return sum(vals) / len(vals) if vals else float("nan")


def _aggregate(nums: Sequence[float], aggregation: str = "mean") -> float:
    """Aggregate numeric values with a named, extensible reducer."""
    vals = sorted(float(x) for x in nums if _is_num(x))
    if not vals:
        return float("nan")
    if aggregation == "mean":
        return sum(vals) / len(vals)
    if aggregation == "median":
        middle = len(vals) // 2
        return vals[middle] if len(vals) % 2 else (vals[middle - 1] + vals[middle]) / 2
    raise ValueError(f"unknown aggregation: {aggregation!r}; expected one of {AGGREGATIONS}")


def _aggregation_suffix(aggregation: str) -> str:
    if aggregation not in AGGREGATIONS:
        raise ValueError(f"unknown aggregation: {aggregation!r}")
    return "" if aggregation == "mean" else f"_{aggregation}"


def _model_dir(runs_root: Path, model: str, subdir: str = "") -> Path:
    """``runs/{model}/`` or, for an ``OUTPUT_SUBDIR`` tree (see slurm/study_array.sh
    and CLAUDE.md's "OUTPUT_SUBDIR" note), ``runs/{model}/{subdir}/``.
    """
    base = Path(runs_root) / model
    return base / subdir if subdir else base


def collect_canonical_group_rows(
    runs_root: Path,
    model: str,
    subdir: str = "",
    *,
    causal_validation_fallback: bool = False,
    representation_view: str = "best_overall",
) -> pd.DataFrame:
    """Load only ``runs/{model}/[{subdir}/]{task}/results.json`` (ignore stray duplicate trees)."""
    rows: List[Dict[str, Any]] = []
    model_dir = _model_dir(runs_root, model, subdir)
    if model_dir.is_dir():
        for path in sorted(model_dir.glob("*/results.json")):
            payload = _load_results(path)
            if not payload:
                continue
            rows.extend(
                collect_group_rows_for_results(
                    payload,
                    results_path=str(path),
                    causal_validation_fallback=causal_validation_fallback,
                    representation_view=representation_view,
                )
            )
    if rows:
        return pd.DataFrame(rows)
    # Keep downstream table builders schema-stable when a model has no synced
    # canonical rows (for example Qwen layer/headline results are absent).
    return pd.DataFrame(
        columns=[
            "model", "task", "suite", "method_group", "backend", "pole",
            "run_status", "n_sources", "source_methods", "encoding_E",
            *MATRIX_METRICS,
        ]
    )


def list_run_tasks(runs_root: Path, model: str, subdir: str = "") -> List[str]:
    """Task ids with a ``results.json``, including failed dumps with no method rows."""
    model_dir = _model_dir(runs_root, model, subdir)
    if not model_dir.is_dir():
        return []
    out: List[str] = []
    for path in sorted(model_dir.glob("*/results.json")):
        task = path.parent.name
        if task and not _is_hidden_task(task):
            out.append(task)
    return out






def _row_key(row: Mapping[str, Any]) -> Tuple[str, str, str]:
    return (str(row["model"]), str(row["task"]), str(row["method_group"]))


def merge_with_fallback(
    current: pd.DataFrame,
    fallback: pd.DataFrame,
    *,
    model: str,
    patch_fields: Sequence[str] = PATCH_FIELDS,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Prefer current rows; patch missing fields and add missing SAE/CAA rows from fallback."""
    cur = current[current["model"] == model].copy()
    fb = fallback[fallback["model"] == model].copy() if not fallback.empty else pd.DataFrame()

    cur_by: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for rec in cur.to_dict(orient="records"):
        cur_by[_row_key(rec)] = dict(rec)

    fb_by: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for rec in fb.to_dict(orient="records"):
        fb_by[_row_key(rec)] = dict(rec)

    provenance: List[Dict[str, Any]] = []

    for key, fb_row in fb_by.items():
        _mod, task, group = key
        if group not in PAPER_METHOD_GROUPS:
            continue
        if key not in cur_by:
            if not str(group).startswith(("sae", "caa:")):
                continue
            row = dict(fb_row)
            row["data_source"] = "fallback"
            row["results_path"] = fb_row.get("results_path")
            cur_by[key] = row
            provenance.append(
                {
                    "model": model,
                    "task": task,
                    "method_group": group,
                    "kind": "row",
                    "fields": list(patch_fields),
                    "from": str(fb_row.get("results_path", "")),
                }
            )
            continue

        row = cur_by[key]
        patched: List[str] = []
        for field in patch_fields:
            if _is_num(row.get(field)):
                continue
            if not _is_num(fb_row.get(field)):
                continue
            row[field] = fb_row[field]
            patched.append(field)
        if patched:
            row.setdefault("data_source", "current")
            row["fallback_fields"] = ",".join(patched)
            provenance.append(
                {
                    "model": model,
                    "task": task,
                    "method_group": group,
                    "kind": "fields",
                    "fields": patched,
                    "from": str(fb_row.get("results_path", "")),
                }
            )

    out = pd.DataFrame(cur_by.values())
    if "data_source" not in out.columns:
        out["data_source"] = "current"
    out.loc[out["data_source"].isna(), "data_source"] = "current"
    return out, provenance


def resolve_matrix_tasks(
    df: pd.DataFrame,
    *,
    model: str,
    tasks_arg: Optional[str],
    disk_tasks: Optional[Sequence[str]] = None,
) -> List[str]:
    available = set()
    if df is not None and not df.empty and "model" in df.columns:
        available.update(df[df["model"] == model]["task"].astype(str).unique())
    if disk_tasks:
        available.update(str(t) for t in disk_tasks)
    if tasks_arg and tasks_arg.lower() not in {"default", "all"}:
        wanted = [t.strip() for t in tasks_arg.split(",") if t.strip()]
        hidden = [t for t in wanted if _is_hidden_task(t)]
        if hidden:
            print(f"Warning: hidden tasks skipped: {', '.join(hidden)}", file=sys.stderr)
        wanted = [t for t in wanted if not _is_hidden_task(t)]
        missing = [t for t in wanted if t not in available]
        if missing:
            print(f"Warning: tasks not in data (skipped): {', '.join(missing)}", file=sys.stderr)
        return _order_tasks([t for t in wanted if t in available])
    return _order_tasks([t for t in PAPER_TASK_ORDER if t in available and not _is_hidden_task(t)])


def resolve_summary_tasks(df: pd.DataFrame, *, model: str) -> List[str]:
    available = set(df[df["model"] == model]["task"].astype(str).unique())
    return _order_tasks([t for t in SUMMARY_TASKS if t in available])


def _format_num(value: Any, *, decimals: int = 3) -> str:
    if not _is_num(value):
        return MISSING_DISPLAY
    return f"{float(value):.{decimals}f}"


def _cell_for_pivot(
    value: Any,
    *,
    expected: bool,
    decimals: int = 3,
    bold: bool = False,
) -> str:
    if not _is_num(value):
        text = GAP_DISPLAY if not expected else MISSING_DISPLAY
    else:
        text = _format_num(value, decimals=decimals)
    return rf"\textbf{{{text}}}" if bold else text


def row_winner_methods(
    matrix: pd.DataFrame,
    task: str,
    methods: Sequence[str],
    *,
    eps: float = WIN_EPS,
) -> set[str]:
    """Method columns within ``eps`` of the row maximum (higher is better)."""
    scored: List[Tuple[str, float]] = []
    for method in methods:
        if task not in matrix.index or method not in matrix.columns:
            continue
        val = matrix.loc[task, method]
        if _is_num(val):
            scored.append((method, float(val)))
    if not scored:
        return set()
    top = max(v for _, v in scored)
    return {method for method, val in scored if abs(val - top) <= eps}


def methods_with_data(
    df: pd.DataFrame,
    *,
    model: str,
    tasks: Sequence[str],
    methods: Sequence[str],
    metric: str,
) -> List[str]:
    """Return method columns that have at least one numeric value for ``metric``."""
    sub = df[
        (df["model"] == model)
        & (df["task"].isin(tasks))
        & (df["method_group"].isin(methods))
    ]
    has_data = {
        str(rec["method_group"])
        for rec in sub.to_dict(orient="records")
        if _is_num(rec.get(metric))
    }
    return [m for m in methods if m in has_data]


def build_task_method_matrix(
    df: pd.DataFrame,
    *,
    model: str,
    tasks: Sequence[str],
    methods: Sequence[str],
    metric: str,
    specs: Mapping[Tuple[str, str], Mapping[str, Any]],
) -> pd.DataFrame:
    """Tasks × methods numeric grid plus Mean row and Mean column."""
    sub = df[(df["model"] == model) & (df["task"].isin(tasks)) & (df["method_group"].isin(methods))]
    lookup: Dict[Tuple[str, str], float] = {}
    for rec in sub.to_dict(orient="records"):
        if _is_num(rec.get(metric)):
            lookup[(str(rec["task"]), str(rec["method_group"]))] = float(rec[metric])

    columns = list(methods) + [MEAN_LABEL]
    rows: List[Dict[str, Any]] = []
    for task in tasks:
        spec = resolve_spec(specs, model=model, task=task)
        rec: Dict[str, Any] = {"task": task}
        row_vals: List[float] = []
        for method in methods:
            expected = group_is_applicable(method, spec, metric=metric)
            val = lookup.get((task, method))
            if _is_num(val):
                rec[method] = float(val)
                row_vals.append(float(val))
            elif expected:
                rec[method] = float("nan")
            else:
                rec[method] = None
        rec[MEAN_LABEL] = _mean(row_vals)
        rows.append(rec)

    mean_rec: Dict[str, Any] = {"task": MEAN_LABEL}
    col_means: List[float] = []
    for method in methods:
        nums = [lookup[(t, method)] for t in tasks if (t, method) in lookup]
        m = _mean(nums)
        mean_rec[method] = m
        if _is_num(m):
            col_means.append(float(m))
    mean_rec[MEAN_LABEL] = _mean(col_means)
    rows.append(mean_rec)

    out = pd.DataFrame(rows).set_index("task")
    return out.reindex(index=list(tasks) + [MEAN_LABEL], columns=columns)


def format_metric_matrix_latex(
    matrix: pd.DataFrame,
    *,
    metric: str,
    model: str,
    tasks: Sequence[str],
    methods: Sequence[str],
    specs: Mapping[Tuple[str, str], Mapping[str, Any]],
    label: Optional[str] = None,
    caption: Optional[str] = None,
    method_set: str = "selected",
) -> str:
    """Methods × tasks LaTeX table with compact, rotated task headers."""
    cap = caption or _metric_label(metric)
    label = label or f"tab:summary-{_metric_slug(metric)}-{model.replace('-', '')}"
    shown_tasks = list(tasks)
    header = "Method & " + " & ".join(
        rf"\rotatebox{{45}}{{{_task_label(task)}}}" for task in shown_tasks
    ) + r" \\"

    lines = [
        r"\begin{table*}[t]",
        r"  \centering",
        r"  \scriptsize",
        r"  \setlength{\tabcolsep}{3pt}",
        r"  \renewcommand{\arraystretch}{1.05}",
        f"  \\caption{{{cap}}}",
        f"  \\label{{{label}}}",
        f"  \\begin{{tabular}}{{l*{{{len(shown_tasks)}}}{{c}}}}",
        r"    \toprule",
        f"    {header}",
        r"    \midrule",
    ]

    rendered_methods = list(methods)
    winners_by_task = {
        task: row_winner_methods(matrix, task, rendered_methods) for task in shown_tasks
    }
    for method_index, method in enumerate(rendered_methods):
        cells = [_method_label(method, method_set=method_set)]
        for task in shown_tasks:
            spec = resolve_spec(specs, model=model, task=task)
            val = matrix.loc[task, method] if task in matrix.index and method in matrix.columns else None
            cells.append(
                _cell_for_pivot(
                    val,
                    expected=group_is_applicable(method, spec, metric=metric),
                    decimals=3,
                    bold=method in winners_by_task[task],
                )
            )
        lines.append("    " + " & ".join(cells) + r" \\")
        separator = _method_separator(rendered_methods, method_index)
        if separator == "signal":
            lines.append(rf"    \SummarySignalRule{{{1 + len(shown_tasks)}}}")
        elif separator == "method":
            lines.append(rf"    \SummaryMethodRule{{{1 + len(shown_tasks)}}}")

    lines.extend(
        [
            r"    \bottomrule",
            r"  \end{tabular}",
            r"\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def format_concatenated_metric_matrices_latex(
    matrices: Mapping[str, pd.DataFrame],
    *,
    metrics: Sequence[str],
    model: str,
    tasks: Sequence[str],
    methods: Sequence[str],
    specs: Mapping[Tuple[str, str], Mapping[str, Any]],
    label: str,
    caption: str,
    method_set: str = "selected",
) -> str:
    """One raw-results matrix with a labelled block for each related metric."""
    shown_tasks = list(tasks)
    header = "Method & " + " & ".join(
        rf"\rotatebox{{60}}{{{_task_label(task)}}}" for task in shown_tasks
    ) + r" \\"
    span = 1 + len(shown_tasks)
    lines = [
        r"\begin{table*}[!t]",
        r"  \centering",
        r"  \tiny",
        r"  \setlength{\tabcolsep}{1pt}",
        r"  \renewcommand{\arraystretch}{0.8}",
        rf"  \caption{{{caption}}}",
        rf"  \label{{{label}}}",
        rf"  \begin{{tabular}}{{l*{{{len(shown_tasks)}}}{{c}}}}",
        r"    \toprule",
        f"    {header}",
        r"    \midrule",
    ]
    for metric_index, metric in enumerate(metrics):
        matrix = matrices[metric]
        active_methods = [
            method
            for method in methods
            if method in matrix.columns
            and any(_is_num(matrix.loc[task, method]) for task in shown_tasks if task in matrix.index)
        ]
        if not active_methods:
            continue
        if metric_index:
            lines.append(r"    \midrule")
        lines.extend(
            [
                rf"    \multicolumn{{{span}}}{{c}}{{{_metric_label(metric)}}} \\",
                r"    \midrule",
            ]
        )
        winners_by_task = {
            task: row_winner_methods(matrix, task, active_methods) for task in shown_tasks
        }
        for method_index, method in enumerate(active_methods):
            cells = [_method_label(method, method_set=method_set)]
            for task in shown_tasks:
                spec = resolve_spec(specs, model=model, task=task)
                value = matrix.loc[task, method] if task in matrix.index else None
                cells.append(
                    _cell_for_pivot(
                        value,
                        expected=group_is_applicable(method, spec, metric=metric),
                        decimals=3,
                        bold=method in winners_by_task[task],
                    )
                )
            lines.append("    " + " & ".join(cells) + r" \\")
            separator = _method_separator(active_methods, method_index)
            if separator == "signal":
                lines.append(rf"    \SummarySignalRule{{{span}}}")
            elif separator == "method":
                lines.append(rf"    \SummaryMethodRule{{{span}}}")
    lines.extend(
        [
            r"    \bottomrule",
            r"  \end{tabular}",
            r"\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def build_method_metric_summary(
    df: pd.DataFrame,
    *,
    model: str,
    tasks: Sequence[str],
    methods: Sequence[str],
    metrics: Sequence[str],
    aggregation: str = "mean",
    specs: Optional[Mapping[Tuple[str, str], Mapping[str, Any]]] = None,
) -> pd.DataFrame:
    """Methods × metrics table: selected task aggregation per cell."""
    sub = df[(df["model"] == model) & (df["task"].isin(tasks)) & (df["method_group"].isin(methods))]
    lookup: Dict[Tuple[str, str, str], float] = {}
    for rec in sub.to_dict(orient="records"):
        group = str(rec["method_group"])
        task = str(rec["task"])
        for metric in metrics:
            spec = resolve_spec(specs, model=model, task=task) if specs is not None else None
            if (
                _is_num(rec.get(metric))
                and (spec is None or group_is_applicable(group, spec, metric=metric))
            ):
                lookup[(group, task, metric)] = float(rec[metric])

    rows: List[Dict[str, Any]] = []
    for method in methods:
        rec: Dict[str, Any] = {"method_group": method}
        for metric in metrics:
            vals = [lookup[(method, t, metric)] for t in tasks if (method, t, metric) in lookup]
            rec[metric] = _aggregate(vals, aggregation)
        rows.append(rec)

    out = pd.DataFrame(rows).set_index("method_group")
    return out.reindex(index=list(methods), columns=list(metrics))


def build_method_metric_completeness(
    df: pd.DataFrame,
    *,
    model: str,
    tasks: Sequence[str],
    methods: Sequence[str],
    metrics: Sequence[str],
    specs: Mapping[Tuple[str, str], Mapping[str, Any]],
) -> pd.DataFrame:
    """Whether every applicable task/raw constituent contributes to a mean."""
    sub = df[
        (df["model"] == model)
        & (df["task"].isin(tasks))
        & (df["method_group"].isin(methods))
    ]
    lookup = {
        (str(record["method_group"]), str(record["task"])): record
        for record in sub.to_dict(orient="records")
    }
    rows: List[Dict[str, Any]] = []
    for method in methods:
        record: Dict[str, Any] = {"method_group": method}
        for metric in metrics:
            expected_tasks = [
                task
                for task in tasks
                if group_is_applicable(
                    method,
                    resolve_spec(specs, model=model, task=task),
                    metric=metric,
                )
            ]
            complete = bool(expected_tasks)
            for task in expected_tasks:
                row = lookup.get((method, str(task)))
                if row is None or not _is_num(row.get(metric)):
                    complete = False
                    continue
                if metric == "detection_score" and not bool(row.get("detection_complete")):
                    complete = False
                if metric == "intervention_score" and not bool(row.get("intervention_complete")):
                    complete = False
            record[metric] = complete
        rows.append(record)
    return pd.DataFrame(rows).set_index("method_group").reindex(
        index=list(methods), columns=list(metrics)
    )


def column_winner_methods(
    summary: pd.DataFrame,
    metric: str,
    methods: Sequence[str],
    *,
    eps: float = WIN_EPS,
) -> set[str]:
    """Method rows within ``eps`` of the column maximum (higher is better)."""
    scored: List[Tuple[str, float]] = []
    for method in methods:
        if method not in summary.index or metric not in summary.columns:
            continue
        val = summary.loc[method, metric]
        if _is_num(val):
            scored.append((method, float(val)))
    if not scored:
        return set()
    top = max(v for _, v in scored)
    return {method for method, val in scored if abs(val - top) <= eps}


def column_winner_methods_min(
    summary: pd.DataFrame,
    metric: str,
    methods: Sequence[str],
    *,
    eps: float = WIN_EPS,
) -> set[str]:
    """Method rows within ``eps`` of the column minimum (lower is better, e.g. rank tables)."""
    scored: List[Tuple[str, float]] = []
    for method in methods:
        if method not in summary.index or metric not in summary.columns:
            continue
        val = summary.loc[method, metric]
        if _is_num(val):
            scored.append((method, float(val)))
    if not scored:
        return set()
    top = min(v for _, v in scored)
    return {method for method, val in scored if abs(val - top) <= eps}


def compute_task_metric_ranks(
    df: pd.DataFrame,
    *,
    model: str,
    tasks: Sequence[str],
    methods: Sequence[str],
    metrics: Sequence[str],
    specs: Optional[Mapping[Tuple[str, str], Mapping[str, Any]]] = None,
) -> Dict[Tuple[str, str], Dict[str, float]]:
    """Per (task, metric): proportional placement of each method with data,
    in ``[0, 1]`` (1 = best/highest value that task, 0 = worst).

    Ranking is restricted to methods that actually have a numeric value for
    that (task, metric) — a method with no data on a task (e.g. a two-pole
    method on a one-pole-only circuit task, or vice versa) neither earns nor
    drags down a rank there, and does not count toward the competing field.
    Ties (within ``WIN_EPS``) share the average of their positions.

    The raw 1-based rank is then normalized by field size,
    ``(n - rank) / (n - 1)`` (1 when there's only one competitor), rather
    than averaged as a raw rank number. One-pole and two-pole methods
    routinely compete against different-sized fields from task to task
    (one-pole-only circuit tasks have no two-pole columns at all, and not
    every baseline has data on every task) — a raw rank of "2" means
    something very different in a field of 2 versus a field of 13, so mixing
    raw ranks across tasks would itself be an apples-to-oranges average.
    Proportional placement keeps every task's contribution on the same 0..1 scale
    regardless of how many methods happened to compete on it.
    """
    sub = df[(df["model"] == model) & (df["task"].isin(tasks)) & (df["method_group"].isin(methods))]
    lookup: Dict[Tuple[str, str, str], float] = {}
    for rec in sub.to_dict(orient="records"):
        group = str(rec["method_group"])
        task = str(rec["task"])
        for metric in metrics:
            spec = resolve_spec(specs, model=model, task=task) if specs is not None else None
            if (
                _is_num(rec.get(metric))
                and (spec is None or group_is_applicable(group, spec, metric=metric))
            ):
                lookup[(task, metric, group)] = float(rec[metric])

    ranks: Dict[Tuple[str, str], Dict[str, float]] = {}
    for task in tasks:
        for metric in metrics:
            vals = [(m, lookup[(task, metric, m)]) for m in methods if (task, metric, m) in lookup]
            if not vals:
                continue
            ordered = sorted(vals, key=lambda item: -item[1])
            n = len(ordered)
            rank_by_method: Dict[str, float] = {}
            i = 0
            while i < n:
                j = i
                while j + 1 < n and abs(ordered[j + 1][1] - ordered[i][1]) <= WIN_EPS:
                    j += 1
                avg_rank = (i + 1 + j + 1) / 2.0
                proportional = (n - avg_rank) / (n - 1.0) if n > 1 else 1.0
                for k in range(i, j + 1):
                    rank_by_method[ordered[k][0]] = proportional
                i = j + 1
            ranks[(task, metric)] = rank_by_method
    return ranks


def build_method_metric_rank_summary(
    df: pd.DataFrame,
    *,
    model: str,
    tasks: Sequence[str],
    methods: Sequence[str],
    metrics: Sequence[str],
    aggregation: str = "mean",
    specs: Optional[Mapping[Tuple[str, str], Mapping[str, Any]]] = None,
) -> pd.DataFrame:
    """Methods × metrics table: mean proportional placement per cell over
    ``tasks`` (1 = always best of its competing field, 0 = always worst).

    Outlier-robust companion to ``build_method_metric_summary``'s raw mean —
    a rank cannot be dragged around by one task's unusually large/small
    metric value the way a mean can — and, via the proportional normalization
    in ``compute_task_metric_ranks``, comparable across methods (e.g.
    one-pole vs two-pole) that don't all compete against the same number of
    other methods on every task.
    """
    ranks = compute_task_metric_ranks(
        df,
        model=model,
        tasks=tasks,
        methods=methods,
        metrics=metrics,
        specs=specs,
    )
    rows: List[Dict[str, Any]] = []
    for method in methods:
        rec: Dict[str, Any] = {"method_group": method}
        for metric in metrics:
            vals = [
                ranks[(task, metric)][method]
                for task in tasks
                if (task, metric) in ranks and method in ranks[(task, metric)]
            ]
            rec[metric] = _aggregate(vals, aggregation)
        rows.append(rec)

    out = pd.DataFrame(rows).set_index("method_group")
    return out.reindex(index=list(methods), columns=list(metrics))


def methods_with_any_summary_data(
    summary: pd.DataFrame,
    methods: Sequence[str],
    metrics: Sequence[str],
) -> List[str]:
    active: List[str] = []
    for method in methods:
        if method not in summary.index:
            continue
        if any(_is_num(summary.loc[method, metric]) for metric in metrics if metric in summary.columns):
            active.append(method)
    return active


def methods_with_encoder_summary_data(
    summary: pd.DataFrame,
    methods: Sequence[str],
) -> List[str]:
    """Methods eligible for the mixed encoder-summary table.

    Causal-only baselines (currently ACTIEND ridge) belong in the dedicated
    causal matrices, not in an encoder table with blank encoder columns.
    """
    return methods_with_any_summary_data(summary, methods, ENCODER_SUMMARY_FIELDS)


def format_method_summary_latex(
    summary: pd.DataFrame,
    *,
    model: str,
    methods: Sequence[str],
    metrics: Sequence[str],
    label: Optional[str] = None,
    caption: str = "Summary",
    decimals: int = 3,
    lower_is_better: bool = False,
    completeness: Optional[pd.DataFrame] = None,
    aggregation: str = "mean",
    method_set: str = "selected",
) -> str:
    """Methods × metrics LaTeX table.

    ``lower_is_better=True`` bolds the column *minimum* instead of the
    maximum for explicitly lower-is-better tables.
    """
    cap = caption
    label = label or f"tab:summary-methods-{model.replace('-', '')}"
    col_headers = [_metric_label(m) for m in metrics]
    cols = "l" + "r" * len(col_headers)
    header = "Method & " + " & ".join(col_headers) + r" \\"

    grouped_header: List[str] = []
    if tuple(metrics) == MATRIX_METRICS:
        det_end = 1 + len(DETECTION_FIELDS)
        int_start = det_end + 1
        int_end = det_end + len(INTERVENTION_FIELDS)
        grouped_header = [
            "    & "
            + rf"\multicolumn{{{len(DETECTION_FIELDS)}}}{{c}}{{Detection}} & "
            + rf"\multicolumn{{{len(INTERVENTION_FIELDS)}}}{{c}}{{Intervention}} \\"
            ,
            rf"    \cmidrule(lr){{2-{det_end}}}\cmidrule(lr){{{int_start}-{int_end}}}",
        ]

    lines = [
        r"\begin{table}[!t]",
        r"  \centering",
        r"  \small",
        f"  \\caption{{{cap}}}",
        f"  \\label{{{label}}}",
        f"  \\begin{{tabular}}{{{cols}}}",
        r"    \toprule",
        *grouped_header,
        f"    {header}",
        r"    \midrule",
    ]

    winner_fn = column_winner_methods_min if lower_is_better else column_winner_methods
    for method_index, method in enumerate(methods):
        cells = [_method_label(method, method_set=method_set)]
        for metric in metrics:
            val = summary.loc[method, metric] if method in summary.index and metric in summary.columns else None
            winners = winner_fn(summary, metric, methods)
            rendered = _cell_for_pivot(
                val, expected=True, decimals=decimals, bold=method in winners
            )
            if (
                _is_num(val)
                and completeness is not None
                and method in completeness.index
                and metric in completeness.columns
                and not bool(completeness.loc[method, metric])
            ):
                if rendered.startswith(r"\textbf{"):
                    rendered = rendered[:-1] + r"\textsuperscript{*}}"
                else:
                    rendered += r"\textsuperscript{*}"
            cells.append(rendered)
        lines.append("    " + " & ".join(cells) + r" \\")
        separator = _method_separator(methods, method_index)
        if separator == "signal":
            lines.append(rf"    \SummarySignalRule{{{1 + len(metrics)}}}")
        elif separator == "method":
            lines.append(rf"    \SummaryMethodRule{{{1 + len(metrics)}}}")

    lines.extend(
        [
            r"    \bottomrule",
            r"  \end{tabular}",
            *(
                [
                    rf"  \par\smallskip\footnotesize \textsuperscript{{*}}Incomplete task coverage or missing raw constituent; {aggregation} uses available values."
                ]
                if (
                    completeness is not None
                    and any(
                        not bool(completeness.loc[method, metric])
                        for method in methods
                        if method in completeness.index
                        for metric in metrics
                        if metric in completeness.columns
                    )
                )
                else []
            ),
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)




def _load_cross_model_sources(
    runs_root: Path,
    sources: Sequence[Tuple[str, str]],
) -> Tuple[
    Dict[str, pd.DataFrame],
    Dict[str, Dict[Tuple[str, str], Mapping[str, Any]]],
    Dict[str, str],
]:
    model_frames: Dict[str, pd.DataFrame] = {}
    model_specs: Dict[str, Dict[Tuple[str, str], Mapping[str, Any]]] = {}
    model_suites: Dict[str, str] = {}
    for model, subdir in sources:
        frame = collect_canonical_group_rows(runs_root, model, subdir=subdir)
        model_frames[model] = frame
        # Cross-model completeness is defined against the declared paper suite,
        # so task/suite YAML is sufficient; scanning every large results.json a
        # second time merely to rebuild specs made this command needlessly slow.
        model_specs[model] = {}
        suites = (
            [str(value) for value in frame["suite"].dropna().unique()]
            if not frame.empty and "suite" in frame.columns
            else []
        )
        model_suites[model] = suites[0] if suites else (
            "full_plus" if subdir == "suite_full2" else "core"
        )
    return model_frames, model_specs, model_suites


def build_cross_model_headline_summary(
    runs_root: Path,
    *,
    metric: str,
    sources: Sequence[Tuple[str, str]] = DEFAULT_CROSS_MODEL_SOURCES,
    methods: Sequence[str] = PAPER_METHOD_GROUPS,
    tasks: Sequence[str] = PAPER_TASK_ORDER,
    aggregation: str = "mean",
    loaded_sources: Optional[
        Tuple[
            Dict[str, pd.DataFrame],
            Dict[str, Dict[Tuple[str, str], Mapping[str, Any]]],
            Dict[str, str],
        ]
    ] = None,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Methods × models task aggregates for one headline metric.

    A model cell is complete only when every applicable paper task has a value
    and every contributing group row contains all raw constituents of the
    headline.  The returned coverage records drive the table's superscript
    stars and make that status machine-readable.
    """
    if metric not in HEADLINE_METRICS:
        raise ValueError(f"cross-model metric must be one of {HEADLINE_METRICS}, got {metric!r}")

    visible_tasks = [str(task) for task in tasks if not _is_hidden_task(str(task))]
    complete_field = "detection_complete" if metric == "detection_score" else "intervention_complete"
    model_frames, model_specs, model_suites = loaded_sources or _load_cross_model_sources(
        runs_root, sources
    )

    values: Dict[Tuple[str, str], float] = {}
    coverage: List[Dict[str, Any]] = []
    applicable_by_cell: Dict[Tuple[str, str], bool] = {}
    complete_by_cell: Dict[Tuple[str, str], bool] = {}

    for method in methods:
        for model, subdir in sources:
            specs = model_specs[model]
            suite = model_suites[model]
            expected_tasks: List[str] = []
            for task in visible_tasks:
                spec = specs.get((str(model), str(task)))
                if not spec:
                    spec = spec_for_task(task, suite=suite, model=model)
                if group_is_applicable(method, spec, metric=metric):
                    expected_tasks.append(task)

            applicable = bool(expected_tasks)
            applicable_by_cell[(method, model)] = applicable
            frame = model_frames[model]
            rows_by_task: Dict[str, Mapping[str, Any]] = {}
            if not frame.empty and {"method_group", "task"}.issubset(frame.columns):
                selected = frame[
                    (frame["method_group"] == method)
                    & (frame["task"].isin(expected_tasks))
                ]
                rows_by_task = {
                    str(record["task"]): record
                    for record in selected.to_dict(orient="records")
                }

            nums: List[float] = []
            raw_complete = True
            for task in expected_tasks:
                record = rows_by_task.get(task)
                if record is None or not _is_num(record.get(metric)):
                    continue
                nums.append(float(record[metric]))
                if record.get(complete_field) is not True:
                    raw_complete = False

            complete = applicable and len(nums) == len(expected_tasks) and raw_complete
            complete_by_cell[(method, model)] = complete
            if nums:
                values[(method, model)] = _aggregate(nums, aggregation)
            coverage.append(
                {
                    "metric": metric,
                    "method_group": method,
                    "model": model,
                    "subdir": subdir or None,
                    "suite": suite,
                    "applicable": applicable,
                    "n_available": len(nums),
                    "n_expected": len(expected_tasks),
                    "complete": complete,
                }
            )

    model_names = [model for model, _subdir in sources]
    active_methods = [
        method
        for method in methods
        if any((method, model) in values for model in model_names)
    ]
    rows: List[Dict[str, Any]] = []
    for method in active_methods:
        record: Dict[str, Any] = {"method_group": method}
        for model in model_names:
            record[model] = values.get((method, model), float("nan"))
        applicable_models = [
            model for model in model_names if applicable_by_cell.get((method, model), False)
        ]
        model_values = [
            values[(method, model)]
            for model in applicable_models
            if (method, model) in values
        ]
        aggregate_label = aggregation.capitalize()
        record[aggregate_label] = _aggregate(model_values, aggregation)
        # The final column is explicitly an across-*all*-models mean.  A method
        # absent from a lean suite is a structural dash in that model column,
        # but the resulting available-model mean is still partial and starred.
        mean_complete = len(model_values) == len(model_names) and all(
            complete_by_cell.get((method, model), False) for model in model_names
        )
        complete_by_cell[(method, aggregate_label)] = mean_complete
        applicable_by_cell[(method, aggregate_label)] = bool(applicable_models)
        coverage.append(
            {
                "metric": metric,
                "method_group": method,
                "model": aggregate_label,
                "subdir": None,
                "suite": None,
                "applicable": bool(applicable_models),
                "n_available": len(model_values),
                "n_expected": len(model_names),
                "complete": mean_complete,
            }
        )
        rows.append(record)

    frame = pd.DataFrame(rows).set_index("method_group") if rows else pd.DataFrame()
    if not frame.empty:
        frame = frame.reindex(index=active_methods, columns=model_names + [aggregation.capitalize()])

    for item in coverage:
        key = (str(item["method_group"]), str(item["model"]))
        aggregate_label = aggregation.capitalize()
        item["value"] = values.get(key) if key[1] != aggregate_label else (
            frame.loc[key[0], aggregate_label] if not frame.empty and key[0] in frame.index else None
        )
    return frame, coverage


def build_cross_model_submetric_summary(
    frames: Mapping[str, pd.DataFrame],
    *,
    methods: Sequence[str],
    aggregation: str = "mean",
) -> pd.DataFrame:
    """Pool every available task--model cell for each headline submetric."""
    if aggregation not in AGGREGATIONS:
        raise ValueError(f"unknown aggregation {aggregation!r}")
    combined = pd.concat(list(frames.values()), ignore_index=True) if frames else pd.DataFrame()
    summary = pd.DataFrame(index=list(methods), columns=list(MATRIX_METRICS), dtype=float)
    if combined.empty:
        return summary
    for method in methods:
        rows = combined[combined["method_group"].astype(str).eq(str(method))]
        for metric in MATRIX_METRICS:
            if metric not in rows:
                continue
            values = pd.to_numeric(rows[metric], errors="coerce").dropna()
            if not values.empty:
                summary.loc[method, metric] = float(
                    values.mean() if aggregation == "mean" else values.median()
                )
    return summary


def cross_model_completion_rates(
    runs_root: Path,
    *,
    sources: Sequence[Tuple[str, str]],
    methods: Sequence[str],
    loaded_sources: Optional[
        Tuple[
            Dict[str, pd.DataFrame],
            Dict[str, Dict[Tuple[str, str], Mapping[str, Any]]],
            Dict[str, str],
        ]
    ] = None,
) -> Dict[str, float]:
    """Return task-weighted headline coverage for every requested model.

    This is deliberately a model-level gate rather than a per-method gate:
    the restricted headline is an equally weighted mean over models whose
    *overall* applicable headline task inventory is at least complete enough.
    A missing raw constituent counts as incomplete through the per-cell
    ``complete`` flag, even when a numerical value happens to be present.
    """
    loaded = loaded_sources or _load_cross_model_sources(runs_root, sources)
    covered: Dict[str, int] = {model: 0 for model, _ in sources}
    expected: Dict[str, int] = {model: 0 for model, _ in sources}
    for metric in HEADLINE_METRICS:
        _summary, coverage = build_cross_model_headline_summary(
            runs_root,
            metric=metric,
            sources=sources,
            methods=methods,
            loaded_sources=loaded,
        )
        for row in coverage:
            model = str(row.get("model"))
            if model not in expected or not row.get("applicable"):
                continue
            # A task is usable only when every raw constituent of its headline
            # score survived; this keeps a partial causal row from qualifying a
            # model merely because it left behind a scalar.
            expected[model] += int(row.get("n_expected") or 0)
            if row.get("complete"):
                covered[model] += int(row.get("n_expected") or 0)
    return {
        model: (covered[model] / expected[model] if expected[model] else 0.0)
        for model, _ in sources
    }


def format_cross_model_headline_latex(
    summary: pd.DataFrame,
    coverage: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    sources: Sequence[Tuple[str, str]] = DEFAULT_CROSS_MODEL_SOURCES,
    decimals: int = 3,
    aggregation: str = "mean",
    method_set: str = "selected",
    min_model_completion: Optional[float] = None,
) -> str:
    """Format one headline metric as a methods × models LaTeX table."""
    model_names = [model for model, _subdir in sources]
    aggregate_label = aggregation.capitalize()
    columns = model_names + [aggregate_label]
    meta = {
        (str(row.get("method_group")), str(row.get("model"))): row
        for row in coverage
        if str(row.get("metric")) == metric
    }
    label_slug = _metric_slug(metric)
    metric_name = "Detection" if metric == "detection_score" else "Intervention"
    formula = (
        r"$\mathrm{Det.}=\min(\mathrm{AUC}_n,\mathrm{Spec}_n,\mathrm{AUC}_o,\mathrm{Excl.})$"
        if metric == "detection_score"
        else r"$\mathrm{Int.}=(\Delta P^{+}+\Delta P^{-})/2$"
    )

    def cell(method: str, column: str, winners: set[str]) -> str:
        info = meta.get((method, column), {})
        if not info.get("applicable", True):
            return GAP_DISPLAY
        value = summary.loc[method, column]
        if not _is_num(value):
            return MISSING_DISPLAY
        body = f"{float(value):.{decimals}f}"
        if info.get("complete") is not True:
            body += r"\textsuperscript{*}"
        return rf"\textbf{{{body}}}" if method in winners else body

    coverage_caption = (
        # Percent signs are special in LaTeX captions; escape the formatted
        # completion threshold before inserting it into the generated table.
        f"; models with at least {min_model_completion:.0%}".replace("%", r"\%")
        + " overall headline coverage"
        if min_model_completion is not None
        else ""
    )
    lines = [
        r"\begin{table}[!t]",
        r"  \centering",
        r"  \small",
        f"  \\caption{{{metric_name} headline by model ({aggregation} aggregation{coverage_caption}; {formula}).}}",
        f"  \\label{{tab:summary-across-models-{label_slug}{_aggregation_suffix(aggregation).replace('_', '-')}}}",
        f"  \\begin{{tabular}}{{l{'r' * len(columns)}}}",
        r"    \toprule",
        "    Method & "
        + " & ".join(MODEL_LATEX.get(column, column) for column in model_names)
        + f" & {aggregate_label}"
        + r" \\",
        r"    \midrule",
    ]
    rendered_methods = list(summary.index.astype(str))
    for method_index, method in enumerate(rendered_methods):
        cells = [_method_label(method, method_set=method_set)]
        for column in columns:
            numeric = [
                (other, float(summary.loc[other, column]))
                for other in summary.index.astype(str)
                if _is_num(summary.loc[other, column])
            ]
            top = max((value for _other, value in numeric), default=float("nan"))
            winners = {
                other for other, value in numeric if _is_num(top) and abs(value - top) <= WIN_EPS
            }
            cells.append(cell(method, column, winners))
        lines.append("    " + " & ".join(cells) + r" \\")
        separator = _method_separator(rendered_methods, method_index)
        if separator == "signal":
            lines.append(rf"    \SummarySignalRule{{{1 + len(columns)}}}")
        elif separator == "method":
            lines.append(rf"    \SummaryMethodRule{{{1 + len(columns)}}}")
    lines.extend(
        [
            r"    \bottomrule",
            r"  \end{tabular}",
            rf"  \par\smallskip\footnotesize \textsuperscript{{*}}Incomplete: a model {aggregation} uses fewer than all applicable paper tasks or a contributing headline lacks a raw constituent; an across-model {aggregation} uses fewer than all paper models. Available models are weighted equally.",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def write_cross_model_headline_summaries(
    runs_root: Path,
    out_dir: Path,
    *,
    sources: Sequence[Tuple[str, str]] = DEFAULT_CROSS_MODEL_SOURCES,
    aggregation: str = "mean",
    method_set: str = "selected",
    min_model_completion: Optional[float] = None,
    loaded_sources: Optional[
        Tuple[
            Dict[str, pd.DataFrame],
            Dict[str, Dict[Tuple[str, str], Mapping[str, Any]]],
            Dict[str, str],
        ]
    ] = None,
) -> List[Path]:
    """Write the two paper headline tables plus numeric/coverage companions."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = [write_summary_table_commands(out_dir)]
    all_coverage: List[Dict[str, Any]] = []
    loaded_sources = loaded_sources or _load_cross_model_sources(runs_root, sources)
    frames = loaded_sources[0]
    combined = pd.concat(list(frames.values()), ignore_index=True) if frames else pd.DataFrame()
    methods = headline_method_groups(method_set, frame=combined)
    completion_rates = cross_model_completion_rates(
        runs_root,
        sources=sources,
        methods=methods,
        loaded_sources=loaded_sources,
    )
    selected_sources = tuple(
        (model, subdir)
        for model, subdir in sources
        if min_model_completion is None or completion_rates[model] >= min_model_completion
    )
    if not selected_sources:
        # Sparse in-progress sweeps are expected.  The ordinary all-model
        # artifact remains useful; there simply is no scientifically honest
        # >=90%-complete companion until at least one model qualifies.
        print(
            f"Skipping >= {min_model_completion:.0%} headline: no eligible model "
            + "(" + ", ".join(
                f"{model}={rate:.1%}" for model, rate in completion_rates.items()
            ) + ")."
        )
        return []
    if min_model_completion is not None:
        selected_names = {model for model, _subdir in selected_sources}
        loaded_sources = (
            {model: frame for model, frame in frames.items() if model in selected_names},
            {model: spec for model, spec in loaded_sources[1].items() if model in selected_names},
            {model: suite for model, suite in loaded_sources[2].items() if model in selected_names},
        )
        frames = loaded_sources[0]
        combined = pd.concat(list(frames.values()), ignore_index=True) if frames else pd.DataFrame()
        methods = headline_method_groups(method_set, frame=combined)
    coverage_suffix = "_complete90" if min_model_completion == 0.9 else (
        f"_complete{int(round(min_model_completion * 100))}" if min_model_completion is not None else ""
    )
    for metric in HEADLINE_METRICS:
        summary, coverage = build_cross_model_headline_summary(
            runs_root,
            metric=metric,
            sources=selected_sources,
            methods=methods,
            loaded_sources=loaded_sources,
            aggregation=aggregation,
        )
        slug = _metric_slug(metric)
        suffix = _aggregation_suffix(aggregation)
        method_suffix = "" if method_set == "selected" else "_full"
        tex_path = out_dir / f"summary_across_models_{slug}{suffix}{coverage_suffix}{method_suffix}.tex"
        csv_path = out_dir / f"summary_across_models_{slug}{suffix}{coverage_suffix}{method_suffix}.csv"
        tex_path.write_text(
            format_cross_model_headline_latex(
                summary,
                coverage,
                metric=metric,
                sources=selected_sources,
                aggregation=aggregation,
                method_set=method_set,
                min_model_completion=min_model_completion,
            ),
            encoding="utf-8",
        )
        summary.to_csv(csv_path, float_format="%.4f")
        written.extend([tex_path, csv_path])
        all_coverage.extend(dict(row) for row in coverage)
    method_suffix = "" if method_set == "selected" else "_full"
    suffix = _aggregation_suffix(aggregation)
    submetric_summary = build_cross_model_submetric_summary(
        frames, methods=methods, aggregation=aggregation
    )
    submetric_methods = methods_with_any_summary_data(
        submetric_summary, methods, MATRIX_METRICS
    )
    submetric_tex_path = out_dir / f"summary_across_models_submetrics{suffix}{coverage_suffix}{method_suffix}.tex"
    submetric_csv_path = out_dir / f"summary_across_models_submetrics{suffix}{coverage_suffix}{method_suffix}.csv"
    if submetric_methods:
        submetric_tex_path.write_text(
            format_method_summary_latex(
                submetric_summary,
                model="all-models",
                methods=submetric_methods,
                metrics=MATRIX_METRICS,
                label=f"tab:summary-across-models-submetrics{suffix.replace('_', '-')}",
                caption=(
                    f"{aggregation.capitalize()} submetrics pooled over all available "
                    "task--model cells"
                ),
                aggregation=aggregation,
                method_set=method_set,
            ),
            encoding="utf-8",
        )
        written.append(submetric_tex_path)
    elif submetric_tex_path.is_file():
        submetric_tex_path.unlink()
    submetric_summary.reindex(submetric_methods).to_csv(
        submetric_csv_path, float_format="%.4f"
    )
    written.append(submetric_csv_path)
    coverage_path = out_dir / f"summary_across_models_coverage{_aggregation_suffix(aggregation)}{coverage_suffix}{method_suffix}.csv"
    pd.DataFrame(all_coverage).to_csv(coverage_path, index=False)
    written.append(coverage_path)
    manifest_path = out_dir / f"summary_across_models_manifest{_aggregation_suffix(aggregation)}{coverage_suffix}{method_suffix}.json"
    manifest_path.write_text(
        json.dumps(
            {
                "sources": [
                    {"model": model, "subdir": subdir or None}
                    for model, subdir in selected_sources
                ],
                "model_completion_rates": completion_rates,
                "min_model_completion": min_model_completion,
                "metrics": list(HEADLINE_METRICS),
                "aggregation": aggregation,
                "incomplete_marker": "*",
                "method_set": method_set,
                "methods": list(methods),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    written.append(manifest_path)
    return written


def table_snippet_for_pdf(tex: str) -> str:
    """Pin snippets to pages and shrink only genuinely wide tabulars.

    Expanding every table to ``textwidth`` made method-summary tables taller
    than the page. Matrices with more than 12 columns still need scaling;
    narrower tables retain their natural dimensions.
    """
    body = tex.strip()
    body = re.sub(
        r"\\begin\{table(\*)?\}\[[^\]]*\]",
        lambda match: rf"\begin{{table{match.group(1) or ''}}}[p]",
        body,
        count=1,
    )
    tabular = re.search(r"\\begin\{tabular\}(\{([^}]+)\})", body)
    if tabular and sum(tabular.group(2).count(char) for char in "lcr") > 12:
        body = re.sub(
            r"\\begin\{tabular\}(\{[^}]+\})",
            r"\\resizebox{\\textwidth}{!}{%\n    \\begin{tabular}\1",
            body,
            count=1,
        )
        body = re.sub(r"\\end\{tabular\}", r"\\end{tabular}%\n  }", body, count=1)
    return body + "\n\\clearpage\n"


def write_summary_table_commands(out_dir: Path) -> Path:
    """Export the user-overridable rule macros used by method-row tables."""
    path = Path(out_dir) / "summary_table_commands.tex"
    path.write_text(
        "% Method-table separators. Override after \\input to restyle/disable.\n"
        + "\n".join(SUMMARY_TABLE_COMMAND_DEFINITIONS)
        + "\n"
        + r"% Disable: \renewcommand{\SummarySignalRule}[1]{}"
        + "\n"
        + r"% Disable: \renewcommand{\SummaryMethodRule}[1]{}"
        + "\n",
        encoding="utf-8",
    )
    return path


def write_result_table_view(
    df: pd.DataFrame,
    out_dir: Path,
    *,
    model: str,
    tasks: Sequence[str],
    methods: Sequence[str],
    specs: Mapping[Tuple[str, str], Mapping[str, Any]],
    view: str,
    aggregation: str = "mean",
    method_set: str = "selected",
) -> List[Path]:
    """Write every metric table for one construction-specific view."""
    view_tasks = result_view_tasks(tasks, view)
    view_methods = result_view_methods(methods, view)
    suffix = _view_suffix(view)
    aggregation_suffix = _aggregation_suffix(aggregation)
    summary_suffix = suffix + aggregation_suffix
    method_suffix = "" if method_set == "selected" else "_full"
    prefix = _view_caption_prefix(view)
    label_suffix = "" if view == "combined" else view.replace("_", "")
    written: List[Path] = []

    for metric in MATRIX_METRICS if aggregation == "mean" else ():
        slug = _metric_slug(metric)
        active_methods = methods_with_data(
            df,
            model=model,
            tasks=view_tasks,
            methods=view_methods,
            metric=metric,
        )
        if not active_methods:
            for extension in (".tex", ".csv"):
                stale = out_dir / f"summary_matrix_{slug}_{model}{suffix}{method_suffix}{extension}"
                if stale.is_file():
                    stale.unlink()
            continue
        matrix = build_task_method_matrix(
            df,
            model=model,
            tasks=view_tasks,
            methods=active_methods,
            metric=metric,
            specs=specs,
        )
        tex = format_metric_matrix_latex(
            matrix,
            metric=metric,
            model=model,
            tasks=view_tasks,
            methods=active_methods,
            specs=specs,
            label=(
                f"tab:summary-{slug}-{model.replace('-', '')}"
                if view == "combined"
                else f"tab:summary-{slug}-{model.replace('-', '')}-{view.replace('_', '')}"
            ),
            caption=f"{prefix}{_metric_label(metric)}" if prefix else None,
            method_set=method_set,
        )
        tex_path = out_dir / f"summary_matrix_{slug}_{model}{suffix}{method_suffix}.tex"
        csv_path = out_dir / f"summary_matrix_{slug}_{model}{suffix}{method_suffix}.csv"
        tex_path.write_text(tex, encoding="utf-8")
        matrix.to_csv(csv_path, float_format="%.4f")
        written.append(tex_path)

    summary = build_method_metric_summary(
        df,
        model=model,
        tasks=view_tasks,
        methods=view_methods,
        metrics=MATRIX_METRICS,
        aggregation=aggregation,
        specs=specs,
    )
    completeness = build_method_metric_completeness(
        df,
        model=model,
        tasks=view_tasks,
        methods=view_methods,
        metrics=MATRIX_METRICS,
        specs=specs,
    )
    summary_methods = methods_with_encoder_summary_data(summary, view_methods)
    summary_tex_path = out_dir / f"summary_methods_{model}{summary_suffix}{method_suffix}.tex"
    if summary_methods:
        summary_tex_path.write_text(
            format_method_summary_latex(
                summary,
                model=model,
                methods=summary_methods,
                metrics=MATRIX_METRICS,
                label=(
                    f"tab:summary-methods-{model.replace('-', '')}{label_suffix}"
                    f"{aggregation_suffix.replace('_', '-')}"
                ),
                caption=(
                    f"{prefix}{aggregation} summary"
                    if prefix
                    else f"{aggregation.capitalize()} summary"
                ),
                completeness=completeness,
                aggregation=aggregation,
                method_set=method_set,
            ),
            encoding="utf-8",
        )
        summary.reindex(summary_methods).to_csv(
            out_dir / f"summary_methods_{model}{summary_suffix}{method_suffix}.csv", float_format="%.4f"
        )
        completeness.reindex(summary_methods).to_csv(
            out_dir / f"summary_methods_completeness_{model}{summary_suffix}{method_suffix}.csv"
        )
        written.append(summary_tex_path)
    elif summary_tex_path.is_file():
        summary_tex_path.unlink()

    rank_summary = build_method_metric_rank_summary(
        df,
        model=model,
        tasks=view_tasks,
        methods=view_methods,
        metrics=MATRIX_METRICS,
        aggregation=aggregation,
        specs=specs,
    )
    rank_methods = methods_with_encoder_summary_data(rank_summary, view_methods)
    rank_tex_path = out_dir / f"summary_methods_rank_{model}{summary_suffix}{method_suffix}.tex"
    if rank_methods:
        rank_tex_path.write_text(
            format_method_summary_latex(
                rank_summary,
                model=model,
                methods=rank_methods,
                metrics=MATRIX_METRICS,
                label=(
                    f"tab:summary-methods-rank-{model.replace('-', '')}"
                    f"{label_suffix}{aggregation_suffix.replace('_', '-')}"
                ),
                caption=(
                    f"{prefix}summary ({aggregation} proportional placement; 1 = best)"
                    if prefix
                    else f"Summary ({aggregation} proportional placement; 1 = best)"
                ),
                decimals=2,
                completeness=completeness,
                aggregation=aggregation,
                method_set=method_set,
            ),
            encoding="utf-8",
        )
        rank_summary.reindex(rank_methods).to_csv(
            out_dir / f"summary_methods_rank_{model}{summary_suffix}{method_suffix}.csv",
            float_format="%.4f",
        )
        written.append(rank_tex_path)
    elif rank_tex_path.is_file():
        rank_tex_path.unlink()

    # Compact raw-result books: Detection has its five constituent metrics;
    # Intervention intentionally omits both LMS columns.
    if aggregation == "mean" and view == "combined":
        concatenated_sets = {
            "detection": DETECTION_FIELDS,
            "intervention": (
                "causal_signed_effect",
                "causal_signed_effect_weaken",
                "intervention_score",
            ),
        }
        for category, category_metrics in concatenated_sets.items():
            matrices = {
                metric: build_task_method_matrix(
                    df, model=model, tasks=view_tasks, methods=view_methods,
                    metric=metric, specs=specs,
                )
                for metric in category_metrics
            }
            path = out_dir / f"summary_concatenated_{category}_{model}{method_suffix}.tex"
            path.write_text(
                format_concatenated_metric_matrices_latex(
                    matrices, metrics=category_metrics, model=model, tasks=view_tasks,
                    methods=view_methods, specs=specs,
                    label=f"tab:summary-concatenated-{category}-{model.replace('-', '')}",
                    caption=(
                        f"{category.capitalize()} raw results"
                        + (" (LMS excluded)" if category == "intervention" else "")
                    ),
                    method_set=method_set,
                ),
                encoding="utf-8",
            )
            written.append(path)

    return written


def collect_snippet_paths(
    out_dir: Path,
    model: str,
    *,
    view: str = "combined",
    aggregation: str = "mean",
    method_set: str = "selected",
) -> List[Path]:
    """Snippet order: Summary, Summary (rank), convergence counts, one matrix
    per ``MATRIX_METRICS``, then the convergence-problems longtable last (it
    can run to dozens of rows and reads as a debugging appendix, not a
    headline table).
    """
    if view not in RESULT_TABLE_VIEWS:
        raise ValueError(f"unknown result table view: {view!r}")
    suffix = _view_suffix(view)
    summary_suffix = suffix + _aggregation_suffix(aggregation)
    method_suffix = "" if method_set == "selected" else "_full"
    paths: List[Path] = []
    methods = Path(out_dir) / f"summary_methods_{model}{summary_suffix}{method_suffix}.tex"
    if methods.is_file():
        paths.append(methods)
    rank = Path(out_dir) / f"summary_methods_rank_{model}{summary_suffix}{method_suffix}.tex"
    if rank.is_file():
        paths.append(rank)
    for metric in MATRIX_METRICS:
        path = Path(out_dir) / f"summary_matrix_{_metric_slug(metric)}_{model}{suffix}{method_suffix}.tex"
        if path.is_file():
            paths.append(path)
    return paths


def collect_diagnostic_snippet_paths(out_dir: Path, model: str) -> List[Path]:
    """Convergence/error material kept out of the result-table book."""
    paths = []
    for stem in ("summary_convergence_counts", "summary_convergence_problems"):
        path = Path(out_dir) / f"{stem}_{model}.tex"
        if path.is_file():
            paths.append(path)
    return paths


def write_tables_book(
    out_dir: Path,
    models: Sequence[str],
    *,
    filename: str = "summary_tables.tex",
    view: str = "combined",
    aggregation: str = "mean",
    figure_dir: Path = DEFAULT_HEADLINE_SCATTER_DIR,
    method_set: str = "selected",
    freshness_reference: Optional[Path] = None,
) -> Path:
    """Standalone document that includes only the paper summary snippets."""
    chunks = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[a4paper,landscape,margin=12mm]{geometry}",
        r"\usepackage{booktabs}",
        r"\usepackage{amsmath,amssymb}",
        r"\usepackage{graphicx}",
        r"\usepackage{longtable}",
        r"\usepackage{caption}",
        FA_ICON_PACKAGE,
        r"\captionsetup{font=small}",
        r"\pagestyle{empty}",
        r"\setlength{\tabcolsep}{3.5pt}",
        *TASK_COMMAND_DEFINITIONS,
        *SUMMARY_TABLE_COMMAND_DEFINITIONS,
        r"\begin{document}",
        "",
    ]
    # Snippets are independently regenerated: a valid book can legitimately
    # combine a freshly refreshed summary with an unchanged matrix snippet.
    # Timestamps therefore cannot distinguish a stale result from a valid
    # reusable artifact.  The existence check below is the reliable contract:
    # fail only when no required snippets are available to build the book.

    n_tables = 0
    for model in models:
        snippets = collect_snippet_paths(
            out_dir, model, view=view, aggregation=aggregation, method_set=method_set
        )
        if not snippets:
            continue
        if len(models) > 1:
            chunks.append(rf"\section*{{{model}}}")
        for path in snippets:
            chunks.append(table_snippet_for_pdf(path.read_text(encoding="utf-8")))
            n_tables += 1
    if n_tables == 0:
        raise FileNotFoundError(f"No summary_*.tex snippets in {out_dir} for models={list(models)}")

    if aggregation == "mean" and len(models) == 1:
        model = models[0]
        scopes = (
            ("non_one_side_tasks", "pairwise-capable tasks; pairwise and one-sided constructions"),
            ("one_side_tasks", "intrinsically one-sided tasks; pairwise methods omitted"),
        )
        if view == "pairwise":
            scopes = scopes[:1]
        elif view == "one_pole":
            scopes = scopes[1:]
        figure_stems = ("", "_light") if method_set == "selected" else ("_full", "_full_light")
        figure_names = tuple(
            (
                f"headline_scatter_{model}_{scope}{stem}.pdf",
                f"{MODEL_LATEX.get(model, model)} Detection and Intervention on {scope_caption}"
                f"{' (neutral-only Detection)' if stem.endswith('light') else ''}"
                f"{' (full ablation set)' if stem.startswith('_full') else ''}.",
            )
            for scope, scope_caption in scopes
            for stem in figure_stems
        )
    elif aggregation == "mean":
        figure_names = tuple(
            item
            for scope, scope_caption in (
                ("non_one_side_tasks", "pairwise-capable tasks; pairwise and one-sided constructions"),
                ("one_side_tasks", "intrinsically one-sided tasks; pairwise methods omitted"),
            )
            for stem in (("", "_light") if method_set == "selected" else ("_full", "_full_light"))
            for item in (
                (f"headline_scatter_{scope}_by_model{stem}.pdf", f"Detection and Intervention by method and base model on {scope_caption}."),
                (f"headline_scatter_{scope}_model_balanced{stem}.pdf", f"Detection and Intervention using model-balanced method means on {scope_caption}."),
            )
        )
    for figure_name, caption in figure_names if aggregation == "mean" else ():
        figure_path = Path(figure_dir) / figure_name
        if not figure_path.is_file():
            continue
        # Compilation runs from out_dir. Relative paths remain valid whether
        # Python/LaTeX run on Windows, Git Bash, WSL (/mnt/c), or Slurm.
        latex_path = Path(
            os.path.relpath(figure_path.resolve(), Path(out_dir).resolve())
        ).as_posix()
        chunks.extend(
            [
                r"\begin{figure}[p]",
                r"  \centering",
                rf"  \includegraphics[width=.86\textwidth,height=.88\textheight,keepaspectratio]{{\detokenize{{{latex_path}}}}}",
                rf"  \caption{{{caption}}}",
                r"\end{figure}",
                r"\clearpage",
                "",
            ]
        )
    chunks.append(r"\end{document}")
    dest = Path(out_dir) / filename
    dest.write_text("\n".join(chunks) + "\n", encoding="utf-8")
    return dest


def write_concatenated_tables_book(
    out_dir: Path,
    models: Sequence[str],
    *,
    category: str,
    filename: str,
    method_set: str = "selected",
) -> Path:
    """Write a standalone PDF source for one concatenated metric category."""
    method_suffix = "" if method_set == "selected" else "_full"
    chunks = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[a4paper,landscape,margin=12mm]{geometry}",
        r"\usepackage{booktabs}",
        r"\usepackage{amsmath,amssymb}",
        r"\usepackage{graphicx}",
        FA_ICON_PACKAGE,
        r"\pagestyle{empty}",
        *TASK_COMMAND_DEFINITIONS,
        *SUMMARY_TABLE_COMMAND_DEFINITIONS,
        r"\begin{document}",
        "",
    ]
    count = 0
    for model in models:
        path = Path(out_dir) / f"summary_concatenated_{category}_{model}{method_suffix}.tex"
        if not path.is_file():
            continue
        if len(models) > 1:
            chunks.append(rf"\section*{{{model}}}")
        chunks.append(table_snippet_for_pdf(path.read_text(encoding="utf-8")))
        count += 1
    if count == 0:
        raise FileNotFoundError(
            f"No concatenated {category} snippets in {out_dir} for models={list(models)}"
        )
    chunks.append(r"\end{document}")
    dest = Path(out_dir) / filename
    dest.write_text("\n".join(chunks) + "\n", encoding="utf-8")
    return dest


def write_headline_tables_book(
    out_dir: Path,
    *,
    filename: str = "summary_headline_tables.tex",
    figure_dir: Path = DEFAULT_HEADLINE_SCATTER_DIR,
    aggregation: str = "mean",
    method_set: str = "selected",
    min_model_completion: Optional[float] = None,
) -> Path:
    """Standalone across-model headline tables followed by both scatter views."""
    out_dir = Path(out_dir)
    suffix = _aggregation_suffix(aggregation)
    method_suffix = "" if method_set == "selected" else "_full"
    coverage_suffix = "_complete90" if min_model_completion == 0.9 else (
        f"_complete{int(round(min_model_completion * 100))}" if min_model_completion is not None else ""
    )
    table_paths = [
        out_dir / f"summary_across_models_submetrics{suffix}{coverage_suffix}{method_suffix}.tex",
        out_dir / f"summary_across_models_detection{suffix}{coverage_suffix}{method_suffix}.tex",
        out_dir / f"summary_across_models_intervention{suffix}{coverage_suffix}{method_suffix}.tex",
    ]
    missing = [path for path in table_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing headline snippet(s): " + ", ".join(map(str, missing)))
    chunks = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[a4paper,landscape,margin=12mm]{geometry}",
        r"\usepackage{booktabs}",
        r"\usepackage{amsmath,amssymb}",
        r"\usepackage{graphicx}",
        r"\usepackage{caption}",
        FA_ICON_PACKAGE,
        r"\captionsetup{font=small}",
        r"\pagestyle{empty}",
        *TASK_COMMAND_DEFINITIONS,
        *SUMMARY_TABLE_COMMAND_DEFINITIONS,
        r"\begin{document}",
        "",
    ]
    for path in table_paths:
        chunks.append(table_snippet_for_pdf(path.read_text(encoding="utf-8")))
    if aggregation == "mean":
        # The first and primary scatter is the combined Summary-table view:
        # all eligible task cells, both constructions.  Scope-restricted
        # figures remain diagnostics and follow it.
        figure_suffix = coverage_suffix
        main_figures = (
            (
                f"headline_scatter_signal_mean{figure_suffix}.pdf",
                "Detection and Intervention pooled across all eligible task--model cells, "
                "with arrows following the signal family within each estimator.",
            ),
            (f"headline_scatter{figure_suffix}.pdf", "Detection and Intervention pooled across all eligible task--model cells."),
            (f"headline_scatter_by_model{figure_suffix}.pdf", "Detection and Intervention by base model (each point averages that model's eligible task cells)."),
        ) if method_set == "selected" else ()
        stems = (
            (figure_suffix, f"{figure_suffix}_light")
            if method_set == "selected"
            else (f"{figure_suffix}_full", f"{figure_suffix}_full_light")
        )
        diagnostic_figures = tuple(
            (f"headline_scatter_{scope}_model_balanced{stem}.pdf", caption)
            for scope, caption in (
                ("non_one_side_tasks", "Detection and Intervention on pairwise-capable tasks, including pairwise and one-sided constructions."),
                ("one_side_tasks", "Detection and Intervention on intrinsically one-sided tasks; pairwise methods are omitted."),
            )
            for stem in stems
        )
        figure_names = main_figures + diagnostic_figures
    else:
        figure_names = ()
    for figure_name, caption in figure_names:
        figure_path = Path(figure_dir) / figure_name
        if not figure_path.is_file():
            continue
        latex_path = Path(
            os.path.relpath(figure_path.resolve(), out_dir.resolve())
        ).as_posix()
        chunks.extend(
            [
                r"\begin{figure}[p]",
                r"  \centering",
                rf"  \includegraphics[width=.86\textwidth,height=.88\textheight,keepaspectratio]{{\detokenize{{{latex_path}}}}}",
                rf"  \caption{{{caption}}}",
                r"\end{figure}",
                r"\clearpage",
                "",
            ]
        )
    chunks.append(r"\end{document}")
    dest = out_dir / filename
    dest.write_text("\n".join(chunks) + "\n", encoding="utf-8")
    return dest


def write_diagnostics_book(
    out_dir: Path,
    models: Sequence[str],
    *,
    filename: str = "summary_diagnostics.tex",
) -> Path:
    """Standalone convergence/error report, separate from result summaries."""
    chunks = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[a4paper,landscape,margin=12mm]{geometry}",
        r"\usepackage{booktabs}",
        r"\usepackage{longtable}",
        r"\usepackage{caption}",
        r"\captionsetup{font=small}",
        r"\pagestyle{empty}",
        r"\begin{document}",
        "",
    ]
    n_reports = 0
    for model in models:
        paths = collect_diagnostic_snippet_paths(out_dir, model)
        if not paths:
            continue
        if len(models) > 1:
            chunks.append(rf"\section*{{{MODEL_LATEX.get(model, model)}}}")
        for path in paths:
            chunks.append(table_snippet_for_pdf(path.read_text(encoding="utf-8")))
            n_reports += 1
    if not n_reports:
        raise FileNotFoundError(f"No diagnostic snippets in {out_dir} for {list(models)}")
    chunks.append(r"\end{document}")
    dest = Path(out_dir) / filename
    dest.write_text("\n".join(chunks) + "\n", encoding="utf-8")
    return dest


def fallback_footnote(provenance: Sequence[Mapping[str, Any]]) -> str:
    if not provenance:
        return ""
    causal_fb = [
        p
        for p in provenance
        if any(f in CAUSAL_FIELDS for f in (p.get("fields") or []))
    ]
    if not causal_fb:
        return ""
    n = len({(p["task"], p["method_group"]) for p in causal_fb})
    return (
        f"SAE/CAA causal values for {n} (task, method) cells use archived "
        f"``gpt2-small_old`` runs pending causal refresh."
    )


def print_task_report(
    df: pd.DataFrame,
    *,
    model: str,
    matrix_tasks: Sequence[str],
    summary_tasks: Sequence[str],
    methods: Sequence[str],
) -> None:
    print(f"=== Task report [{model}] ===")
    print(f"Matrix tasks ({len(matrix_tasks)}): {', '.join(matrix_tasks)}")
    print(f"Summary tasks ({len(summary_tasks)}): {', '.join(summary_tasks)}")
    print(f"Methods ({len(methods)}): {', '.join(_method_label(m) for m in methods)}")
    print()
    for metric in MATRIX_METRICS:
        if metric not in df.columns:
            continue
        active = methods_with_data(
            df, model=model, tasks=matrix_tasks, methods=methods, metric=metric
        )
        covered = 0
        total = len(matrix_tasks) * len(active)
        sub = df[
            (df["model"] == model)
            & (df["task"].isin(matrix_tasks))
            & (df["method_group"].isin(active))
        ]
        for rec in sub.to_dict(orient="records"):
            if _is_num(rec.get(metric)):
                covered += 1
        dropped = len(methods) - len(active)
        drop_note = f"  ({dropped} empty cols dropped)" if dropped else ""
        print(
            f"  {_metric_label(metric):12}  {covered}/{total} cells"
            f"  [{len(active)} cols]{drop_note}"
        )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--model", default="gpt2-small")
    parser.add_argument(
        "--causal-validation-fallback",
        action="store_true",
        default=os.environ.get("CAUSAL_VALIDATION_FALLBACK", "0") == "1",
        help=(
            "Prefer valid frozen-test causal scores, but use split-honest "
            "validation-selected scores when test is missing. Known-invalid "
            "artifact/grid combinations remain NaN."
        ),
    )
    parser.add_argument(
        "--subdir",
        default="",
        help=(
            "OUTPUT_SUBDIR value (see slurm/study_array.sh), e.g. 'suite_full'. "
            "Reads runs/{model}/{subdir}/{task}/results.json instead of "
            "runs/{model}/{task}/results.json — the default runs/{model}/ tree "
            "and an OUTPUT_SUBDIR tree are never merged together."
        ),
    )
    parser.add_argument(
        "--tasks",
        default="default",
        help="'default'/'all' (every task in data), or comma-separated task ids for metric matrices",
    )
    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="Print task coverage and exit",
    )
    parser.add_argument(
        "--write-book",
        metavar="MODELS",
        default=None,
        help="Write combined summary_tables.tex from existing snippets (comma-separated models)",
    )
    parser.add_argument(
        "--write-concatenated-book",
        metavar="CATEGORY:MODELS",
        help="Write a standalone concatenated raw-matrix book, e.g. detection:gpt2-small.",
    )
    parser.add_argument(
        "--freshness-reference",
        type=Path,
        default=None,
        help=(
            "Require snippets passed to --write-book to be newer than this file. "
            "Used by the PDF launcher to validate its current regeneration run."
        ),
    )
    parser.add_argument(
        "--book-view",
        choices=RESULT_TABLE_VIEWS,
        default="combined",
        help="Select combined, pairwise, or one-sided snippets for --write-book",
    )
    parser.add_argument(
        "--aggregation",
        choices=AGGREGATIONS,
        default="mean",
        help="Reducer used for task/model summary cells (default: mean)",
    )
    parser.add_argument(
        "--write-diagnostics-book",
        metavar="MODELS",
        default=None,
        help="Write summary_diagnostics.tex from convergence/error snippets",
    )
    parser.add_argument(
        "--write-headline-books",
        action="store_true",
        help=(
            "Rewrite only the across-model headline book .tex files from existing "
            "summary snippets and scatter figures (no results scan)."
        ),
    )
    parser.add_argument(
        "--report-mode",
        choices=("essential", "full"),
        default="full",
        help="Across-model artifact scope; full preserves the standalone historical matrix.",
    )
    parser.add_argument(
        "--across-models",
        action="store_true",
        help=(
            "Write the two methods-by-model headline tables. Uses the built-in "
            "model/source map: GPT-2 and Pythia suite_full2; Llama and Qwen "
            "default run trees."
        ),
    )
    parser.add_argument(
        "--method-set",
        choices=("selected", "full"),
        default="selected",
        help=(
            "Headline method set: selected excludes SAE_pre/CGA_tn by default; "
            "full includes every canonical group present in the frame. "
            "Override with HEADLINE_METHODS/HEADLINE_FULL_METHODS."
        ),
    )
    parser.add_argument(
        "--source",
        action="append",
        default=None,
        metavar="MODEL[=OUTPUT_SUBDIR]",
        help="Restrict --across-models to these repeatable model/source entries.",
    )
    parser.add_argument(
        "--summary-csv",
        action="append",
        default=None,
        metavar="MODEL=PATH",
        help=(
            "Use compact per-model summary_merged CSVs for --across-models. "
            "This efficient local path does not reopen results.json."
        ),
    )
    parser.add_argument(
        "--complete90-summary-csv",
        action="append",
        default=None,
        metavar="MODEL=PATH",
        help=(
            "Candidate compact CSVs used only for the >=90%%-complete companion. "
            "Ordinary headline tables retain --summary-csv's paper-model set."
        ),
    )
    args = parser.parse_args()

    if args.write_headline_books:
        args.out.mkdir(parents=True, exist_ok=True)
        written: List[Path] = []
        method_sets = ("selected",) if args.report_mode == "essential" else ("selected", "full")
        aggregations = ("mean",) if args.report_mode == "essential" else AGGREGATIONS
        completion_levels = (None,) if args.report_mode == "essential" else (None, 0.9)
        for method_set in method_sets:
            for aggregation in aggregations:
                for min_model_completion in completion_levels:
                    stem = "summary_headline_tables"
                    if method_set == "full":
                        stem += "_full"
                    if aggregation != "mean":
                        stem += f"_{aggregation}"
                    if min_model_completion is not None:
                        stem += f"_complete{int(min_model_completion * 100)}"
                    coverage_suffix = "_complete90" if min_model_completion == 0.9 else ""
                    if not (args.out / f"summary_across_models_detection{_aggregation_suffix(aggregation)}{coverage_suffix}.tex").is_file():
                        print(f"Skipping {stem}: its headline snippets do not exist.")
                        continue
                    written.append(
                        write_headline_tables_book(
                            args.out,
                            filename=f"{stem}.tex",
                            aggregation=aggregation,
                            method_set=method_set,
                            min_model_completion=min_model_completion,
                        )
                    )
        for path in written:
            print(f"Wrote {path}")
        return

    if args.across_models:
        if args.source and args.summary_csv:
            parser.error("use either --source or --summary-csv with --across-models, not both")
        if args.source and args.complete90_summary_csv:
            parser.error("--complete90-summary-csv requires --summary-csv")
        csv_sources = parse_summary_csv_sources(args.summary_csv) if args.summary_csv else ()
        complete90_csv_sources = (
            parse_summary_csv_sources(args.complete90_summary_csv)
            if args.complete90_summary_csv
            else csv_sources
        )
        sources = (
            tuple((model, "") for model, _path in csv_sources)
            if csv_sources
            else parse_cross_model_sources(args.source)
        )
        loaded_sources = load_cross_model_summary_csvs(csv_sources) if csv_sources else None
        complete90_sources = tuple((model, "") for model, _path in complete90_csv_sources) if complete90_csv_sources else sources
        complete90_loaded_sources = (
            load_cross_model_summary_csvs(complete90_csv_sources)
            if complete90_csv_sources
            else loaded_sources
        )
        written: List[Path] = []
        method_sets = ("selected",) if args.report_mode == "essential" else ("selected", "full")
        aggregations = ("mean",) if args.report_mode == "essential" else AGGREGATIONS
        completion_levels = (None,) if args.report_mode == "essential" else (None, 0.9)
        for method_set in method_sets:
            for aggregation in aggregations:
                for min_model_completion in completion_levels:
                    pass_sources = complete90_sources if min_model_completion == 0.9 else sources
                    pass_loaded_sources = complete90_loaded_sources if min_model_completion == 0.9 else loaded_sources
                    artifacts = write_cross_model_headline_summaries(
                            args.runs,
                            args.out,
                            sources=pass_sources,
                            aggregation=aggregation,
                            method_set=method_set,
                            min_model_completion=min_model_completion,
                            loaded_sources=pass_loaded_sources,
                        )
                    written.extend(artifacts)
                    if not artifacts:
                        continue
                    stem = "summary_headline_tables"
                    if method_set == "full":
                        stem += "_full"
                    if aggregation != "mean":
                        stem += f"_{aggregation}"
                    if min_model_completion is not None:
                        stem += f"_complete{int(min_model_completion * 100)}"
                    written.append(
                        write_headline_tables_book(
                            args.out,
                            filename=f"{stem}.tex",
                            aggregation=aggregation,
                            method_set=method_set,
                            min_model_completion=min_model_completion,
                        )
                    )
        for path in written:
            print(f"Wrote {path}")
        return

    if args.write_book:
        models = [m.strip() for m in args.write_book.split(",") if m.strip()]
        args.out.mkdir(parents=True, exist_ok=True)
        book_stem = (
            "summary_tables"
            if args.book_view == "combined"
            else f"summary_tables_{args.book_view}"
        )
        if args.method_set == "full":
            book_stem += "_full"
        filename = f"{book_stem}{_aggregation_suffix(args.aggregation)}.tex"
        book = write_tables_book(
            args.out,
            models,
            filename=filename,
            view=args.book_view,
            aggregation=args.aggregation,
            method_set=args.method_set,
            freshness_reference=args.freshness_reference,
        )
        print(f"Wrote {book}")
        return

    if args.write_concatenated_book:
        try:
            category, models_text = args.write_concatenated_book.split(":", 1)
        except ValueError:
            parser.error("--write-concatenated-book must be CATEGORY:MODELS")
        if category not in {"detection", "intervention"}:
            parser.error("concatenated category must be detection or intervention")
        models = [m.strip() for m in models_text.split(",") if m.strip()]
        if not models:
            parser.error("--write-concatenated-book requires at least one model")
        args.out.mkdir(parents=True, exist_ok=True)
        method_suffix = "" if args.method_set == "selected" else "_full"
        book = write_concatenated_tables_book(
            args.out,
            models,
            category=category,
            filename=f"summary_tables_{category}{method_suffix}.tex",
            method_set=args.method_set,
        )
        print(f"Wrote {book}")
        return

    if args.write_diagnostics_book:
        models = [m.strip() for m in args.write_diagnostics_book.split(",") if m.strip()]
        args.out.mkdir(parents=True, exist_ok=True)
        book = write_diagnostics_book(args.out, models)
        print(f"Wrote {book}")
        return

    df = collect_canonical_group_rows(
        args.runs,
        args.model,
        subdir=args.subdir,
        causal_validation_fallback=args.causal_validation_fallback,
    )
    specs = collect_task_specs(_model_dir(args.runs, args.model, args.subdir))
    provenance: List[Dict[str, Any]] = []
    disk_tasks = list_run_tasks(args.runs, args.model, subdir=args.subdir)
    matrix_tasks = resolve_matrix_tasks(
        df, model=args.model, tasks_arg=args.tasks, disk_tasks=disk_tasks
    )
    summary_tasks = matrix_tasks
    methods = _order_methods(
        list(headline_method_groups(args.method_set, frame=df))
    )

    if args.list_tasks:
        print_task_report(
            df,
            model=args.model,
            matrix_tasks=matrix_tasks,
            summary_tasks=summary_tasks,
            methods=methods,
        )
        if provenance:
            print("Fallback provenance:")
            for p in provenance:
                print(f"  {p['task']}/{p['method_group']}: {p['kind']} {p.get('fields')}")
        return

    args.out.mkdir(parents=True, exist_ok=True)
    commands_path = write_summary_table_commands(args.out)
    layer_agg = canonical_layer_aggregations(args.runs, args.model, args.subdir)
    layer_csv_path = args.out / f"layer_aggregations_{args.model}.csv"
    layer_tex_path = args.out / f"summary_layer_aggregations_{args.model}.tex"
    layer_agg.to_csv(layer_csv_path, index=False)
    layer_tex = format_layer_aggregation_latex(layer_agg, model=args.model)
    if layer_tex:
        layer_tex_path.write_text(layer_tex, encoding="utf-8")
    elif layer_tex_path.is_file():
        layer_tex_path.unlink()
    for pattern in DEPRECATED_OUTPUTS:
        stale = args.out / pattern.format(model=args.model)
        if stale.is_file():
            stale.unlink()
    written: List[str] = [str(commands_path), str(layer_csv_path)]
    if layer_tex:
        written.append(str(layer_tex_path))
    for aggregation in AGGREGATIONS:
        for view in RESULT_TABLE_VIEWS:
            view_written = write_result_table_view(
                df,
                args.out,
                model=args.model,
                tasks=summary_tasks,
                methods=methods,
                specs=specs,
                view=view,
                aggregation=aggregation,
                method_set=args.method_set,
            )
            written.extend(str(path) for path in view_written)

    # GRADIEND/ACTIEND training-convergence status -- a debugging companion to
    # the headline method tables, generated here (not a separate command) so
    # `--write-book` / scripts/summary_tables_pdf.sh always fold it into
    # summary_tables.pdf. See analysis/convergence_overview.py.
    conv_rows = collect_convergence_rows(args.runs, args.model, subdir=args.subdir)
    conv_counts_path = args.out / f"summary_convergence_counts_{args.model}.tex"
    conv_problems_path = args.out / f"summary_convergence_problems_{args.model}.tex"
    conv_csv_path = args.out / f"summary_convergence_{args.model}.csv"
    if conv_rows:
        conv_counts_path.write_text(
            format_convergence_counts_latex(conv_rows, model=args.model), encoding="utf-8"
        )
        written.append(str(conv_counts_path))
        conv_problems_tex = format_convergence_problems_latex(conv_rows, model=args.model)
        if conv_problems_tex:
            conv_problems_path.write_text(conv_problems_tex, encoding="utf-8")
            written.append(str(conv_problems_path))
        elif conv_problems_path.is_file():
            conv_problems_path.unlink()
        pd.DataFrame(conv_rows).to_csv(conv_csv_path, index=False)
    else:
        for stale in (conv_counts_path, conv_problems_path, conv_csv_path):
            if stale.is_file():
                stale.unlink()

    long_path = args.out / f"summary_merged_{args.model}.csv"
    df[df["model"] == args.model].to_csv(long_path, index=False)

    manifest = {
        "model": args.model,
        "runs": str(args.runs),
        "subdir": args.subdir or None,
        "fallback": None,
        "causal_validation_fallback": args.causal_validation_fallback,
        "matrix_tasks": list(matrix_tasks),
        "summary_tasks": list(summary_tasks),
        "methods": methods,
        "metrics": list(MATRIX_METRICS),
        "provenance": provenance,
        "convergence_runs": len(conv_rows),
        "convergence_problems": sum(1 for r in conv_rows if r["status"] in CONVERGENCE_PROBLEM_STATUSES),
    }
    manifest_path = args.out / f"summary_manifest_{args.model}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print_task_report(
        df,
        model=args.model,
        matrix_tasks=matrix_tasks,
        summary_tasks=summary_tasks,
        methods=methods,
    )
    for path in written:
        print(f"Wrote {path}")
    print(f"Wrote {manifest_path}")
    if conv_rows:
        print(
            f"Convergence: {manifest['convergence_problems']}/{len(conv_rows)} "
            "gradiend/actiend runs flagged (collapsed / not_converged / unknown_low_quality)"
        )
    note = fallback_footnote(provenance)
    if note:
        print(f"Note: {note}")


if __name__ == "__main__":
    main()
