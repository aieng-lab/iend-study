"""Global family × task visualizations and aggregation tables.

Built on the same best-per-class cells as ``summarize_family_overview``.
Figures emphasize *who wins where* and coverage (SAE/CAA means can look
strong while missing many tasks). Higher is better for every overview metric.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from analysis.family_overview import FAMILIES, OVERVIEW_METRICS, pivot_family_task
from analysis.plot_style import method_color, strip_figure_titles
from analysis.task_specs import family_is_applicable, resolve_spec
from analysis.task_order import order_tasks

ApplicableFn = Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIG = ROOT / "analysis" / "figures" / "family"
DEFAULT_TABLE = ROOT / "analysis" / "tables" / "family"

WIN_EPS = 1e-6

_GLOBAL_STYLE_FAMILIES = (
    "gradiend", "actiend", "actiend_pre", "actiend_ridge", "sae", "sae_pre",
    "caa", "cga", "cga_tensor_norm", "caga", "agiend",
    "gradiend:two_pole", "gradiend:one_pole", "actiend:two_pole",
    "actiend:one_pole", "actiend_pre:two_pole", "actiend_pre:one_pole",
    "sae:kstar", "sae:k1", "sae_pre:kstar", "sae_pre:k1",
    "caa:act_prediction", "caa:all_act_prediction",
)
FAMILY_COLORS: Dict[str, str] = {
    family: method_color(family) for family in _GLOBAL_STYLE_FAMILIES
}
FAMILY_LABELS: Dict[str, str] = {
    "gradiend": "GRADIEND",
    "actiend": "ACTIEND",
    "actiend_pre": "ACTIEND-PRE",
    "sae": "SAE",
    "sae_pre": "SAE-PRE",
    "caa": "CAA",
    "gradiend:two_pole": "GRADIEND two-pole",
    "gradiend:one_pole": "GRADIEND one-pole",
    "actiend:two_pole": "ACTIEND two-pole",
    "actiend:one_pole": "ACTIEND one-pole",
    "actiend_pre:two_pole": "ACTIEND-PRE two-pole",
    "actiend_pre:one_pole": "ACTIEND-PRE one-pole",
    "sae:kstar": r"SAE $k^*$",
    "sae:k1": "SAE k=1",
    "sae_pre:kstar": r"SAE-PRE $k^*$",
    "sae_pre:k1": "SAE-PRE k=1",
    "caa:act_prediction": "CAA act",
    "caa:all_act_prediction": "CAA all",
}
FAMILY_SHORT: Dict[str, str] = {
    "gradiend": "GRAD",
    "actiend": "ACT",
    "actiend_pre": "A-PRE",
    "sae": "SAE",
    "sae_pre": "PRE",
    "caa": "CAA",
    "gradiend:two_pole": "G-2p",
    "gradiend:one_pole": "G-1p",
    "actiend:two_pole": "A-2p",
    "actiend:one_pole": "A-1p",
    "actiend_pre:two_pole": "AP-2p",
    "actiend_pre:one_pole": "AP-1p",
    "sae:kstar": r"$k^*$",
    "sae:k1": "k1",
    "sae_pre:kstar": r"$k^*$",
    "sae_pre:k1": "k1",
    "caa:act_prediction": "CAA",
    "caa:all_act_prediction": "CAA-all",
}

METRIC_LABELS: Dict[str, str] = {
    "encoding_E": "Encoding E",
    "roc_auc_neutral": "AUC (neutral)",
    "roc_auc_other": "AUC (other)",
    "neutral_specificity": "Neutral spec.",
    "class_exclusivity": "Exclusivity",
    "suitability": "Suitability S",
    "suitability_E": "Suitability E",
    "suitability_G": "Suitability G",
    "causal_signed_effect": "Causal ΔP",
    "causal_lms": "Causal LMS",
}

TASK_LABELS: Dict[str, str] = {
    "emotion": "Emotion",
    "function_composition": "Fn composition",
    "gender_en": "Gender",
    "induction": "Induction",
    "ioi": "IOI",
    "ioi_mib": "IOI (MIB)",
    "key_value": "Key–value",
    "language": "Language",
    "pronoun_number": "Pron. number",
    "pronoun_person": "Pron. person",
    "race": "Race",
    "ravel_continent": "RAVEL continent",
    "ravel_country": "RAVEL country",
    "ravel_language": "RAVEL language",
    "religion": "Religion",
    "repetition": "Repetition",
}

TASK_GROUPS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("Social", ("race", "religion", "gender_en", "emotion")),
    ("Factual", ("ravel_continent", "ravel_country", "ravel_language")),
    ("Language", ("language", "pronoun_person", "pronoun_number")),
    ("Circuits", ("induction", "ioi", "ioi_mib", "repetition")),
    ("Compositional", ("function_composition", "key_value")),
)

UNIT_METRICS = frozenset(
    {
        "encoding_E",
        "roc_auc_neutral",
        "roc_auc_other",
        "neutral_specificity",
        "class_exclusivity",
        "suitability",
        "suitability_E",
        "suitability_G",
    }
)
HEADLINE_METRICS: Tuple[str, ...] = (
    "encoding_E",
    "roc_auc_other",
    "class_exclusivity",
    "suitability",
    "causal_signed_effect",
)

_TIE_FACE = "#F3E3A3"
_GAP_FACE = "#D0D0D0"
_MISS_FACE = "#F4D6D6"
_OK_CMAP = "cividis"


def _metric_label(metric: str) -> str:
    return METRIC_LABELS.get(metric, metric.replace("_", " "))


def _task_label(task: str) -> str:
    return TASK_LABELS.get(task, task.replace("_", " "))


def _family_label(family: str) -> str:
    return FAMILY_LABELS.get(family, family)


def _family_color(family: str) -> str:
    return FAMILY_COLORS.get(family, "#555555")


def _family_short(family: str) -> str:
    return FAMILY_SHORT.get(family, family.split(":")[-1][:6])


def _order_tasks(tasks: Sequence[str]) -> List[str]:
    return order_tasks(tasks)


def _order_metrics(metrics: Sequence[str]) -> List[str]:
    rank = {m: i for i, m in enumerate(OVERVIEW_METRICS)}
    return sorted(metrics, key=lambda m: (rank.get(m, 1_000), m))


@dataclass
class ScoreGrid:
    """Family × task numeric grid for one (model, metric)."""

    model: str
    metric: str
    families: List[str]
    tasks: List[str]
    mean: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    status: np.ndarray  # "ok" | "gap" | "missing"

    @property
    def n_fam(self) -> int:
        return len(self.families)

    @property
    def n_task(self) -> int:
        return len(self.tasks)

    def fam_index(self, family: str) -> int:
        return self.families.index(family)


def build_score_grid(
    cells: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    model: str,
    specs: Optional[Mapping[Tuple[str, str], Mapping[str, Any]]] = None,
    families: Optional[Sequence[str]] = None,
    applicable: Optional[ApplicableFn] = None,
) -> ScoreGrid:
    applicable_fn = applicable or family_is_applicable
    families, tasks, lookup = pivot_family_task(
        cells, metric=metric, model=model, specs=specs, families=families
    )
    tasks = _order_tasks(tasks)
    mean = np.full((len(families), len(tasks)), np.nan, dtype=float)
    lo = np.full_like(mean, np.nan)
    hi = np.full_like(mean, np.nan)
    status = np.empty(mean.shape, dtype=object)
    for i, fam in enumerate(families):
        for j, task in enumerate(tasks):
            cell = lookup.get((fam, task))
            spec = resolve_spec(specs, model=model, task=task)
            expected = applicable_fn(fam, spec)
            if not cell:
                status[i, j] = "gap" if not expected else "missing"
                continue
            mean[i, j] = float(cell["mean"])
            lo[i, j] = float(cell["min"])
            hi[i, j] = float(cell["max"])
            status[i, j] = "ok"
    return ScoreGrid(
        model=model,
        metric=metric,
        families=list(families),
        tasks=tasks,
        mean=mean,
        lo=lo,
        hi=hi,
        status=status,
    )


def task_winners(grid: ScoreGrid, *, eps: float = WIN_EPS) -> List[List[str]]:
    """Per-task families within ``eps`` of the finite maximum."""
    out: List[List[str]] = []
    for j in range(grid.n_task):
        col = grid.mean[:, j]
        if not np.any(np.isfinite(col)):
            out.append([])
            continue
        top = float(np.nanmax(col))
        hits = [
            grid.families[i]
            for i in range(grid.n_fam)
            if np.isfinite(col[i]) and abs(float(col[i]) - top) <= eps
        ]
        out.append(hits)
    return out


def pairwise_wins(grid: ScoreGrid, *, eps: float = WIN_EPS) -> np.ndarray:
    """``wins[a, b]`` = tasks where family a strictly beats b (both scored)."""
    n = grid.n_fam
    wins = np.zeros((n, n), dtype=int)
    for a in range(n):
        for b in range(n):
            if a == b:
                continue
            for j in range(grid.n_task):
                va, vb = grid.mean[a, j], grid.mean[b, j]
                if np.isfinite(va) and np.isfinite(vb) and float(va) > float(vb) + eps:
                    wins[a, b] += 1
    return wins


def common_support_mask(grid: ScoreGrid) -> np.ndarray:
    """Tasks scored by every family (4-way intersection)."""
    return np.all(np.isfinite(grid.mean), axis=0)


def aggregate_family_stats(grid: ScoreGrid, *, eps: float = WIN_EPS) -> List[Dict[str, Any]]:
    """One rollup row per family for ``grid.metric``."""
    winners = task_winners(grid, eps=eps)
    common = common_support_mask(grid)
    n_applicable = []
    for i in range(grid.n_fam):
        n_applicable.append(int(np.sum(grid.status[i] != "gap")))

    rows: List[Dict[str, Any]] = []
    for i, fam in enumerate(grid.families):
        vals = grid.mean[i]
        ok = np.isfinite(vals)
        scored = [float(v) for v in vals[ok]]
        n = len(scored)
        n_app = n_applicable[i] if n_applicable[i] else n
        ranks: List[float] = []
        gaps: List[float] = []
        unique_wins = 0
        tied_first = 0
        for j in range(grid.n_task):
            if not ok[j]:
                continue
            col = grid.mean[:, j]
            finite = [(grid.families[k], float(col[k])) for k in range(grid.n_fam) if np.isfinite(col[k])]
            finite.sort(key=lambda kv: -kv[1])
            rank = 1 + sum(1 for _, v in finite if v > float(vals[j]) + eps)
            ranks.append(float(rank))
            gaps.append(float(np.nanmax(col) - vals[j]))
            hits = winners[j]
            if fam in hits:
                tied_first += 1
                if len(hits) == 1:
                    unique_wins += 1
        best_task = worst_task = ""
        if scored:
            order = [j for j in range(grid.n_task) if ok[j]]
            best_j = max(order, key=lambda j: float(vals[j]))
            worst_j = min(order, key=lambda j: float(vals[j]))
            best_task = grid.tasks[best_j]
            worst_task = grid.tasks[worst_j]
        common_vals = [float(vals[j]) for j in range(grid.n_task) if common[j]]
        rows.append(
            {
                "model": grid.model,
                "metric": grid.metric,
                "family": fam,
                "n_scored": n,
                "n_applicable": n_app,
                "coverage": (n / n_app) if n_app else float("nan"),
                "mean": (sum(scored) / n) if n else float("nan"),
                "mean_common": (sum(common_vals) / len(common_vals)) if common_vals else float("nan"),
                "n_common": int(np.sum(common)),
                "median": float(np.median(scored)) if scored else float("nan"),
                "wins": unique_wins,
                "ties_first": tied_first,
                "mean_rank": (sum(ranks) / len(ranks)) if ranks else float("nan"),
                "mean_gap": (sum(gaps) / len(gaps)) if gaps else float("nan"),
                "best_task": best_task,
                "worst_task": worst_task,
            }
        )
    return rows


def winner_rows(grid: ScoreGrid, *, eps: float = WIN_EPS) -> List[Dict[str, Any]]:
    """One row per task: winner(s), score, margin, runner-up."""
    winners = task_winners(grid, eps=eps)
    out: List[Dict[str, Any]] = []
    for j, task in enumerate(grid.tasks):
        col = grid.mean[:, j]
        hits = winners[j]
        if not hits:
            out.append(
                {
                    "model": grid.model,
                    "metric": grid.metric,
                    "task": task,
                    "winners": "",
                    "score": float("nan"),
                    "margin": float("nan"),
                    "runner_up": "",
                    "runner_score": float("nan"),
                    "n_scored": 0,
                }
            )
            continue
        top = float(np.nanmax(col))
        others = sorted(
            (
                (grid.families[i], float(col[i]))
                for i in range(grid.n_fam)
                if np.isfinite(col[i]) and grid.families[i] not in hits
            ),
            key=lambda kv: -kv[1],
        )
        runner, runner_s = others[0] if others else ("", float("nan"))
        margin = (top - runner_s) if others else float("nan")
        out.append(
            {
                "model": grid.model,
                "metric": grid.metric,
                "task": task,
                "winners": ",".join(hits),
                "score": top,
                "margin": margin,
                "runner_up": runner,
                "runner_score": runner_s,
                "n_scored": int(np.sum(np.isfinite(col))),
            }
        )
    return out


def _fmt(v: Optional[float], digits: int = 3) -> str:
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "NaN"
    return f"{v:.{digits}f}"


def format_global_stats_text(rows: Sequence[Mapping[str, Any]], *, metric: str, model: str) -> str:
    n_common = next((int(r["n_common"]) for r in rows if r.get("n_common") is not None), 0)
    common_note = (
        f"common = mean on the {n_common}-task intersection scored by all families"
        if n_common
        else "common = n/a (no task scored by every family)"
    )
    header = (
        f"Global aggregation -- family vs all tasks ({metric}) [{model}]\n"
        f"mean = available-task mean (same as table MEAN); {common_note}\n"
        f"wins = unique first place; ties_first includes ties; rank/gap ignore gaps & missing"
    )
    name_w = max(10, max((len(str(r.get("family") or "")) for r in rows), default=6))
    titles = (
        f"{'family':<{name_w}} {'n/N':>6} {'cov':>6} {'mean':>7} {'common':>7} {'median':>7} "
        f"{'wins':>5} {'ties':>5} {'rank':>6} {'gap':>6}  {'best':<20} {'worst':<20}"
    )
    lines = [header, titles, "-" * len(titles)]
    for r in rows:
        n = f"{int(r['n_scored'])}/{int(r['n_applicable'])}"
        lines.append(
            f"{str(r['family']):<{name_w}} {n:>6} {_fmt(r.get('coverage')):>6} {_fmt(r.get('mean')):>7} "
            f"{_fmt(r.get('mean_common')):>7} {_fmt(r.get('median')):>7} {int(r['wins']):>5} "
            f"{int(r['ties_first']):>5} {_fmt(r.get('mean_rank')):>6} {_fmt(r.get('mean_gap')):>6}  "
            f"{str(r.get('best_task') or '-'):<20} {str(r.get('worst_task') or '-'):<20}"
        )
    return "\n".join(lines)


def format_winners_text(rows: Sequence[Mapping[str, Any]], *, metric: str, model: str) -> str:
    lines = [
        f"Who wins -- {metric} [{model}]",
        f"{'task':<22} {'winner(s)':<28} {'score':>7} {'margin':>7}  runner-up",
        "-" * 88,
    ]
    for r in rows:
        winners = r.get("winners") or "-"
        runner = r.get("runner_up") or "-"
        rs = r.get("runner_score")
        runner_s = f"{runner} ({_fmt(rs)})" if runner != "-" else "-"
        lines.append(
            f"{str(r['task']):<22} {winners:<28} {_fmt(r.get('score')):>7} {_fmt(r.get('margin')):>7}  {runner_s}"
        )
    return "\n".join(lines)


def format_cross_metric_text(
    stats_by_metric: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    model: str,
    metrics: Sequence[str],
    families: Optional[Sequence[str]] = None,
) -> str:
    families = list(families) if families is not None else list(FAMILIES)
    short = {
        "encoding_E": "E",
        "roc_auc_neutral": "AUC_n",
        "roc_auc_other": "AUC_o",
        "neutral_specificity": "spec",
        "class_exclusivity": "excl",
        "suitability": "S",
        "suitability_E": "S_E",
        "suitability_G": "S_G",
        "causal_signed_effect": "dP",
        "causal_lms": "LMS",
    }
    col_w = 14
    name_w = max(10, max((len(f) for f in families), default=6))
    labels = [short.get(m, m[:8]) for m in metrics]
    head = f"{'family':<{name_w}}" + "".join(f"{lab:>{col_w}}" for lab in labels) + f"{'SUM_WINS':>{col_w}}{'MEAN_RANK':>{col_w}}"
    lines = [
        f"Cross-metric rollup [{model}]  (available mean / unique wins)",
        head,
        "-" * len(head),
    ]
    for fam in families:
        row = f"{fam:<{name_w}}"
        wins = 0
        ranks: List[float] = []
        for metric in metrics:
            hit = next((r for r in stats_by_metric.get(metric, []) if r["family"] == fam), None)
            if not hit:
                row += f"{'—':>{col_w}}"
                continue
            wins += int(hit["wins"])
            if hit.get("mean_rank") is not None and not math.isnan(float(hit["mean_rank"])):
                ranks.append(float(hit["mean_rank"]))
            cell = f"{_fmt(hit.get('mean'))} ({int(hit['wins'])}w)"
            row += f"{cell:>{col_w}}"
        mean_rank = sum(ranks) / len(ranks) if ranks else float("nan")
        row += f"{wins:>{col_w}}{_fmt(mean_rank):>{col_w}}"
        lines.append(row)
    return "\n".join(lines)


def format_global_latex(
    stats_by_metric: Mapping[str, Sequence[Mapping[str, Any]]],
    winners_by_metric: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    model: str,
    metrics: Sequence[str],
    families: Optional[Sequence[str]] = None,
) -> str:
    chunks = [
        "% Auto-generated by analysis/plot_family_overview.py",
        "% Global family aggregation (available mean, common-support mean, wins).",
        "",
    ]
    for metric in metrics:
        rows = list(stats_by_metric.get(metric, []))
        if not rows:
            continue
        n_common = int(rows[0]["n_common"]) if rows else 0
        metric_tex = metric.replace("_", r"\_")
        lines = [
            r"\begin{table}[t]",
            r"\centering",
            (
                f"\\caption{{Global family aggregation ({metric_tex}) on {model}: "
                f"available-task mean, {n_common}-task common-support mean, "
                f"unique wins, mean rank, mean gap to best.}}"
            ),
            f"\\label{{tab:family-global-{metric}-{model}}}",
            r"\begin{tabular}{lrrrrrrrr}",
            r"\toprule",
            r"family & n & cov & mean & common & wins & rank & gap & best \\",
            r"\midrule",
        ]
        for r in rows:
            n = f"{int(r['n_scored'])}/{int(r['n_applicable'])}"
            best = str(r.get("best_task") or "").replace("_", r"\_")
            lines.append(
                " & ".join(
                    [
                        str(r["family"]),
                        n,
                        _fmt(r.get("coverage")),
                        _fmt(r.get("mean")),
                        _fmt(r.get("mean_common")),
                        str(int(r["wins"])),
                        _fmt(r.get("mean_rank")),
                        _fmt(r.get("mean_gap")),
                        best,
                    ]
                )
                + r" \\"
            )
        lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
        chunks.append("\n".join(lines))

        wrows = list(winners_by_metric.get(metric, []))
        if wrows:
            wlines = [
                r"\begin{table}[t]",
                r"\centering",
                f"\\caption{{Per-task winner ({metric_tex}) on {model}.}}",
                f"\\label{{tab:family-winners-{metric}-{model}}}",
                r"\begin{tabular}{llrrl}",
                r"\toprule",
                r"task & winner(s) & score & margin & runner-up \\",
                r"\midrule",
            ]
            for r in wrows:
                wlines.append(
                    " & ".join(
                        [
                            str(r["task"]).replace("_", r"\_"),
                            (r.get("winners") or "—").replace("_", r"\_"),
                            _fmt(r.get("score")),
                            _fmt(r.get("margin")),
                            (r.get("runner_up") or "—").replace("_", r"\_"),
                        ]
                    )
                    + r" \\"
                )
            wlines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
            chunks.append("\n".join(wlines))

    # Cross-metric rollup
    families = list(families) if families is not None else list(FAMILIES)
    cols = "l" + "c" * len(metrics) + "cc"
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        f"\\caption{{Cross-metric rollup on {model}: available mean (unique wins).}}",
        f"\\label{{tab:family-global-rollup-{model}}}",
        f"\\begin{{tabular}}{{{cols}}}",
        r"\toprule",
        "family & "
        + " & ".join(m.replace("_", r"\_") for m in metrics)
        + r" & $\Sigma$ wins & mean rank \\",
        r"\midrule",
    ]
    for fam in families:
        cells_tex = [fam]
        wins = 0
        ranks: List[float] = []
        for metric in metrics:
            hit = next((r for r in stats_by_metric.get(metric, []) if r["family"] == fam), None)
            if not hit:
                cells_tex.append("—")
                continue
            wins += int(hit["wins"])
            if hit.get("mean_rank") is not None and not math.isnan(float(hit["mean_rank"])):
                ranks.append(float(hit["mean_rank"]))
            cells_tex.append(f"{_fmt(hit.get('mean'))} ({int(hit['wins'])})")
        mean_rank = sum(ranks) / len(ranks) if ranks else float("nan")
        cells_tex.extend([str(wins), _fmt(mean_rank)])
        lines.append(" & ".join(cells_tex) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    chunks.append("\n".join(lines))
    return "\n".join(chunks)


def write_global_tables(
    cells: Sequence[Mapping[str, Any]],
    *,
    out: Path = DEFAULT_TABLE,
    model: Optional[str] = None,
    metrics: Optional[Sequence[str]] = None,
    specs: Optional[Mapping[Tuple[str, str], Mapping[str, Any]]] = None,
    families: Optional[Sequence[str]] = None,
    applicable: Optional[ApplicableFn] = None,
    prefix: str = "family",
) -> List[Path]:
    """Write global aggregation / winner tables (txt, tex, csv)."""
    metric_list = _order_metrics(metrics or OVERVIEW_METRICS)
    models = sorted({str(c["model"]) for c in cells})
    if model:
        models = [m for m in models if m == model]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    stats_fields = [
        "model",
        "metric",
        "family",
        "n_scored",
        "n_applicable",
        "coverage",
        "mean",
        "mean_common",
        "n_common",
        "median",
        "wins",
        "ties_first",
        "mean_rank",
        "mean_gap",
        "best_task",
        "worst_task",
    ]
    win_fields = [
        "model",
        "metric",
        "task",
        "winners",
        "score",
        "margin",
        "runner_up",
        "runner_score",
        "n_scored",
    ]
    all_stats: List[Dict[str, Any]] = []
    all_wins: List[Dict[str, Any]] = []
    all_txt: List[str] = []
    all_tex: List[str] = []

    for mod in models:
        stats_by: Dict[str, List[Dict[str, Any]]] = {}
        wins_by: Dict[str, List[Dict[str, Any]]] = {}
        for metric in metric_list:
            grid = build_score_grid(
                cells,
                metric=metric,
                model=mod,
                specs=specs,
                families=families,
                applicable=applicable,
            )
            if grid.n_task == 0:
                continue
            stats = aggregate_family_stats(grid)
            wins = winner_rows(grid)
            stats_by[metric] = stats
            wins_by[metric] = wins
            all_stats.extend(stats)
            all_wins.extend(wins)
            all_txt.append(format_global_stats_text(stats, metric=metric, model=mod))
            all_txt.append("")
            all_txt.append(format_winners_text(wins, metric=metric, model=mod))
        if stats_by:
            all_txt.append(
                format_cross_metric_text(
                    stats_by, model=mod, metrics=list(stats_by), families=families
                )
            )
            all_tex.append(
                format_global_latex(
                    stats_by, wins_by, model=mod, metrics=list(stats_by), families=families
                )
            )

    written: List[Path] = []

    def _csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
        with path.open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(fields), extrasaction="ignore")
            w.writeheader()
            for row in rows:
                w.writerow({k: row.get(k) for k in fields})

    stats_csv = out / f"{prefix}_global_stats.csv"
    wins_csv = out / f"{prefix}_global_winners.csv"
    txt_path = out / f"{prefix}_global_all.txt"
    tex_path = out / f"{prefix}_global_all.tex"
    _csv(stats_csv, all_stats, stats_fields)
    _csv(wins_csv, all_wins, win_fields)
    txt_path.write_text("\n\n".join(all_txt).rstrip() + "\n", encoding="utf-8")
    tex_path.write_text("\n".join(all_tex), encoding="utf-8")
    written.extend([stats_csv, wins_csv, txt_path, tex_path])
    return written


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _pyplot():
    import matplotlib

    matplotlib.rcParams["text.usetex"] = False
    from matplotlib import pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.titleweight": "semibold",
            "axes.labelsize": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.dpi": 180,
            "pdf.fonttype": 42,
            "text.usetex": False,
        }
    )
    return plt


def _save(fig, path: Path, *, plt) -> List[Path]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    out = path.with_suffix(".pdf")
    strip_figure_titles(fig)
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    written.append(out)
    plt.close(fig)
    return written


def _text_color_for(hex_color: str) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4))
    # Rec. 709 luminance
    return "white" if (0.2126 * r + 0.7152 * g + 0.0722 * b) < 0.55 else "#1a1a1a"


def _draw_group_brackets(ax, tasks: Sequence[str], *, y: float = -0.18) -> None:
    """Label conceptual task groups under a heatmap x-axis."""
    groups: List[Tuple[str, int, int]] = []
    for name, members in TASK_GROUPS:
        idx = [i for i, t in enumerate(tasks) if t in members]
        if idx:
            groups.append((name, min(idx), max(idx)))
    known = {t for _, ts in TASK_GROUPS for t in ts}
    other = [i for i, t in enumerate(tasks) if t not in known]
    if other:
        groups.append(("Other", min(other), max(other)))
    trans = ax.get_xaxis_transform()
    for name, a, b in groups:
        ax.plot([a - 0.4, b + 0.4], [y, y], color="#555", lw=1.0, transform=trans, clip_on=False)
        ax.text((a + b) / 2.0, y - 0.06, name, ha="center", va="top", fontsize=7.5, color="#444", transform=trans)


def _score_clim(grid: ScoreGrid) -> Tuple[float, float]:
    if grid.metric in UNIT_METRICS:
        return 0.0, 1.0
    finite = grid.mean[np.isfinite(grid.mean)]
    if finite.size == 0:
        return 0.0, 1.0
    lo, hi = float(np.min(finite)), float(np.max(finite))
    if abs(hi - lo) < 1e-9:
        return lo - 0.01, hi + 0.01
    pad = 0.05 * (hi - lo)
    return lo - pad, hi + pad


def plot_winner_mosaic(
    grids: Mapping[str, ScoreGrid],
    path: Path,
    *,
    plt,
) -> List[Path]:
    """Task × metric mosaic colored by the winning family."""
    metrics = _order_metrics([m for m, g in grids.items() if g.n_task])
    if not metrics:
        return []
    tasks = next(grids[m].tasks for m in metrics)
    # Align on the union of tasks (already ordered per grid).
    all_tasks = _order_tasks(list(dict.fromkeys(t for m in metrics for t in grids[m].tasks)))
    tasks = all_tasks
    fig_w = max(10.5, 0.72 * len(tasks) + 3.2)
    fig_h = max(5.4, 0.48 * len(metrics) + 2.4)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(-0.5, len(tasks) - 0.5)
    ax.set_ylim(len(metrics) - 0.5, -0.5)
    from matplotlib.patches import Rectangle

    for i, metric in enumerate(metrics):
        grid = grids[metric]
        index = {t: j for j, t in enumerate(grid.tasks)}
        winners = task_winners(grid)
        for j, task in enumerate(tasks):
            gj = index.get(task)
            if gj is None:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="#F7F7F7", edgecolor="white", lw=0.6))
                continue
            hits = winners[gj]
            col = grid.mean[:, gj]
            if not hits:
                any_missing = any(grid.status[k, gj] == "missing" for k in range(grid.n_fam))
                face = _MISS_FACE if any_missing else _GAP_FACE
                label = "×" if any_missing else "–"
                tc = "#666"
            elif len(hits) > 1:
                face = _TIE_FACE
                score = f"{float(np.nanmax(col)):.2f}" if np.any(np.isfinite(col)) else ""
                if len(hits) <= 2:
                    names = "/".join(_family_short(h) for h in hits)
                    label = f"{names}\n{score}" if score else names
                else:
                    label = f"TIE×{len(hits)}\n{score}" if score else f"TIE×{len(hits)}"
                tc = "#1a1a1a"
            else:
                face = _family_color(hits[0])
                label = f"{_family_short(hits[0])}\n{float(np.nanmax(col)):.2f}"
                tc = _text_color_for(face)
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=face, edgecolor="white", lw=0.7))
            ax.text(j, i, label, ha="center", va="center", fontsize=6.4, color=tc, linespacing=1.15)
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels([_task_label(t) for t in tasks], rotation=50, ha="right", fontsize=8)
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels([_metric_label(m) for m in metrics], fontsize=8.5)
    ax.set_aspect("equal")
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    model = next(iter(grids.values())).model
    ax.set_title(f"Who wins where  ·  {model}\ncell = winning family (best-per-class mean)", pad=10)
    from matplotlib.patches import Patch

    legend_fams = next(iter(grids.values())).families
    handles = [Patch(facecolor=_family_color(f), edgecolor="none", label=_family_label(f)) for f in legend_fams]
    handles += [
        Patch(facecolor=_TIE_FACE, edgecolor="#ccc", label="Tie"),
        Patch(facecolor=_MISS_FACE, edgecolor="#ccc", label="Missing (×)"),
        Patch(facecolor=_GAP_FACE, edgecolor="#ccc", label="Not applicable (–)"),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=8)
    _draw_group_brackets(ax, tasks, y=-0.22)
    fig.subplots_adjust(bottom=0.22, right=0.82)
    return _save(fig, path, plt=plt)


def _draw_score_heatmap(ax, grid: ScoreGrid, *, annotate: bool, plt, show_cbar: bool = True) -> Any:
    from matplotlib.patches import Rectangle

    vmin, vmax = _score_clim(grid)
    cmap = plt.get_cmap(_OK_CMAP).copy()
    cmap.set_bad("#F3F3F3")
    im = ax.imshow(grid.mean, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    winners = task_winners(grid)
    for i in range(grid.n_fam):
        for j in range(grid.n_task):
            st = grid.status[i, j]
            if st != "ok":
                face = _GAP_FACE if st == "gap" else _MISS_FACE
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=face, edgecolor="white", lw=0.4))
                ax.text(j, i, "–" if st == "gap" else "×", ha="center", va="center", fontsize=7, color="#555")
                continue
            if annotate:
                val = float(grid.mean[i, j])
                lo, hi = float(grid.lo[i, j]), float(grid.hi[i, j])
                if abs(hi - lo) > 1e-6:
                    label = f"{val:.2f}\n{lo:.2f}–{hi:.2f}"
                    fs = 5.6
                else:
                    label = f"{val:.2f}"
                    fs = 7.0
                # luminance of mapped color
                norm = (val - vmin) / (vmax - vmin + 1e-12)
                tc = "white" if norm > 0.62 else "#111"
                ax.text(j, i, label, ha="center", va="center", fontsize=fs, color=tc, linespacing=1.05)
    for j, hits in enumerate(winners):
        for fam in hits:
            i = grid.fam_index(fam)
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="#F5C518", lw=1.8, zorder=4))
    ax.set_xticks(range(grid.n_task))
    ax.set_xticklabels([_task_label(t) for t in grid.tasks], rotation=50, ha="right", fontsize=7.5)
    ax.set_yticks(range(grid.n_fam))
    ax.set_yticklabels([_family_label(f) for f in grid.families], fontsize=8)
    ax.tick_params(length=0)
    ax.set_title(_metric_label(grid.metric), pad=6)
    if show_cbar:
        cbar = ax.figure.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cbar.ax.tick_params(labelsize=7)
    return im


def plot_annotated_heatmap(grid: ScoreGrid, path: Path, *, plt) -> List[Path]:
    fig_w = max(11.0, 0.78 * grid.n_task + 3.0)
    fig, ax = plt.subplots(figsize=(fig_w, 3.6))
    _draw_score_heatmap(ax, grid, annotate=True, plt=plt)
    ax.set_title(
        f"{_metric_label(grid.metric)}  ·  {grid.model}\n"
        "best-per-class mean [min–max]; gold frame = task winner",
        pad=8,
    )
    _draw_group_brackets(ax, grid.tasks, y=-0.28)
    fig.subplots_adjust(bottom=0.28)
    return _save(fig, path, plt=plt)


def plot_heatmap_panel(grids: Mapping[str, ScoreGrid], path: Path, *, plt) -> List[Path]:
    metrics = _order_metrics([m for m, g in grids.items() if g.n_task])
    if not metrics:
        return []
    ncols = 2
    nrows = int(math.ceil(len(metrics) / ncols))
    n_task = max(grids[m].n_task for m in metrics)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(max(12.5, 0.55 * n_task + 6.5), 2.55 * nrows + 1.2),
        squeeze=False,
    )
    model = next(iter(grids.values())).model
    for k, metric in enumerate(metrics):
        ax = axes[k // ncols][k % ncols]
        _draw_score_heatmap(ax, grids[metric], annotate=False, plt=plt, show_cbar=True)
    for k in range(len(metrics), nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")
    fig.suptitle(
        f"Family × task scores  ·  {model}\n"
        "color = best-per-class mean; gold frame = winner;  –  not applicable;  ×  missing",
        y=1.01,
    )
    fig.tight_layout()
    return _save(fig, path, plt=plt)


def plot_gap_panel(grids: Mapping[str, ScoreGrid], path: Path, *, plt) -> List[Path]:
    from matplotlib.patches import Rectangle

    metrics = _order_metrics([m for m, g in grids.items() if g.n_task])
    if not metrics:
        return []
    ncols = 2
    nrows = int(math.ceil(len(metrics) / ncols))
    n_task = max(grids[m].n_task for m in metrics)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(max(12.5, 0.55 * n_task + 6.5), 2.55 * nrows + 1.2),
        squeeze=False,
    )
    model = next(iter(grids.values())).model
    cmap = plt.get_cmap("YlOrRd").copy()
    cmap.set_bad("#F3F3F3")
    for k, metric in enumerate(metrics):
        ax = axes[k // ncols][k % ncols]
        grid = grids[metric]
        best = np.full((1, grid.n_task), np.nan)
        for j in range(grid.n_task):
            col = grid.mean[:, j]
            if np.any(np.isfinite(col)):
                best[0, j] = float(np.max(col[np.isfinite(col)]))
        gap = best - grid.mean  # 0 = winner
        vmax = float(np.nanmax(gap)) if np.any(np.isfinite(gap)) else 1.0
        if vmax < 1e-9:
            vmax = 0.05
        ax.imshow(gap, cmap=cmap, vmin=0.0, vmax=vmax, aspect="auto")
        for i in range(grid.n_fam):
            for j in range(grid.n_task):
                st = grid.status[i, j]
                if st != "ok":
                    face = _GAP_FACE if st == "gap" else _MISS_FACE
                    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=face, edgecolor="white", lw=0.4))
                    continue
                g = float(gap[i, j])
                if g <= WIN_EPS:
                    ax.plot(j, i, marker="*", color="#1a1a1a", markersize=8, zorder=5)
                else:
                    ax.text(j, i, f"{g:.2f}", ha="center", va="center", fontsize=6.2, color="#111")
        ax.set_xticks(range(grid.n_task))
        ax.set_xticklabels([_task_label(t) for t in grid.tasks], rotation=50, ha="right", fontsize=7)
        ax.set_yticks(range(grid.n_fam))
        ax.set_yticklabels([_family_label(f) for f in grid.families], fontsize=8)
        ax.tick_params(length=0)
        ax.set_title(_metric_label(metric), pad=6)
    for k in range(len(metrics), nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")
    fig.suptitle(
        f"Gap to the task-best family  ·  {model}\n"
        "0 / star = winner; larger = farther behind (only among families that scored the task)",
        y=1.01,
    )
    fig.tight_layout()
    return _save(fig, path, plt=plt)


def plot_parallel_profiles(grids: Mapping[str, ScoreGrid], path: Path, *, plt) -> List[Path]:
    metrics = [m for m in HEADLINE_METRICS if m in grids and grids[m].n_task >= 3]
    if not metrics:
        metrics = [m for m, g in grids.items() if g.n_task >= 3][:4]
    if not metrics:
        return []
    fig, axes = plt.subplots(len(metrics), 1, figsize=(max(11.0, 0.7 * grids[metrics[0]].n_task + 3), 2.7 * len(metrics) + 0.6), sharex=False)
    if len(metrics) == 1:
        axes = [axes]
    model = grids[metrics[0]].model
    for ax, metric in zip(axes, metrics):
        grid = grids[metric]
        xs = np.arange(grid.n_task)
        for i, fam in enumerate(grid.families):
            ys = grid.mean[i].astype(float)
            ax.plot(xs, ys, color=_family_color(fam), lw=2.0, marker="o", ms=5, label=_family_label(fam), zorder=3)
            # break line across NaNs already happens with matplotlib
            lo, hi = grid.lo[i], grid.hi[i]
            for j in xs:
                if np.isfinite(ys[j]) and np.isfinite(lo[j]) and np.isfinite(hi[j]) and abs(hi[j] - lo[j]) > 1e-6:
                    ax.vlines(j, lo[j], hi[j], color=_family_color(fam), lw=1.1, alpha=0.45)
        ax.set_xticks(xs)
        ax.set_xticklabels([_task_label(t) for t in grid.tasks], rotation=40, ha="right", fontsize=7.5)
        ax.set_ylabel(_metric_label(metric))
        if grid.metric in UNIT_METRICS:
            ax.set_ylim(-0.03, 1.05)
        ax.set_xlim(-0.4, grid.n_task - 0.6)
        ax.grid(axis="y", color="#eee", lw=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=8)
    fig.suptitle(
        f"Method profiles across tasks  ·  {model}\n"
        "line = best-per-class mean; whisker = per-class min–max",
        y=1.01,
    )
    fig.tight_layout()
    return _save(fig, path, plt=plt)


def plot_radar_common(grids: Mapping[str, ScoreGrid], path: Path, *, plt) -> List[Path]:
    pair = [m for m in ("encoding_E", "roc_auc_other", "suitability") if m in grids]
    pair = pair[:2] or [m for m, g in grids.items() if np.sum(common_support_mask(g)) >= 3][:2]
    usable: List[ScoreGrid] = []
    for m in pair:
        g = grids[m]
        if int(np.sum(common_support_mask(g))) >= 3:
            usable.append(g)
    if not usable:
        return []
    fig, axes = plt.subplots(1, len(usable), figsize=(6.4 * len(usable), 6.2), subplot_kw={"polar": True})
    if len(usable) == 1:
        axes = [axes]
    model = usable[0].model
    for ax, grid in zip(axes, usable):
        mask = common_support_mask(grid)
        tasks = [t for t, keep in zip(grid.tasks, mask) if keep]
        idx = [j for j, keep in enumerate(mask) if keep]
        n = len(tasks)
        angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        angles_c = np.concatenate([angles, angles[:1]])
        for i, fam in enumerate(grid.families):
            ys = np.array([float(grid.mean[i, j]) for j in idx], dtype=float)
            ys_c = np.concatenate([ys, ys[:1]])
            ax.plot(angles_c, ys_c, color=_family_color(fam), lw=2.0, label=_family_label(fam))
            ax.fill(angles_c, ys_c, color=_family_color(fam), alpha=0.10)
        ax.set_xticks(angles)
        ax.set_xticklabels([_task_label(t) for t in tasks], fontsize=7.5)
        if grid.metric in UNIT_METRICS:
            ax.set_ylim(0.0, 1.0)
            ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_title(f"{_metric_label(grid.metric)}\ncommon-support tasks only", pad=16, fontsize=10)
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.12, 1.08), frameon=False, fontsize=8)
    fig.suptitle(f"Common-support radar  ·  {model}", y=1.04)
    fig.tight_layout()
    return _save(fig, path, plt=plt)


def plot_rank_bumps(grids: Mapping[str, ScoreGrid], path: Path, *, plt) -> List[Path]:
    metrics = [m for m in ("encoding_E", "suitability", "roc_auc_other", "class_exclusivity") if m in grids]
    metrics = [m for m in metrics if grids[m].n_task >= 3][:3]
    if not metrics:
        return []
    fig, axes = plt.subplots(len(metrics), 1, figsize=(max(11.0, 0.7 * grids[metrics[0]].n_task + 3), 2.8 * len(metrics)))
    if len(metrics) == 1:
        axes = [axes]
    model = grids[metrics[0]].model
    for ax, metric in zip(axes, metrics):
        grid = grids[metric]
        xs = np.arange(grid.n_task)
        for i, fam in enumerate(grid.families):
            ranks = []
            for j in range(grid.n_task):
                col = grid.mean[:, j]
                if not np.isfinite(grid.mean[i, j]):
                    ranks.append(np.nan)
                    continue
                ranks.append(1.0 + sum(1 for v in col if np.isfinite(v) and float(v) > float(grid.mean[i, j]) + WIN_EPS))
            ax.plot(xs, ranks, color=_family_color(fam), lw=2.1, marker="o", ms=6, label=_family_label(fam))
        ax.set_yticks([1, 2, 3, 4])
        ax.set_ylim(4.4, 0.6)
        ax.set_ylabel("Rank (1 = best)")
        ax.set_xticks(xs)
        ax.set_xticklabels([_task_label(t) for t in grid.tasks], rotation=40, ha="right", fontsize=7.5)
        ax.set_title(_metric_label(metric), loc="left", fontsize=10)
        ax.grid(axis="y", color="#eee")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=8)
    fig.suptitle(f"Rank across tasks  ·  {model}", y=1.01)
    fig.tight_layout()
    return _save(fig, path, plt=plt)


def plot_head_to_head(grids: Mapping[str, ScoreGrid], path: Path, *, plt) -> List[Path]:
    metrics = [m for m in HEADLINE_METRICS if m in grids]
    if not metrics:
        metrics = _order_metrics(list(grids))[:5]
    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(3.35 * n + 1.6, 4.0), squeeze=False)
    model = next(iter(grids.values())).model
    families = list(next(iter(grids.values())).families)
    for ax, metric in zip(axes[0], metrics):
        wins = pairwise_wins(grids[metric])
        im = ax.imshow(wins, cmap="PuBuGn", vmin=0, aspect="equal")
        for a in range(len(families)):
            for b in range(len(families)):
                if a == b:
                    ax.add_patch(
                        __import__("matplotlib.patches", fromlist=["Rectangle"]).Rectangle(
                            (b - 0.5, a - 0.5), 1, 1, facecolor="#EFEFEF", edgecolor="white"
                        )
                    )
                    continue
                ax.text(b, a, str(int(wins[a, b])), ha="center", va="center", fontsize=9, color="#111")
        ax.set_xticks(range(len(families)))
        ax.set_yticks(range(len(families)))
        ax.set_xticklabels([_family_short(f) for f in families], fontsize=7)
        ax.set_yticklabels([_family_short(f) for f in families], fontsize=7)
        ax.set_xlabel("loses to →")
        ax.set_ylabel("wins ↓" if ax is axes[0][0] else "")
        ax.set_title(_metric_label(metric), fontsize=9)
        ax.tick_params(length=0)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(
        f"Head-to-head task wins  ·  {model}\n"
        "cell (row, col) = # tasks where row strictly beats column (both scored)",
        y=1.04,
    )
    fig.tight_layout()
    return _save(fig, path, plt=plt)


def plot_global_portrait(grids: Mapping[str, ScoreGrid], path: Path, *, plt) -> List[Path]:
    metrics = _order_metrics([m for m, g in grids.items() if g.n_task])
    if not metrics:
        return []
    families = list(next(iter(grids.values())).families)
    means = np.full((len(families), len(metrics)), np.nan)
    wins = np.zeros_like(means)
    cov = np.full_like(means, np.nan)
    for j, metric in enumerate(metrics):
        for row in aggregate_family_stats(grids[metric]):
            i = families.index(row["family"])
            means[i, j] = row["mean"]
            wins[i, j] = row["wins"]
            cov[i, j] = row["coverage"]
    fig_w = max(10.5, 1.15 * len(metrics) + 3.2)
    fig, ax = plt.subplots(figsize=(fig_w, 3.8))
    cmap = plt.get_cmap(_OK_CMAP).copy()
    cmap.set_bad("#F3F3F3")
    # Mix of unit and causal metrics: show per-column percentile color via rank, but annotate raw mean.
    # Color by column-wise min-max so causal ΔP remains readable next to AUCs.
    shown = np.full_like(means, np.nan)
    for j in range(len(metrics)):
        col = means[:, j]
        finite = col[np.isfinite(col)]
        if finite.size == 0:
            continue
        lo, hi = float(np.min(finite)), float(np.max(finite))
        if abs(hi - lo) < 1e-12:
            shown[:, j] = 1.0
        else:
            shown[:, j] = (col - lo) / (hi - lo)
    ax.imshow(shown, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
    for i in range(len(families)):
        for j in range(len(metrics)):
            if not np.isfinite(means[i, j]):
                ax.text(j, i, "—", ha="center", va="center", color="#666")
                continue
            label = f"{means[i, j]:.3f}\n{int(wins[i, j])} win · {cov[i, j]:.0%}"
            tc = "white" if shown[i, j] > 0.62 else "#111"
            ax.text(j, i, label, ha="center", va="center", fontsize=7.2, color=tc, linespacing=1.15)
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([_metric_label(m) for m in metrics], rotation=30, ha="right")
    ax.set_yticks(range(len(families)))
    ax.set_yticklabels([_family_label(f) for f in families])
    ax.tick_params(length=0)
    model = next(iter(grids.values())).model
    ax.set_title(
        f"Global portrait  ·  {model}\n"
        "cell = available-task mean / unique wins / coverage; color = within-metric rank",
        pad=10,
    )
    fig.tight_layout()
    return _save(fig, path, plt=plt)


def plot_range_strips(grid: ScoreGrid, path: Path, *, plt) -> List[Path]:
    """Cleveland-style mean + per-class range, one row per task."""
    if grid.n_task == 0:
        return []
    fig_h = max(5.5, 0.42 * grid.n_task + 2.0)
    fig, ax = plt.subplots(figsize=(9.2, fig_h))
    y = np.arange(grid.n_task)
    winners = task_winners(grid)
    offsets = np.linspace(-0.24, 0.24, grid.n_fam)
    xmin, xmax = _score_clim(grid)
    for i, fam in enumerate(grid.families):
        for j in range(grid.n_task):
            if not np.isfinite(grid.mean[i, j]):
                continue
            yy = y[j] + offsets[i]
            lo, hi, mu = float(grid.lo[i, j]), float(grid.hi[i, j]), float(grid.mean[i, j])
            ax.hlines(yy, lo, hi, color=_family_color(fam), lw=1.6, alpha=0.85)
            is_win = fam in winners[j] and len(winners[j]) == 1
            marker = "*" if is_win else "o"
            ms = 11 if is_win else 6
            ax.plot(mu, yy, marker=marker, color=_family_color(fam), ms=ms, zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels([_task_label(t) for t in grid.tasks])
    ax.set_xlabel(_metric_label(grid.metric))
    ax.set_xlim(xmin - 0.02, xmax + 0.02)
    ax.invert_yaxis()
    ax.grid(axis="x", color="#eee")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    from matplotlib.lines import Line2D

    handles = [
        Line2D([0], [0], color=_family_color(f), marker="o", lw=1.6, label=_family_label(f))
        for f in grid.families
    ]
    handles.append(Line2D([0], [0], color="#333", marker="*", lw=0, ms=10, label="Unique winner"))
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=8)
    ax.set_title(
        f"{_metric_label(grid.metric)}  ·  {grid.model}\n"
        "dot = best-per-class mean; line = per-class min–max; star = unique task winner",
        pad=8,
    )
    fig.tight_layout()
    return _save(fig, path, plt=plt)


def write_family_overview_figures(
    cells: Sequence[Mapping[str, Any]],
    *,
    out: Path = DEFAULT_FIG,
    model: Optional[str] = None,
    metrics: Optional[Sequence[str]] = None,
    specs: Optional[Mapping[Tuple[str, str], Mapping[str, Any]]] = None,
    families: Optional[Sequence[str]] = None,
    applicable: Optional[ApplicableFn] = None,
    plots: str = "full",
) -> List[Path]:
    """Render global comparison figures for each model. Returns written paths."""
    try:
        plt = _pyplot()
    except Exception as exc:
        print(f"family overview figures: matplotlib unavailable; skipping ({exc})")
        return []

    metric_list = _order_metrics(metrics or OVERVIEW_METRICS)
    models = sorted({str(c["model"]) for c in cells})
    if model:
        models = [m for m in models if m == model]
    written: List[Path] = []
    for mod in models:
        grids = {
            metric: build_score_grid(
                cells,
                metric=metric,
                model=mod,
                specs=specs,
                families=families,
                applicable=applicable,
            )
            for metric in metric_list
        }
        grids = {m: g for m, g in grids.items() if g.n_task}
        if not grids:
            continue
        dest = Path(out) / mod
        dest.mkdir(parents=True, exist_ok=True)
        written += plot_winner_mosaic(grids, dest / "winner_mosaic", plt=plt)
        written += plot_heatmap_panel(grids, dest / "score_heatmaps", plt=plt)
        written += plot_gap_panel(grids, dest / "gap_from_best", plt=plt)
        written += plot_global_portrait(grids, dest / "global_portrait", plt=plt)
        if "encoding_E" in grids:
            written += plot_annotated_heatmap(grids["encoding_E"], dest / "heatmap_encoding_E", plt=plt)
        if plots == "full":
            written += plot_parallel_profiles(grids, dest / "parallel_profiles", plt=plt)
            written += plot_radar_common(grids, dest / "radar_common", plt=plt)
            written += plot_rank_bumps(grids, dest / "rank_bumps", plt=plt)
            written += plot_head_to_head(grids, dest / "head_to_head", plt=plt)
            for metric in HEADLINE_METRICS:
                if metric not in grids:
                    continue
                if metric != "encoding_E":
                    written += plot_annotated_heatmap(grids[metric], dest / f"heatmap_{metric}", plt=plt)
                written += plot_range_strips(grids[metric], dest / f"range_{metric}", plt=plt)
    return written
