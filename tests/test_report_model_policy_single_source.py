"""configs/report_model_sets.json is the only definition of the paper-model scope.

Every figure/table/report entry point must derive its model population (and each
model's run-tree subdir) from ``analysis/report_model_policy.py`` instead of a
private list, so dropping a model from the JSON drops it everywhere.
"""

from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis import report_model_policy as policy  # noqa: E402

# Display-name / size lookup tables and prose that merely mention a model; these
# do not choose which models are reported.
RETIRED_MODEL_ALLOWLIST = {
    "analysis/summary_latex.py",            # MODEL_LATEX / MODEL_PARAMS lookups
    "scripts/merge_workstation_results.py",  # docstring example
    "scripts/promote_subdir_results.py",     # docstring example
    "scripts/rsync_analysis.ps1",            # AXBENCH dump directory name
    "scripts/rsync_analysis.sh",
}


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_consumers_use_the_policy_population():
    from analysis import appendix_evidence, layer_selection_table, summary_latex

    paper = tuple(policy.PAPER_MODELS)
    assert tuple(appendix_evidence.paper_model_sources()) == paper
    assert tuple(summary_latex.DEFAULT_CROSS_MODEL_SOURCES) == paper
    assert layer_selection_table.MODEL_ORDER == tuple(model for model, _ in paper)
    headline = _load_script("headline_completeness")
    assert tuple(model for model, _path in headline.DEFAULT_SUMMARIES) == tuple(m for m, _ in paper)
    for (model, subdir), (_m, path) in zip(paper, headline.DEFAULT_SUMMARIES):
        assert path == policy.summary_csv_path(model, subdir)


def test_subdir_cli_matches_policy():
    for model, subdir in (*policy.PAPER_MODELS, *policy.ABLATION_CANDIDATE_MODELS):
        out = subprocess.run(
            [sys.executable, str(ROOT / "analysis" / "report_model_policy.py"), "--subdir-of", model],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        assert out == subdir
    assert policy.subdir_for("not-in-policy") == ""


@pytest.mark.parametrize("script", ["scripts/summary_tables_pdf.sh", "slurm/_appendix_evidence_cmd.sh"])
def test_shell_entry_points_have_no_private_model_lists(script):
    text = (ROOT / script).read_text(encoding="utf-8")
    assert not re.search(r"--(source|model)[ =]+\"?[a-z0-9.\-]+-(small|deduped|\d+b\w*)\b", text), script
    assert not re.search(r"^\s*[\w.\-|]+\|[\w.\-|]+\)\s*printf", text, re.M), (
        f"{script}: per-model subdir case statement; use report_model_policy.py --subdir-of"
    )


def test_models_outside_the_policy_are_not_hardcoded_in_report_code():
    in_policy = {m for m, _ in (*policy.PAPER_MODELS, *policy.ABLATION_CANDIDATE_MODELS)}
    retired = {"gemma-2-2b"} - in_policy
    offenders = []
    for folder, patterns in (("analysis", ("*.py",)), ("scripts", ("*.py", "*.sh", "*.ps1")), ("slurm", ("*.sh",))):
        for pattern in patterns:
            for path in (ROOT / folder).glob(pattern):
                rel = path.relative_to(ROOT).as_posix()
                if rel in RETIRED_MODEL_ALLOWLIST:
                    continue
                text = path.read_text(encoding="utf-8", errors="ignore")
                if any(model in text for model in retired):
                    offenders.append(rel)
    assert not offenders, f"models dropped from configs/report_model_sets.json still referenced in: {offenders}"
