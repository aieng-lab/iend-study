"""Task registry: canonical builders + YAML ``extends`` / ``builder`` resolution."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from study.data.mib import (
    build_ioi_mib,
    build_ravel_continent,
    build_ravel_country,
    build_ravel_language,
)
from study.data.synthetic import build_circuit_task
from study.tasks import TaskBundle
from study.tasks.emotion import build_emotion
from study.tasks.gender_en import build_task as build_gender_en
from study.tasks.prior import (
    build_pronoun_number,
    build_pronoun_person,
    build_race,
    build_religion,
)

Builder = Callable[..., TaskBundle]

CIRCUIT_TASKS = (
    "ioi",
    "key_value",
    "induction",
    "function_composition",
    "repetition",
    "language",
)


def _circuit(task_id: str) -> Builder:
    def _build(cfg: Dict[str, Any], *, smoke: bool = False) -> TaskBundle:
        return build_circuit_task(task_id, cfg, smoke=smoke)

    return _build


# Canonical builders only. Variants (gender_en_pre, race_one_pole, …) resolve via
# YAML ``extends:`` or optional ``builder:`` — do not hardcode every leaf id here.
TASK_BUILDERS: Dict[str, Builder] = {
    "gender_en": build_gender_en,
    "race": build_race,
    "religion": build_religion,
    "pronoun_person": build_pronoun_person,
    "pronoun_number": build_pronoun_number,
    "emotion": build_emotion,
    "ioi_mib": build_ioi_mib,
    "ravel_continent": build_ravel_continent,
    "ravel_country": build_ravel_country,
    "ravel_language": build_ravel_language,
    **{tid: _circuit(tid) for tid in CIRCUIT_TASKS},
}


def _yaml_builder_hint(task_id: str) -> Optional[str]:
    """Return ``builder:`` from the leaf YAML only (not merged extends)."""
    from study.config import CONFIGS, _load_yaml

    path = CONFIGS / "tasks" / f"{task_id}.yaml"
    if not path.is_file():
        return None
    raw = _load_yaml(path)
    hint = raw.get("builder")
    return str(hint).strip() if hint else None


def _extends_chain(task_id: str) -> List[str]:
    """Leaf → … → base task ids from ``extends:`` (leaf first)."""
    from study.config import CONFIGS, _load_yaml

    seen: List[str] = []
    current = str(task_id)
    while current and current not in seen:
        seen.append(current)
        path = CONFIGS / "tasks" / f"{current}.yaml"
        if not path.is_file():
            break
        parent = (_load_yaml(path) or {}).get("extends")
        if not parent:
            break
        current = str(parent).strip()
    return seen


def resolve_builder_id(task_id: str) -> str:
    """Map a task YAML id to a key in ``TASK_BUILDERS``.

    Order:
      1. exact ``task_id`` registered
      2. leaf YAML ``builder:`` field
      3. first ancestor in ``extends:`` chain that is registered
    """
    tid = str(task_id).strip()
    if tid in TASK_BUILDERS:
        return tid

    hint = _yaml_builder_hint(tid)
    if hint:
        if hint not in TASK_BUILDERS:
            raise ValueError(
                f"Task {tid!r} sets builder={hint!r}, but that builder is not "
                f"registered. Known builders: {sorted(TASK_BUILDERS)}"
            )
        return hint

    for ancestor in _extends_chain(tid)[1:]:
        if ancestor in TASK_BUILDERS:
            return ancestor

    raise ValueError(
        f"Unknown task {tid!r}: not in TASK_BUILDERS and no resolvable "
        f"extends:/builder: chain. Known builders: {sorted(TASK_BUILDERS)}. "
        f"Add configs/tasks/{tid}.yaml with extends: <base> or builder: <base>."
    )


def build_task(task_id: str, cfg: Dict[str, Any], *, smoke: bool = False) -> TaskBundle:
    builder_id = resolve_builder_id(task_id)
    return TASK_BUILDERS[builder_id](cfg, smoke=smoke)
