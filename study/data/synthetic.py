"""Synthetic circuit / language task builders (study-owned).

Default size: ~10k rows per factual class (8k/1k/1k train/val/test).

Protocols (one-pole factual; ``source=both`` swaps poles in training):

- **ioi**: namexact-style ABBA/BABA copy; factual=IO, alt=SUBJECT
- **key_value**: ``k1 = red; k2 = blue; k1 = [MASK]`` → ``red`` (VALUE vs OTHER)
- **induction**: n-gram / induction heads; factual=MATCH continuation
- **function_composition**: assignment chains ``x=red, a=blue, y=x, y=[MASK]`` → ``red``
- **repetition**: controlled next-token confidence sweep;
  ``Sequence: yes ... yes [MASK]`` → ``yes`` (alternative ``no``)
- **language**: 3-class language-of-the-text cloze (real OPUS sides + MUSE
  rival-class alternatives). Default classes ``en/fr/de``; neutrals are other
  OPUS languages (``es/it/nl/pt``) in the same ``[lang] prefix[MASK]`` wrapper.
  Smoke uses an offline fixture; the study cache is not templated glosses.

Difficulty (``key_value`` / ``induction`` / ``function_composition``):
  Set ``data.difficulty: easy|medium|hard`` (default **easy**). Each level gets a
  separate cache under ``data/synthetic/{task}_{difficulty}.csv``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from study.tasks import ROOT, TaskBundle, load_neutral_from_cfg

SYN_DIR = ROOT / "data" / "synthetic"
# Bumped: namexact-sourced IOI names, expanded/rounded value & induction-token
# pools, disjoint per-split vocab for the predicted-token vocabularies (was
# independent per-split draws from one shared pool -- see synthetic.py's
# vocab-pool section docstring). Old caches under this version are stale.
DATA_VERSION = 11
KEY_VALUE_DATA_VERSION = 12
INDUCTION_DATA_VERSION = 12
REPETITION_DATA_VERSION = 3
# Separate from DATA_VERSION on purpose (mirrors REPETITION_DATA_VERSION):
# bumping the shared DATA_VERSION would force ioi/key_value/induction/
# function_composition to regenerate and retrain too, for a bug that only
# affects language's neutral rows -- not acceptable mid an already-running
# full-suite rerun. Bumped 2026-08-22: build_language_id's neutral rows used
# to set "text" to the already-built cloze ("[lang] ...[MASK]..."), not the
# original raw sentence -- crashed live in gradiend's
# create_masked_pair_from_text, which requires genuinely unmasked text so it
# can derive its own mask position. Any language_neutral.csv written before
# this fix carries the bad text and must be regenerated, not reused.
LANGUAGE_DATA_VERSION = 1
REPETITION_DELIMITER_VARIANT_VERSION = 1

# ~10k per factual class
DEFAULT_ROWS: Dict[str, int] = {"train": 8000, "validation": 1000, "test": 1000}
SMOKE_ROWS: Dict[str, int] = {"train": 64, "validation": 16, "test": 16}

DIFFICULTIES: Tuple[str, ...] = ("easy", "medium", "hard")
DEFAULT_DIFFICULTY = "easy"
# Tasks that support graded difficulty (separate caches per level).
DIFFICULTY_TASKS: Tuple[str, ...] = ("key_value", "induction", "function_composition")
# Study mix: skip n=1 (too little confidence). Theory E1 still passes its own counts.
REPETITION_COUNTS: Tuple[int, ...] = (4, 8, 12, 16, 32)
REPETITION_DELIMITER_PREFIXES: Dict[str, str] = {
    # Inserted before the space+[MASK] slot. Comma + " [MASK]" / label "yes"
    # renders as ", yes" and scores the word-boundary continuation.
    "space": "",
    "comma": ",",
    "semicolon": ";",
    "pipe": " |",
}
DEFAULT_REPETITION_DELIMITERS: Tuple[str, ...] = ("space",)

_REPETITION_ROLES: Tuple[str, ...] = (
    "analyst", "editor", "operator", "reviewer", "student", "teacher",
    "visitor", "writer", "researcher", "coordinator", "reader", "planner",
)
_REPETITION_TOPICS: Tuple[str, ...] = (
    "weather", "music", "travel", "books", "cooking", "science",
    "sports", "history", "gardening", "design", "movies", "language",
)
_REPETITION_CONTEXT_TEMPLATES: Tuple[str, ...] = (
    "Survey {context_id} for the {role} about {topic}; recent answers: ",
    "Record {context_id} | role={role} | topic={topic} | response history: ",
    "Case {context_id}: the {role} reviewed {topic}. Previous binary responses: ",
    "Session {context_id}. Category: {topic}. The {role}'s recent replies: ",
    "Binary log {context_id} ({topic}, {role}): ",
    "For checklist {context_id} on {topic}, the {role} answered: ",
    "Item {context_id}: {role}; subject={topic}; recorded choices: ",
    "Case {context_id}: a {role} considered {topic}. The latest responses were: ",
)


def _write_csv(path: Path, df: pd.DataFrame, *, extra_meta: Optional[Dict[str, Any]] = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    meta = path.with_suffix(".meta.json")
    payload: Dict[str, Any] = {
        "version": DATA_VERSION,
        "n": len(df),
        "classes": sorted(df["label_class"].astype(str).unique().tolist())
        if "label_class" in df.columns
        else [],
    }
    if extra_meta:
        payload.update(extra_meta)
    meta.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _rows_ok(path: Path, *, min_n: int, expected_version: int = DATA_VERSION) -> bool:
    if not path.is_file():
        return False
    meta = path.with_suffix(".meta.json")
    if meta.is_file():
        try:
            m = json.loads(meta.read_text(encoding="utf-8"))
            return int(m.get("version", 0)) == int(expected_version) and int(m.get("n", 0)) >= min_n
        except Exception:
            return False
    return False


def _data_version_for_task(task_id: str) -> int:
    return {
        "key_value": KEY_VALUE_DATA_VERSION,
        "induction": INDUCTION_DATA_VERSION,
        "repetition": REPETITION_DATA_VERSION,
        "language": LANGUAGE_DATA_VERSION,
    }.get(task_id, DATA_VERSION)


def _language_artifact_ok(
    path: Path,
    *,
    expected_version: int,
    min_n: int,
    classes: Sequence[str],
    neutral: bool = False,
) -> bool:
    if not _rows_ok(path, min_n=min_n, expected_version=expected_version):
        return False
    try:
        df = pd.read_csv(path)
    except (OSError, ValueError, pd.errors.EmptyDataError):
        return False
    key_col = "text" if neutral else "masked"
    if key_col not in df.columns or "split" not in df.columns:
        return False
    if df[key_col].isna().any() or df[key_col].astype(str).duplicated().any():
        return False
    if set(df["split"].astype(str)) != {"train", "validation", "test"}:
        return False
    if not neutral:
        if "label_class" not in df.columns:
            return False
        if set(df["label_class"].astype(str)) != set(map(str, classes)):
            return False
    return True


# ---------------------------------------------------------------------------
# Shared lexicons for variance
#
# Design rule: only vocabulary that can appear as the *predicted* [MASK] token
# (IOI names, key_value/function_composition VALUEs, induction tokens) needs
# single-token verification against the study's tokenizers -- everything else
# (places, filler objects/verbs, key/variable identifiers) is never scored, so
# it carries no tokenization constraint and is sized purely for readability.
#
# Split policy: pools that back the predicted token are partitioned into
# disjoint train/validation/test sub-pools (see ``_split_pool`` /
# ``_SPLIT_POOLS``) so a held-out example can use content genuinely absent
# from training, not just a new combination of already-seen content. Purely
# structural/filler vocab (places, key/variable identifiers) is intentionally
# *not* split: the model has certainly seen "x = 5"-style variable names or
# "k1 =>"-style key strings regardless of which specific string appears, so
# reusing them across splits doesn't leak predicted content the way reusing a
# predicted name/value/token would.
# ---------------------------------------------------------------------------

_NAMES = [
    "alice", "bob", "carol", "david", "eve", "frank", "grace", "heidi",
    "ivan", "judy", "kyle", "laura", "mike", "nina", "omar", "paula",
    "quinn", "riley", "sam", "tina", "uma", "victor", "wendy", "xavier",
    "yvonne", "zack", "amy", "ben", "claire", "dan", "ella", "finn",
    "gina", "harry", "iris", "jake", "kate", "leo", "maya", "noah",
]

# IOI names come from aieng-lab/namexact (see ``_ioi_name_pools``), not a
# hand-typed list -- see that function's docstring for sourcing/caching.

_COLORS = [
    "red", "blue", "green", "yellow", "orange", "purple", "pink", "brown",
    "black", "white", "gray", "cyan", "gold", "silver", "violet", "coral",
    "lime", "rose", "sand", "rust", "slate", "bronze", "copper", "mint", "plum",
]

_ANIMALS = [
    "cat", "dog", "bird", "fish", "horse", "tiger", "lion", "bear",
    "wolf", "fox", "deer", "eagle", "shark", "whale", "mouse", "rabbit",
    "goat", "sheep", "duck", "crow", "frog", "snake", "turtle", "seal", "camel",
]

_OBJECTS = [
    "book", "key", "box", "cup", "pen", "bag", "hat", "map",
    "coin", "ring", "lamp", "chair", "desk", "door", "stone", "rope",
    "clock", "knife", "spoon", "fork", "plate", "bowl", "jar", "mirror", "candle",
]

# All three verified single-token (GPT-2 + Pythia-NeoX) -- see
# ``scripts/verify_synthetic_vocab.py``. 25 each, one round number reused
# across categories rather than three arbitrary counts.
assert len(_COLORS) == len(_ANIMALS) == len(_OBJECTS) == 25

_PLACES = [
    "the park", "the cafe", "the office", "the museum", "the library", "the station",
    "the market", "the school", "the hospital", "the bakery", "the beach", "the hotel",
    "the garden", "the studio", "the clinic", "the campus", "the harbor", "the plaza",
    "the theater", "the airport", "the stadium", "the palace", "the tower", "the village",
    "the harbor town",
]

_IOI_OBJECTS = [
    "a book", "a gift", "a letter", "a key", "a drink", "a ticket",
    "a map", "a note", "a parcel", "a flower", "a photo", "a pen",
    "a bag", "a box", "a card", "a message", "a present", "a bottle",
    "a report", "a folder", "a laptop", "a phone", "a coat", "a hat",
    "an umbrella",
]

_IOI_VERBS = [
    "gave", "handed", "sent", "brought", "passed", "offered",
    "delivered", "showed", "lent", "returned", "tossed", "sold",
    "mailed", "carried", "presented", "forwarded", "shipped", "granted",
    "threw", "slid", "pushed", "dropped", "left", "gifted", "loaned",
]

# _PLACES / _IOI_OBJECTS / _IOI_VERBS never appear as the predicted [MASK]
# token, so they carry no tokenization constraint -- rounded to 25 to match
# _COLORS/_ANIMALS/_OBJECTS, not because 25 has any other significance here.
assert len(_PLACES) == len(_IOI_OBJECTS) == len(_IOI_VERBS) == 25

_VALUES = _COLORS + _ANIMALS + _OBJECTS  # 75, split disjointly -- see below

# Single ASCII letters: never the predicted token (query_var is context, the
# predicted token is the bound _VALUES entry), so no tokenization constraint;
# using the full alphabet instead of an unexplained 21-letter subset.
_VAR_NAMES = list("abcdefghijklmnopqrstuvwxyz")

# Systematic key-identifier scheme (k1..k20, key1..key5, slot1..slot5 = 30)
# rather than a hand-picked assortment -- also never the predicted token.
_KEY_NAMES = (
    [f"k{i}" for i in range(1, 21)]
    + [f"key{i}" for i in range(1, 6)]
    + [f"slot{i}" for i in range(1, 6)]
)

_SEP_BIND = [" = ", "=", " : ", ": ", " -> ", " => "]
_SEP_STMT = ["; ", ". ", ", ", " | ", " / ", "\n"]

# Induction's predicted token is drawn from this same vocab (unit[1]), so it
# needs the tokenization guarantee -- 26 letters + 24 verified single-token
# short words, split disjointly like _VALUES (see below).
_INDUCTION_WORDS_SAFE = [
    "red", "blue", "cat", "dog", "one", "two", "apple", "pear", "left", "right",
    "green", "black", "white", "fish", "bird", "frog", "zero", "five", "six", "nine",
    "plum", "rose", "sand", "rock",
]
assert len(_INDUCTION_WORDS_SAFE) == 24
_INDUCTION_TOKENS_SAFE = _VAR_NAMES + _INDUCTION_WORDS_SAFE  # 50
_INDUCTION_TOKENS = [f"t{i}" for i in range(40)] + list(_INDUCTION_TOKENS_SAFE)


# ---------------------------------------------------------------------------
# Disjoint per-split vocabulary pools (fixes train/test content leakage for
# the predicted-token vocabularies: IOI names, key_value/function_composition
# VALUEs, induction tokens). Split once with a fixed seed, independent of a
# given generation call's row counts, so which entries land in which split
# stays stable across regenerations.
# ---------------------------------------------------------------------------

_VOCAB_SPLIT_SEED = 20260819
_VOCAB_SPLIT_FRACTIONS: Dict[str, float] = {"train": 0.8, "validation": 0.1, "test": 0.1}


def _split_pool(
    pool: Sequence[str],
    *,
    seed: int = _VOCAB_SPLIT_SEED,
    fractions: Optional[Dict[str, float]] = None,
) -> Dict[str, List[str]]:
    """Partition ``pool`` into disjoint train/validation/test sub-pools.

    Deterministic given ``pool``/``seed``/``fractions``; independent of any
    particular generation call's requested row counts.
    """
    fractions = fractions or _VOCAB_SPLIT_FRACTIONS
    items = list(pool)
    np.random.default_rng(seed).shuffle(items)
    n = len(items)
    out: Dict[str, List[str]] = {}
    cursor = 0
    keys = list(fractions)
    for i, key in enumerate(keys):
        if i == len(keys) - 1:
            out[key] = items[cursor:]
        else:
            take = int(round(n * fractions[key]))
            out[key] = items[cursor : cursor + take]
            cursor += take
    return out


_VALUES_BY_SPLIT: Dict[str, List[str]] = _split_pool(_VALUES)
# A smaller total pool (50) than _VALUES (75): use a less extreme split so
# validation/test still comfortably support unit_len up to 3 (medium preset)
# plus 3 distinct distractors drawn from the same per-split pool.
_INDUCTION_TOKENS_SAFE_BY_SPLIT: Dict[str, List[str]] = _split_pool(
    _INDUCTION_TOKENS_SAFE, fractions={"train": 0.52, "validation": 0.24, "test": 0.24}
)


def _pool_for_split(by_split: Dict[str, List[str]], split: str) -> List[str]:
    key = "validation" if str(split).lower() in {"validation", "val"} else str(split).lower()
    pool = by_split.get(key)
    if not pool:
        raise ValueError(f"no vocab pool for split {split!r} (have {sorted(by_split)})")
    return pool


# ---------------------------------------------------------------------------
# IOI names: aieng-lab/namexact (citable, HF), filtered to names that are a
# single token for every tokenizer this study trains circuit tasks on.
# ---------------------------------------------------------------------------

_IOI_TOKENIZERS = ("gpt2", "EleutherAI/pythia-70m-deduped")
_IOI_NAMES_CACHE = SYN_DIR / "cache" / "ioi_names.json"

# Hand-picked fallback so generation still works offline / without `datasets`
# installed; used only if the namexact load fails. Not tokenizer-verified.
_IOI_NAMES_FALLBACK = [
    "John", "Mary", "James", "Robert", "Michael", "William", "David", "Richard",
    "Joseph", "Thomas", "Charles", "Daniel", "Matthew", "Anthony", "Donald", "Steven",
    "Paul", "Andrew", "Joshua", "Kenneth", "Kevin", "Brian", "George", "Timothy",
    "Edward", "Jason", "Jeffrey", "Ryan", "Jacob", "Gary", "Eric", "Stephen",
    "Larry", "Justin", "Scott", "Brandon", "Samuel", "Frank", "Patrick", "Jack",
    "Aaron", "Adam", "Nathan", "Henry", "Peter", "Kyle", "Sam", "Max",
]


def _build_ioi_name_pools() -> Dict[str, List[str]]:
    """Load namexact, keep names single-token for every ``_IOI_TOKENIZERS``
    entry, and reuse namexact's own train/val/test split (relabeling its
    ``val`` to this study's ``validation``).

    Tokenizer intersection is deliberate: this study currently trains circuit
    tasks on gpt2-small and pythia-70m-deduped from one shared CSV per task,
    so a name must be single-token for *both* to keep that file valid for
    both models. If a third tokenizer with a materially different vocab ever
    joins the circuit-task suite, switch to one cached name pool per
    tokenizer (cache key already includes the tokenizer tuple) rather than
    further narrowing the intersection.

    No gender rebalancing: unlike gender_en, IOI's causal claim (subject vs.
    indirect-object position) does not depend on gender, so there is no
    confound to control for by matching M/F counts -- using every verified
    name maximizes pool size instead.
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer

    ds = load_dataset("aieng-lab/namexact", split="train")
    tokenizers = [AutoTokenizer.from_pretrained(t) for t in _IOI_TOKENIZERS]

    def _single_token_everywhere(name: str) -> bool:
        return all(len(tok.encode(" " + name)) == 1 for tok in tokenizers)

    pools: Dict[str, List[str]] = {"train": [], "validation": [], "test": []}
    for row in ds:
        split = "validation" if str(row["split"]) == "val" else str(row["split"])
        if split not in pools:
            continue
        if _single_token_everywhere(str(row["name"])):
            pools[split].append(str(row["name"]))
    for split, names in pools.items():
        if not names:
            raise RuntimeError(f"namexact yielded no single-token names for split {split!r}")
    return pools


def _ioi_name_pools() -> Dict[str, List[str]]:
    if _IOI_NAMES_CACHE.is_file():
        try:
            cached = json.loads(_IOI_NAMES_CACHE.read_text(encoding="utf-8"))
            if isinstance(cached, dict) and cached.get("tokenizers") == list(_IOI_TOKENIZERS):
                pools = cached.get("pools") or {}
                if all(pools.get(s) for s in ("train", "validation", "test")):
                    return pools
        except Exception:
            pass
    try:
        pools = _build_ioi_name_pools()
    except Exception as exc:
        print(f"  ioi: namexact load failed ({exc}); using fallback name list", flush=True)
        return _split_pool(_IOI_NAMES_FALLBACK)
    _IOI_NAMES_CACHE.parent.mkdir(parents=True, exist_ok=True)
    _IOI_NAMES_CACHE.write_text(
        json.dumps({"tokenizers": list(_IOI_TOKENIZERS), "pools": pools}, indent=2),
        encoding="utf-8",
    )
    return pools

# Inclusive (lo, hi) ranges + format / vocab flags per difficulty.
DIFFICULTY_PRESETS: Dict[str, Dict[str, Dict[str, Any]]] = {
    "key_value": {
        "easy": {"n_bindings": (2, 2), "format_noise": False},
        "medium": {"n_bindings": (3, 4), "format_noise": True},
        "hard": {"n_bindings": (5, 5), "format_noise": True},
    },
    "induction": {
        "easy": {
            # Three-token units leave >1,000 distinct prompt-identifiable
            # sequences in the 12-token validation/test vocabularies. The old
            # two-token form had only 42 distinct test prompts for 1,000 rows.
            "unit_len": (3, 3),
            "repeats": (1, 1),
            "noise": (0, 0),
            "format_noise": False,
            "single_token_vocab": True,
        },
        "medium": {
            "unit_len": (3, 3),
            "repeats": (1, 2),
            "noise": (0, 1),
            "format_noise": True,
            "single_token_vocab": True,
        },
        "hard": {
            "unit_len": (4, 5),
            "repeats": (1, 3),
            "noise": (0, 3),
            "format_noise": True,
            "single_token_vocab": False,
        },
    },
    "function_composition": {
        "easy": {"chain_depth": (1, 1), "format_noise": False},
        "medium": {"chain_depth": (2, 2), "format_noise": True},
        "hard": {"chain_depth": (3, 3), "format_noise": True},
    },
}




def _split_counts(rows_per_split: Optional[Dict[str, int]]) -> Dict[str, int]:
    return dict(rows_per_split or DEFAULT_ROWS)


def _normalize_difficulty(difficulty: Optional[str]) -> str:
    d = str(difficulty or DEFAULT_DIFFICULTY).strip().lower()
    if d not in DIFFICULTIES:
        raise ValueError(f"difficulty must be one of {DIFFICULTIES}, got {difficulty!r}")
    return d


def _rand_inclusive(rng: np.random.Generator, lo_hi: Tuple[int, int]) -> int:
    lo, hi = int(lo_hi[0]), int(lo_hi[1])
    if hi < lo:
        raise ValueError(f"invalid inclusive range {lo_hi!r}")
    return int(rng.integers(lo, hi + 1))


def _resolve_preset(
    task_id: str,
    difficulty: Optional[str],
    *,
    overrides: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Dict[str, Any]]:
    d = _normalize_difficulty(difficulty)
    preset = dict(DIFFICULTY_PRESETS[task_id][d])
    if overrides:
        for k, v in overrides.items():
            if v is not None:
                preset[k] = v
    return d, preset


def _cache_stem(task_id: str, *, difficulty: Optional[str] = None, smoke: bool = False) -> str:
    """Filename stem for synthetic caches (no ``.csv``)."""
    if task_id in DIFFICULTY_TASKS:
        d = _normalize_difficulty(difficulty)
        stem = f"{task_id}_{d}"
    else:
        stem = task_id
    return f"{stem}_smoke" if smoke else stem


# ---------------------------------------------------------------------------
# IOI
# ---------------------------------------------------------------------------

_IOI_TEMPLATES = [
    "When {s} and {io} went to {place}, {s} {verb} {obj} to [MASK]",
    "After {s} met {io} at {place}, {s} {verb} {obj} to [MASK]",
    "{s} and {io} were at {place}. Then {s} {verb} {obj} to [MASK]",
    "At {place}, {s} saw {io} and {s} {verb} {obj} to [MASK]",
    "Because {s} and {io} visited {place}, {s} {verb} {obj} to [MASK]",
    "While {io} waited at {place}, {s} arrived and {s} {verb} {obj} to [MASK]",
    "Earlier {s} and {io} left {place}; later {s} {verb} {obj} to [MASK]",
]


def build_ioi(*, rows_per_split: Optional[Dict[str, int]] = None, seed: int = 0) -> pd.DataFrame:
    rows_per_split = _split_counts(rows_per_split)
    rng = np.random.default_rng(seed)
    name_pools = _ioi_name_pools()
    rows: List[dict] = []
    for split, n in rows_per_split.items():
        names = np.asarray(_pool_for_split(name_pools, split), dtype=object)
        for _ in range(int(n)):
            s, io = rng.choice(names, size=2, replace=False)
            s_c, io_c = str(s), str(io)
            place = str(rng.choice(_PLACES))
            obj = str(rng.choice(_IOI_OBJECTS))
            verb = str(rng.choice(_IOI_VERBS))
            tmpl = str(rng.choice(_IOI_TEMPLATES))
            # ABBA vs BABA: which name repeats as subject
            if bool(rng.integers(0, 2)):
                pattern = "ABBA"
                masked = tmpl.format(s=s_c, io=io_c, place=place, verb=verb, obj=obj)
                label, alt = str(io), str(s)
            else:
                pattern = "BABA"
                # swap roles in template slots so the repeated subject is io_c
                masked = tmpl.format(s=io_c, io=s_c, place=place, verb=verb, obj=obj)
                label, alt = str(s), str(io)
            rows.append(
                {
                    "masked": masked,
                    "label": label,
                    "label_class": "IO",
                    "alternative": alt,
                    "alternative_class": "SUBJECT",
                    "pattern": pattern,
                    "split": split,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Key–value binding: k1 = red; k2 = blue; k1 = [MASK]
# ---------------------------------------------------------------------------

def _format_binding(key: str, value: str, sep: str) -> str:
    return f"{key}{sep}{value}"


def build_key_value(
    *,
    rows_per_split: Optional[Dict[str, int]] = None,
    seed: int = 0,
    difficulty: str = DEFAULT_DIFFICULTY,
    n_bindings: Optional[Tuple[int, int]] = None,
    format_noise: Optional[bool] = None,
) -> pd.DataFrame:
    """Factual VALUE at queried key; alternative = other binding's value (OTHER)."""
    diff, preset = _resolve_preset(
        "key_value",
        difficulty,
        overrides={"n_bindings": n_bindings, "format_noise": format_noise},
    )
    bind_range: Tuple[int, int] = tuple(preset["n_bindings"])  # type: ignore[assignment]
    noise = bool(preset["format_noise"])
    rows_per_split = _split_counts(rows_per_split)
    rng = np.random.default_rng(seed)
    rows: List[dict] = []
    for split, n in rows_per_split.items():
        values = _pool_for_split(_VALUES_BY_SPLIT, split)
        seen_masked = set()
        attempts = 0
        while len(seen_masked) < int(n):
            attempts += 1
            if attempts > max(10_000, int(n) * 100):
                raise RuntimeError(
                    f"Could not generate {n} unique key-value prompts for split={split}; "
                    f"only found {len(seen_masked)}. Expand the construction space."
                )
            n_bind = _rand_inclusive(rng, bind_range)
            keys = list(rng.choice(_KEY_NAMES, size=n_bind, replace=False))
            vals = list(rng.choice(values, size=n_bind, replace=False))
            if noise:
                sep = str(rng.choice(_SEP_BIND))
                stmt = str(rng.choice(_SEP_STMT))
                order = list(range(n_bind))
                rng.shuffle(order)
                keys_ord = [keys[i] for i in order]
                vals_ord = [vals[i] for i in order]
                prefix = str(
                    rng.choice(
                        [
                            "",
                            "Bindings: ",
                            "Store: ",
                            "Memory: ",
                            "Let ",
                            "Context — ",
                            f"Round {int(rng.integers(1, 20))}: ",
                        ]
                    )
                )
            else:
                sep, stmt, prefix = " = ", "; ", ""
                keys_ord, vals_ord = keys, vals

            bindings = [_format_binding(k, v, sep) for k, v in zip(keys_ord, vals_ord)]
            q_idx = int(rng.integers(0, n_bind))
            q_key = keys_ord[q_idx]
            q_val = vals_ord[q_idx]
            other_idx = (q_idx + 1 + int(rng.integers(0, n_bind - 1))) % n_bind
            other_val = vals_ord[other_idx]

            body = stmt.join(bindings)
            if noise:
                query = str(
                    rng.choice(
                        [
                            f"{stmt}{q_key}{sep}[MASK]",
                            f"{stmt}recall {q_key}{sep}[MASK]",
                            f"{stmt}{q_key} again{sep}[MASK]",
                            f"{stmt}what is {q_key}{sep}[MASK]",
                            f"{stmt}lookup {q_key}{sep}[MASK]",
                        ]
                    )
                )
            else:
                query = f"{stmt}{q_key}{sep}[MASK]"
            masked = f"{prefix}{body}{query}"
            if masked in seen_masked:
                continue
            seen_masked.add(masked)
            rows.append(
                {
                    "masked": masked,
                    "label": q_val,
                    "label_class": "VALUE",
                    "alternative": other_val,
                    "alternative_class": "OTHER",
                    "query_key": q_key,
                    "n_bindings": n_bind,
                    "difficulty": diff,
                    "split": split,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Induction
# ---------------------------------------------------------------------------

def build_induction(
    *,
    rows_per_split: Optional[Dict[str, int]] = None,
    seed: int = 0,
    difficulty: str = DEFAULT_DIFFICULTY,
    unit_len: Optional[Tuple[int, int]] = None,
    repeats: Optional[Tuple[int, int]] = None,
    noise: Optional[Tuple[int, int]] = None,
    format_noise: Optional[bool] = None,
    single_token_vocab: Optional[bool] = None,
) -> pd.DataFrame:
    diff, preset = _resolve_preset(
        "induction",
        difficulty,
        overrides={
            "unit_len": unit_len,
            "repeats": repeats,
            "noise": noise,
            "format_noise": format_noise,
            "single_token_vocab": single_token_vocab,
        },
    )
    unit_range: Tuple[int, int] = tuple(preset["unit_len"])  # type: ignore[assignment]
    repeats_range: Tuple[int, int] = tuple(preset["repeats"])  # type: ignore[assignment]
    noise_range: Tuple[int, int] = tuple(preset["noise"])  # type: ignore[assignment]
    fmt_noise = bool(preset["format_noise"])
    safe_vocab = bool(preset["single_token_vocab"])
    rows_per_split = _split_counts(rows_per_split)
    rng = np.random.default_rng(seed)
    rows: List[dict] = []
    for split, n in rows_per_split.items():
        # Disjoint per-split token pool only for the tokenizer-verified safe
        # vocab (easy/medium): that's the vocab the predicted token is drawn
        # from. The unrestricted "hard" vocab (single_token_vocab=False) is
        # not currently used by any configured task; keep it shared for now.
        vocab = np.asarray(
            _pool_for_split(_INDUCTION_TOKENS_SAFE_BY_SPLIT, split)
            if safe_vocab
            else _INDUCTION_TOKENS,
            dtype=object,
        )
        seen_masked = set()
        attempts = 0
        while len(seen_masked) < int(n):
            attempts += 1
            if attempts > max(10_000, int(n) * 100):
                raise RuntimeError(
                    f"Could not generate {n} unique induction prompts for split={split}; "
                    f"only found {len(seen_masked)}. Expand the unit/vocabulary space."
                )
            # Classic induction: [A][B] … [A] [MASK] → [B]
            k = _rand_inclusive(rng, unit_range)
            unit = list(rng.choice(vocab, size=k, replace=False))
            noise_n = _rand_inclusive(rng, noise_range)
            noise_toks = (
                list(rng.choice(vocab, size=noise_n, replace=True)) if noise_n else []
            )
            noise_toks = [t for t in noise_toks if t != unit[0]]
            n_repeats = _rand_inclusive(rng, repeats_range)
            seq: List[str] = list(noise_toks)
            for _r in range(n_repeats):
                seq.extend(unit)
            seq.append(unit[0])
            label = unit[1] if k >= 2 else unit[0]
            distractors = [t for t in unit if t != label] + list(
                rng.choice([t for t in vocab if t not in unit], size=3, replace=False)
            )
            alt = str(rng.choice(distractors))

            if fmt_noise:
                sep = str(rng.choice([" ", "  ", " | ", ","]))
                fluff = str(
                    rng.choice(
                        [
                            "",
                            "Sequence: ",
                            "Continue: ",
                            "Pattern — ",
                            "Next token after: ",
                            f"Example {int(rng.integers(1, 99))}: ",
                        ]
                    )
                )
            else:
                sep, fluff = " ", ""
            if sep == ",":
                text = ", ".join(seq) + ", [MASK]"
            else:
                text = sep.join(seq) + f"{sep}[MASK]"
            masked = fluff + text
            if masked in seen_masked:
                continue
            seen_masked.add(masked)
            rows.append(
                {
                    "masked": masked,
                    "label": str(label),
                    "label_class": "MATCH",
                    "alternative": alt,
                    "alternative_class": "DISTRACTOR",
                    "unit_len": k,
                    "repeats": n_repeats,
                    "noise_n": len(noise_toks),
                    "difficulty": diff,
                    "split": split,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Controlled repetition confidence: yes ... yes [MASK] -> yes (alternative no)
# ---------------------------------------------------------------------------

def build_repetition(
    *,
    rows_per_split: Optional[Dict[str, int]] = None,
    seed: int = 0,
    repetition_counts: Sequence[int] = REPETITION_COUNTS,
    repetition_delimiters: Sequence[str] = DEFAULT_REPETITION_DELIMITERS,
) -> pd.DataFrame:
    """Build a matched one-pole confidence-control task.

    The continuation rule and candidate tokens are fixed. Context template,
    role, and topic vary across samples, while the same ``context_id`` and
    context fields and delimiter are shared across repetition levels. Thus
    repetition count is the only within-context confidence intervention.
    Split-specific id offsets prevent literal train/test duplicates.
    """
    counts = tuple(sorted({int(value) for value in repetition_counts}))
    if not counts or any(value < 1 for value in counts):
        raise ValueError("repetition_counts must contain positive integers")
    delimiters = tuple(dict.fromkeys(str(value) for value in repetition_delimiters))
    unknown = sorted(set(delimiters) - set(REPETITION_DELIMITER_PREFIXES))
    if not delimiters or unknown:
        raise ValueError(
            "repetition_delimiters must be nonempty and drawn from "
            f"{sorted(REPETITION_DELIMITER_PREFIXES)}; unknown={unknown}"
        )

    rows_per_split = _split_counts(rows_per_split)
    rng = np.random.default_rng(seed)
    split_offsets = {"train": 100_000, "validation": 200_000, "test": 300_000}
    rows: List[dict] = []
    for split, requested in rows_per_split.items():
        n = int(requested)
        order: List[Tuple[int, int]] = []
        base = 0
        while len(order) < n:
            shuffled = list(counts)
            rng.shuffle(shuffled)
            order.extend((base, repeat_count) for repeat_count in shuffled)
            base += 1
        offset = split_offsets.get(str(split), 400_000)
        for base_id, repeat_count in order[:n]:
            context_id = offset + int(base_id)
            context_seed = int(seed) * 1_000_003 + context_id * 97
            context_rng = np.random.default_rng(context_seed)
            family_id = int(context_rng.integers(0, len(_REPETITION_CONTEXT_TEMPLATES)))
            role = str(context_rng.choice(_REPETITION_ROLES))
            topic = str(context_rng.choice(_REPETITION_TOPICS))
            # Cycling by base id balances delimiter families while keeping the
            # delimiter fixed for a matched context at every repetition level.
            delimiter_id = int(base_id) % len(delimiters)
            delimiter_name = delimiters[delimiter_id]
            delimiter_prefix = REPETITION_DELIMITER_PREFIXES[delimiter_name]
            prefix = _REPETITION_CONTEXT_TEMPLATES[family_id].format(
                context_id=context_id,
                role=role,
                topic=topic,
            )
            separator = f"{delimiter_prefix} "
            repeated = separator.join(["yes"] * int(repeat_count))
            rows.append(
                {
                    # Trailing space before [MASK] so GPT-2 / GPT-NeoX score the
                    # word-initial Ġyes continuation. The decoder lstrips the
                    # label and uses prefix-gap whitespace; a glued `yes[MASK]`
                    # would score the unspaced `yes` token instead.
                    "masked": f"{prefix}{repeated}{delimiter_prefix} [MASK]",
                    "label": "yes",
                    "label_class": "YES",
                    "alternative": "no",
                    "alternative_class": "NO",
                    "repetition_count": int(repeat_count),
                    "repetition_delimiter": delimiter_name,
                    "repetition_delimiter_prefix": delimiter_prefix,
                    "repetition_surface_version": REPETITION_DELIMITER_VARIANT_VERSION,
                    "context_id": str(context_id),
                    "context_family": int(family_id),
                    "context_role": role,
                    "context_topic": topic,
                    "split": split,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Function composition = assignment / alias chains (NOT arithmetic)
# x=red, a=blue, y=x, y=[MASK] → red
# ---------------------------------------------------------------------------

def build_function_composition(
    *,
    rows_per_split: Optional[Dict[str, int]] = None,
    seed: int = 0,
    difficulty: str = DEFAULT_DIFFICULTY,
    chain_depth: Optional[Tuple[int, int]] = None,
    format_noise: Optional[bool] = None,
) -> pd.DataFrame:
    diff, preset = _resolve_preset(
        "function_composition",
        difficulty,
        overrides={"chain_depth": chain_depth, "format_noise": format_noise},
    )
    depth_range: Tuple[int, int] = tuple(preset["chain_depth"])  # type: ignore[assignment]
    fmt_noise = bool(preset["format_noise"])
    rows_per_split = _split_counts(rows_per_split)
    rng = np.random.default_rng(seed)
    rows: List[dict] = []
    for split, n in rows_per_split.items():
        values = _pool_for_split(_VALUES_BY_SPLIT, split)
        for _ in range(int(n)):
            # Chain depth: root value + ``depth`` alias hops (+ optional distractors)
            depth = _rand_inclusive(rng, depth_range)
            n_distractors = (
                int(rng.integers(0, 3)) if fmt_noise else int(rng.integers(0, 2))
            )
            n_vars = 1 + depth + n_distractors
            vars_ = list(
                rng.choice(_VAR_NAMES, size=min(n_vars, len(_VAR_NAMES)), replace=False)
            )
            root_var = vars_[0]
            alias_names = list(vars_[1 : 1 + depth])
            distractor_vars = list(vars_[1 + depth :])
            root_val = str(rng.choice(values))
            concrete: Dict[str, str] = {root_var: root_val}
            unused_vals: List[str] = []
            for v in distractor_vars:
                val = str(rng.choice([x for x in values if x != root_val]))
                concrete[v] = val
                unused_vals.append(val)

            ordered_stmts: List[Tuple[str, str]] = [(root_var, root_val)]
            extras = [(uv, val) for uv, val in concrete.items() if uv != root_var]
            if fmt_noise:
                rng.shuffle(extras)
            ordered_stmts.extend(extras)
            prev = root_var
            for alias in alias_names:
                ordered_stmts.append((alias, prev))
                prev = alias
            query_var = prev
            stmts = ordered_stmts
            if fmt_noise and len(extras) > 1 and bool(rng.integers(0, 2)):
                stmts = [stmts[0]] + list(reversed(extras)) + stmts[1 + len(extras) :]

            if fmt_noise:
                sep = str(rng.choice(_SEP_BIND))
                stmt_sep = str(rng.choice(_SEP_STMT))
                prefix = str(
                    rng.choice(
                        [
                            "",
                            "Let ",
                            "Assignments: ",
                            "Env: ",
                            "Code: ",
                            f"Step {int(rng.integers(1, 50))}: ",
                        ]
                    )
                )
                query = str(
                    rng.choice(
                        [
                            f"{stmt_sep}{query_var}{sep}[MASK]",
                            f"{stmt_sep}therefore {query_var}{sep}[MASK]",
                            f"{stmt_sep}so {query_var}{sep}[MASK]",
                            f"{stmt_sep}eval {query_var}{sep}[MASK]",
                            f"{stmt_sep}{query_var} resolves to [MASK]",
                        ]
                    )
                )
            else:
                sep, stmt_sep, prefix = " = ", "; ", ""
                query = f"{stmt_sep}{query_var}{sep}[MASK]"

            parts = [_format_binding(a, b, sep) for a, b in stmts]
            body = stmt_sep.join(parts)
            masked = f"{prefix}{body}{query}"
            distractor = (
                str(rng.choice(unused_vals))
                if unused_vals
                else str(rng.choice([x for x in values if x != root_val]))
            )
            rows.append(
                {
                    "masked": masked,
                    "label": root_val,
                    "label_class": "RESULT",
                    "alternative": distractor,
                    "alternative_class": "DISTRACTOR",
                    "chain_depth": depth,
                    "query_var": query_var,
                    "difficulty": diff,
                    "split": split,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Language ID (OPUS sides + MUSE alternatives)
# ---------------------------------------------------------------------------

def build_language(
    *,
    rows_per_split: Optional[Dict[str, int]] = None,
    seed: int = 0,
    use_fixture: bool = False,
    classes: Optional[Sequence[str]] = None,
    neutral_langs: Optional[Sequence[str]] = None,
    **kwargs,
) -> pd.DataFrame:
    """Study ``language`` task: 3-class language-of-the-text cloze.

    Full builds use real OPUS sentences. ``use_fixture=True`` is tests/smoke only.
    """
    labeled, _neu = build_language_with_neutrals(
        rows_per_split=rows_per_split,
        seed=seed,
        use_fixture=use_fixture,
        classes=classes,
        neutral_langs=neutral_langs,
        **kwargs,
    )
    return labeled


def build_language_with_neutrals(
    *,
    rows_per_split: Optional[Dict[str, int]] = None,
    seed: int = 0,
    use_fixture: bool = False,
    classes: Optional[Sequence[str]] = None,
    neutral_langs: Optional[Sequence[str]] = None,
    **kwargs,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    from study.data.language_parallel import build_language_id

    return build_language_id(
        classes=classes,
        neutral_langs=neutral_langs,
        rows_per_split=rows_per_split,
        seed=seed,
        use_fixture=use_fixture,
        **kwargs,
    )


GENERATORS = {
    "ioi": build_ioi,
    "key_value": build_key_value,
    "induction": build_induction,
    "function_composition": build_function_composition,
    "repetition": build_repetition,
    "language": build_language,
}


def ensure_synthetic(
    task_id: str,
    *,
    smoke: bool = False,
    seed: int = 0,
    force: bool = False,
    difficulty: Optional[str] = None,
    classes: Optional[Sequence[str]] = None,
    neutral_langs: Optional[Sequence[str]] = None,
) -> Path:
    diff = _normalize_difficulty(difficulty) if task_id in DIFFICULTY_TASKS else None
    stem = _cache_stem(task_id, difficulty=diff, smoke=smoke)
    path = SYN_DIR / f"{stem}.csv"
    min_n = 50 if smoke else int(sum(DEFAULT_ROWS.values()) * 0.9)
    expected_version = _data_version_for_task(task_id)
    neu_path = SYN_DIR / f"{stem}_neutral.csv" if task_id == "language" else None
    lang_min_n = 50 if smoke else int(sum(DEFAULT_ROWS.values()) * 3 * 0.9)
    if task_id == "language":
        language_classes = list(classes or ["en", "fr", "de"])
        if (
            not force
            and not smoke
            and _language_artifact_ok(
                path,
                min_n=lang_min_n,
                expected_version=expected_version,
                classes=language_classes,
            )
            and neu_path is not None
            and _language_artifact_ok(
                neu_path,
                min_n=min_n,
                expected_version=expected_version,
                classes=language_classes,
                neutral=True,
            )
        ):
            return path
        rows = SMOKE_ROWS if smoke else DEFAULT_ROWS
        labeled, neu = build_language_with_neutrals(
            rows_per_split=rows,
            seed=seed,
            use_fixture=bool(smoke),
            classes=classes,
            neutral_langs=neutral_langs,
        )
        extra: Dict[str, Any] = {
            "version": expected_version,
            "classes": list(classes or ["en", "fr", "de"]),
            "neutral_langs": list(neutral_langs or ["es", "it", "nl", "pt"]),
        }
        _write_csv(neu_path, neu, extra_meta=extra)
        return _write_csv(path, labeled, extra_meta=extra)

    if not force and not smoke and _rows_ok(
        path, min_n=min_n, expected_version=expected_version
    ):
        return path
    if task_id not in GENERATORS:
        raise KeyError(f"Unknown synthetic task {task_id!r}")
    gen = GENERATORS[task_id]
    rows = SMOKE_ROWS if smoke else DEFAULT_ROWS
    gen_kw: Dict[str, Any] = {"rows_per_split": rows, "seed": seed}
    if task_id in DIFFICULTY_TASKS:
        gen_kw["difficulty"] = diff
    df = gen(**gen_kw)
    extra = {"version": expected_version}
    if diff is not None:
        extra["difficulty"] = diff
    return _write_csv(path, df, extra_meta=extra)


def build_circuit_task(task_id: str, cfg: Dict[str, Any], *, smoke: bool = False) -> TaskBundle:
    task = cfg.get("task") or cfg
    data_cfg = task.get("data") or {}
    ensure_kw: Dict[str, Any] = {
        "smoke": smoke,
        "force": bool(data_cfg.get("force_regenerate", False)),
    }
    difficulty: Optional[str] = None
    if task_id in DIFFICULTY_TASKS:
        difficulty = _normalize_difficulty(data_cfg.get("difficulty"))
        ensure_kw["difficulty"] = difficulty
    if task_id == "language":
        classes_cfg = list(data_cfg.get("classes") or task.get("classes") or ["en", "fr", "de"])
        neu_langs = list(data_cfg.get("neutral_langs") or ["es", "it", "nl", "pt"])
        ensure_kw["classes"] = [str(c) for c in classes_cfg]
        ensure_kw["neutral_langs"] = [str(c) for c in neu_langs]
        task = dict(task)
        task["classes"] = [str(c) for c in classes_cfg]
    path = ensure_synthetic(task_id, **ensure_kw)
    df = pd.read_csv(path)
    classes = list(task.get("classes") or [])
    data_per_class: Dict[str, pd.DataFrame] = {}
    for cls in classes:
        sub = df[df["label_class"].astype(str) == str(cls)]
        if sub.empty:
            continue
        frame = sub[["masked", "split", "label"]].copy()
        frame = frame.rename(columns={"label": str(cls)})
        data_per_class[str(cls)] = frame
    if len(data_per_class) < 2 and "alternative" in df.columns and len(classes) >= 2:
        pos, neg = classes[0], classes[1]
        pos_rows = df[df["label_class"].astype(str) == str(pos)].copy()
        if pos_rows.empty and "alternative_class" in df.columns:
            # one-pole files: all factual is first class
            pos_rows = df.copy()
            pos_rows["label_class"] = pos
        data_per_class[pos] = pos_rows[["masked", "split", "label"]].rename(columns={"label": pos})
        neg_frame = pos_rows.copy()
        neg_frame["label"] = neg_frame["alternative"]
        data_per_class[neg] = neg_frame[["masked", "split", "label"]].rename(columns={"label": neg})

    if task_id == "language":
        from study.data.mib import _load_local_neutral

        neu_path = path.with_name(f"{path.stem}_neutral.csv")
        neutrals = _load_local_neutral(neu_path, cfg, smoke=smoke)
    else:
        neutrals = load_neutral_from_cfg(
            data_cfg.get("neutral_hf", "aieng-lab/biasneutral-ajibawa"),
            cfg,
            smoke=smoke,
        )
    # Exclude name tokens used in the task from neutral LMS/eval text.
    # Drop missing values before normalizing: pandas may retain a float ``NaN``
    # in a string-backed Series, which otherwise leaves mixed types for sorted().
    excluded: List[str] = []
    if {"label", "alternative"}.issubset(df.columns):
        names = {
            str(value).lower().strip()
            for column in ("label", "alternative")
            for value in df[column].dropna()
        }
        excluded = sorted(name for name in names if name)
    training_overrides = dict(task.get("training") or {})
    note = f"synthetic {task_id} v{_data_version_for_task(task_id)}"
    if difficulty is not None:
        note = f"{note} difficulty={difficulty}"
    eval_spec: Dict[str, Any] = {
        "causal_score": (task.get("causal") or {}).get("score", "continuation_match"),
        "source_both": True,
    }
    if task_id == "language":
        eval_spec["multiclass"] = True
        mba = (task.get("eval") or {}).get("min_base_accuracy")
        if mba is not None:
            eval_spec["min_base_accuracy"] = mba
    return TaskBundle(
        task_id=task_id,
        classes=classes,
        data_per_class=data_per_class,
        merged_df=df,
        neutrals=neutrals,
        eval_spec=eval_spec,
        training_overrides=training_overrides,
        excluded_words=excluded,
        notes=note,
    )


def generate_all(*, smoke: bool = False, force: bool = True) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for tid in ("ioi", "language"):
        out[tid] = ensure_synthetic(tid, smoke=smoke, force=force)
    for tid in DIFFICULTY_TASKS:
        for diff in DIFFICULTIES:
            out[f"{tid}_{diff}"] = ensure_synthetic(
                tid, smoke=smoke, force=force, difficulty=diff
            )
    return out
