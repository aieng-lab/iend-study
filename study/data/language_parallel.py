"""Language-ID cloze from real OPUS/Tatoeba sentences + MUSE translations.

Study task ``language``
-----------------------
Classes are **three languages of the text** (default ``en``, ``fr``, ``de``).
Neutrals are **other real languages** (default ``es``, ``it``, ``nl``, ``pt``)
in the **same cloze wrapper**.

1. **Sentences.** Load real bitext from HuggingFace ``Helsinki-NLP/opus-100`` /
   ``opus_books``, with an OPUS-Tatoeba moses zip fallback (cached under
   ``data/synthetic/cache/language/``). Each class/neutral language uses **one
   side** of a pair (English from ``en-fr``, French from ``en-fr``, German from
   ``en-de``, Spanish from ``en-es``, …). The prompt never concatenates two
   languages.
2. **Dictionary.** For labeled rows, look up a content-word translation in a
   *rival class* language from the public MUSE lexicon (Facebook
   ``{src}-{tgt}.txt``). That translation is the decoder ``alternative`` only;
   it is not inserted into the context.
3. **CLM cloze.** Mask a content word with **≥10 words** of left context
   (GRADIEND/ACTIEND default). Decoder-only models need that prefix::

       [{lang}] {prefix}[MASK]

   ``label`` / ``label_class`` = masked word / language of the sentence;
   ``alternative`` / ``alternative_class`` = MUSE rival-class translation.

Smoke/tests pass ``use_fixture=True`` (templated sentences, no download).
The study cache is OPUS, not those templates.
"""

from __future__ import annotations

import json
import re
import unicodedata
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from study.tasks import ROOT

from error_tracker import track_error

CACHE_DIR = ROOT / "data" / "synthetic" / "cache" / "language"
MUSE_URL = "https://dl.fbaipublicfiles.com/arrival/dictionaries/{pair}.txt"

LANG_NAMES = {
    "en": "English",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "it": "Italian",
    "pt": "Portuguese",
    "nl": "Dutch",
}

DEFAULT_CLASSES: Tuple[str, ...] = ("en", "fr", "de")
DEFAULT_NEUTRAL_LANGS: Tuple[str, ...] = ("es", "it", "nl", "pt")

# Function-word / interrogative / pronoun filters (do not mask these).
_STOP = {
    "en": {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of",
        "is", "are", "was", "were", "be", "been", "being", "it", "this", "that",
        "he", "she", "they", "we", "you", "i", "my", "his", "her", "their", "our",
        "your", "its", "him", "them", "us", "me", "with", "from", "by", "as",
        "not", "no", "yes", "do", "does", "did", "have", "has", "had", "will",
        "would", "can", "could", "should", "may", "might", "must", "shall",
        "who", "what", "when", "where", "why", "how", "which", "whom", "whose",
        "one", "ones", "here", "there", "then", "than", "so", "if", "also",
        "just", "only", "very", "more", "most", "other", "into", "about", "over",
        "after", "before", "because", "while", "though", "still", "even", "now",
        "well", "such", "same", "own", "both", "each", "few", "many", "much",
        "all", "some", "any", "these", "those", "am", "up", "out", "off", "too",
    },
    "fr": {
        "le", "la", "les", "un", "une", "des", "et", "ou", "mais", "de", "du",
        "à", "au", "aux", "en", "dans", "sur", "pour", "par", "avec", "est", "sont",
        "été", "être", "il", "elle", "ils", "elles", "je", "tu", "nous", "vous",
        "ce", "cet", "cette", "ces", "ne", "pas", "plus", "que", "qui", "dont",
        "quoi", "où", "comment", "pourquoi", "quand", "on", "leur", "lui", "y",
        "moins", "très", "tout", "tous", "toute", "toutes", "autre", "encore",
        "déjà", "bien", "aussi", "donc", "car", "ni", "si", "comme", "mon", "ma",
        "mes", "ton", "ta", "tes", "son", "sa", "ses", "nos", "vos", "leurs",
        "oui", "non", "ici", "là", "alors", "ainsi",
    },
    "de": {
        "der", "die", "das", "ein", "eine", "und", "oder", "aber", "in", "an", "auf",
        "zu", "von", "mit", "für", "ist", "sind", "war", "waren", "sein", "er", "sie",
        "es", "wir", "ihr", "ich", "du", "mein", "nicht", "auch", "den", "dem", "des",
        "im", "am", "vom", "wer", "was", "wo", "wie", "warum", "wann", "welche",
        "man", "uns", "euch", "ihm", "ihn", "ihnen", "hier", "da", "dort", "dann",
        "denn", "wenn", "als", "nur", "sehr", "einem", "einen", "einer", "eines",
        "noch", "schon", "mehr", "alle", "alles", "etwas", "nichts", "ja", "nein",
        "mein", "meine", "dein", "deine", "sein", "seine", "unser", "euer", "ihre",
        "sich", "so", "auch", "nur",
    },
    "es": {
        "el", "la", "los", "las", "un", "una", "unos", "unas", "y", "o", "pero",
        "de", "del", "a", "al", "en", "con", "por", "para", "es", "son", "está",
        "están", "ser", "estar", "que", "quien", "qué", "cómo", "cuándo", "dónde",
        "porqué", "no", "sí", "lo", "le", "les", "se", "me", "te", "nos", "yo",
        "tú", "él", "ella", "ellos", "ellas", "mi", "su", "este", "esta", "eso",
        "más", "muy", "ya", "también", "como", "si", "porque", "cuando", "donde",
        "hay", "fue", "era", "tiene", "tiene", "todo", "todos", "una",
    },
    "it": {
        "il", "lo", "la", "i", "gli", "le", "un", "uno", "una", "e", "o", "ma",
        "di", "del", "della", "a", "al", "in", "con", "per", "da", "è", "sono",
        "essere", "che", "chi", "cosa", "come", "quando", "dove", "perché", "non",
        "sì", "si", "mi", "ti", "ci", "vi", "io", "tu", "lui", "lei", "noi", "voi",
        "loro", "questo", "questa", "quello", "più", "molto", "già", "anche", "se",
        "perché", "c'è", "ha", "hanno", "tutto", "tutti",
    },
    "nl": {
        "de", "het", "een", "en", "of", "maar", "in", "op", "aan", "te", "van",
        "met", "voor", "is", "zijn", "was", "waren", "hij", "zij", "ze", "we",
        "jij", "ik", "mijn", "niet", "ook", "wie", "wat", "waar", "waarom", "hoe",
        "wanneer", "welke", "hier", "daar", "dan", "als", "nog", "al", "meer",
        "alles", "iets", "niets", "ja", "nee", "dit", "dat", "deze", "die", "er",
        "om", "bij", "uit", "over", "naar",
    },
    "pt": {
        "o", "a", "os", "as", "um", "uma", "e", "ou", "mas", "de", "do", "da",
        "dos", "das", "em", "no", "na", "com", "por", "para", "é", "são", "está",
        "estão", "ser", "estar", "que", "quem", "como", "quando", "onde", "porque",
        "não", "sim", "se", "me", "te", "nos", "eu", "tu", "ele", "ela", "eles",
        "elas", "meu", "sua", "este", "esta", "isso", "mais", "muito", "já",
        "também", "se", "foi", "era", "tem", "tudo", "todos",
    },
}

_TOKEN_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


def _split_counts(rows_per_split: Optional[Dict[str, int]]) -> Dict[str, int]:
    from study.data.synthetic import DEFAULT_ROWS

    if not rows_per_split:
        return dict(DEFAULT_ROWS)
    return {str(k): int(v) for k, v in rows_per_split.items() if int(v) > 0}


def _cache_path(*parts: str) -> Path:
    path = CACHE_DIR.joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _download(url: str, dest: Path, *, timeout: int = 120) -> Path:
    if dest.is_file() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    print(f"  language: downloading {url} -> {dest}", flush=True)
    with urllib.request.urlopen(url, timeout=timeout) as resp, open(tmp, "wb") as out:
        out.write(resp.read())
    tmp.replace(dest)
    return dest


def load_muse_lexicon(
    factual_lang: str,
    cf_lang: str,
    *,
    max_entries: int = 80_000,
) -> Dict[str, str]:
    """Return ``{factual_word_lower: cf_word_surface}`` from MUSE."""
    fac, cf = str(factual_lang).lower(), str(cf_lang).lower()
    pair = f"{fac}-{cf}"
    rev = f"{cf}-{fac}"
    dest = _cache_path("muse", f"{pair}.txt")
    lexicon: Dict[str, str] = {}
    try:
        _download(MUSE_URL.format(pair=pair), dest)
        invert = False
    except Exception:
        dest_rev = _cache_path("muse", f"{rev}.txt")
        _download(MUSE_URL.format(pair=rev), dest_rev)
        dest = dest_rev
        invert = True
    with open(dest, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            w0, w1 = parts[0], parts[1]
            if invert:
                w_fac, w_cf = w1, w0
            else:
                w_fac, w_cf = w0, w1
            key = w_fac.lower()
            if key in lexicon:
                continue
            if len(key) < 2 or len(w_cf) < 2:
                continue
            lexicon[key] = w_cf
            if len(lexicon) >= int(max_entries):
                break
    if len(lexicon) < 100:
        raise RuntimeError(
            f"MUSE lexicon {fac}-{cf} too small ({len(lexicon)} entries) after download"
        )
    print(f"  language: MUSE {fac}-{cf} entries={len(lexicon)}", flush=True)
    return lexicon




def load_parallel_sentences(
    factual_lang: str,
    cf_lang: str,
    *,
    max_pairs: int = 200_000,
) -> List[Tuple[str, str]]:
    """Load ``(factual_sentence, cf_sentence)`` pairs (OPUS-100, OPUS Books, Tatoeba moses)."""
    fac, cf = str(factual_lang).lower(), str(cf_lang).lower()
    cache = _cache_path("parallel", f"{fac}-{cf}.jsonl")
    if cache.is_file():
        out: List[Tuple[str, str]] = []
        with open(cache, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                obj = json.loads(line)
                out.append((str(obj["fac"]), str(obj["cf"])))
                if len(out) >= int(max_pairs):
                    break
        if len(out) >= 500:
            print(f"  language: cached parallel {fac}-{cf} n={len(out)}", flush=True)
            return out

    pairs: List[Tuple[str, str]] = []
    seen = set()

    def _add(a: str, b: str) -> None:
        a, b = a.strip(), b.strip()
        if len(a.split()) < 3 or len(b.split()) < 3:
            return
        if len(a) > 400 or len(b) > 400:
            return
        key = (a.lower(), b.lower())
        if key in seen:
            return
        seen.add(key)
        pairs.append((a, b))

    # 1) OPUS-100 (script-free HF dataset)
    try:
        from datasets import load_dataset

        config = f"{fac}-{cf}"
        print(f"  language: loading opus-100 {config} ...", flush=True)
        try:
            ds = load_dataset("Helsinki-NLP/opus-100", config, split="train")
            swapped = False
        except Exception:
            ds = load_dataset("Helsinki-NLP/opus-100", f"{cf}-{fac}", split="train")
            swapped = True
        for row in ds:
            if len(pairs) >= int(max_pairs):
                break
            tr = row.get("translation") or row
            if not isinstance(tr, dict):
                continue
            if swapped:
                if cf in tr and fac in tr:
                    _add(str(tr[fac]), str(tr[cf]))
            else:
                if fac in tr and cf in tr:
                    _add(str(tr[fac]), str(tr[cf]))
    except Exception as exc:
        track_error(exc, context="language: opus-100 load")

    # 2) OPUS Books
    if len(pairs) < int(max_pairs):
        try:
            from datasets import load_dataset

            config = f"{fac}-{cf}"
            print(f"  language: loading opus_books {config} ...", flush=True)
            try:
                ds = load_dataset("opus_books", config, split="train")
                swapped = False
            except Exception:
                ds = load_dataset("opus_books", f"{cf}-{fac}", split="train")
                swapped = True
            for row in ds:
                if len(pairs) >= int(max_pairs):
                    break
                tr = row.get("translation") or row
                if not isinstance(tr, dict):
                    continue
                if swapped:
                    if cf in tr and fac in tr:
                        _add(str(tr[fac]), str(tr[cf]))
                else:
                    if fac in tr and cf in tr:
                        _add(str(tr[fac]), str(tr[cf]))
        except Exception as exc:
            track_error(exc, context="language: opus_books load")

    # 3) Tatoeba moses dump (no dataset script)
    if len(pairs) < max(1000, int(max_pairs) // 4):
        try:
            _load_tatoeba_moses(fac, cf, add_fn=_add, max_pairs=max_pairs, pairs=pairs)
        except Exception as exc:
            track_error(exc, context="language: tatoeba moses load")

    if len(pairs) < 200:
        raise RuntimeError(
            f"Could not load enough parallel sentences for {fac}-{cf} "
            f"(got {len(pairs)}). Install `datasets` and allow network, or pass "
            f"pairs=... explicitly."
        )

    with open(cache, "w", encoding="utf-8") as handle:
        for a, b in pairs[: int(max_pairs)]:
            handle.write(json.dumps({"fac": a, "cf": b}, ensure_ascii=False) + "\n")
    print(f"  language: parallel {fac}-{cf} n={len(pairs)} (cached)", flush=True)
    return pairs[: int(max_pairs)]


def _load_tatoeba_moses(
    fac: str,
    cf: str,
    *,
    add_fn,
    max_pairs: int,
    pairs: List[Tuple[str, str]],
) -> None:
    """Fetch OPUS-Tatoeba moses bitext if HF tatoeba scripts are unavailable."""
    import io
    import zipfile

    # Prefer smaller recent Tatoeba moses packages when present on OPUS.
    urls = [
        f"https://object.pouta.csc.fi/OPUS-Tatoeba/v2023-04-12/moses/{fac}-{cf}.txt.zip",
        f"https://object.pouta.csc.fi/OPUS-Tatoeba/v2023-04-12/moses/{cf}-{fac}.txt.zip",
    ]
    dest_zip = _cache_path("tatoeba", f"{fac}-{cf}.txt.zip")
    swapped = False
    for i, url in enumerate(urls):
        try:
            _download(url, dest_zip if i == 0 else _cache_path("tatoeba", f"{cf}-{fac}.txt.zip"))
            if i == 1:
                dest_zip = _cache_path("tatoeba", f"{cf}-{fac}.txt.zip")
                swapped = True
            break
        except Exception:
            continue
    else:
        raise RuntimeError("no Tatoeba moses zip available")

    with zipfile.ZipFile(dest_zip, "r") as zf:
        names = zf.namelist()
        # Files look like Tatoeba.en-fr.en / Tatoeba.en-fr.fr
        fac_name = next((n for n in names if n.endswith(f".{fac}")), None)
        cf_name = next((n for n in names if n.endswith(f".{cf}")), None)
        if fac_name is None or cf_name is None:
            # zip may use source-target order only
            raise RuntimeError(f"unexpected tatoeba zip members: {names[:8]}")
        with zf.open(fac_name) as fa, zf.open(cf_name) as fb:
            for la, lb in zip(io.TextIOWrapper(fa, encoding="utf-8", errors="replace"),
                              io.TextIOWrapper(fb, encoding="utf-8", errors="replace")):
                if len(pairs) >= int(max_pairs):
                    break
                a, b = la.strip(), lb.strip()
                if swapped:
                    add_fn(b, a)
                else:
                    add_fn(a, b)
    print(f"  language: tatoeba moses added; n={len(pairs)}", flush=True)


def _tokens_with_spans(text: str) -> List[Tuple[str, int, int]]:
    return [(m.group(0), m.start(), m.end()) for m in _TOKEN_RE.finditer(text)]


def _fold_token(text: str) -> str:
    nk = unicodedata.normalize("NFKD", str(text).lower())
    return "".join(ch for ch in nk if not unicodedata.combining(ch))


def _format_cloze(*, lang: str, prefix: str) -> str:
    prefix = " ".join(str(prefix).split())
    if prefix and not prefix.endswith((" ", "\n")):
        prefix = prefix + " "
    return f"[{str(lang).lower()}] {prefix}[MASK]"


# GRADIEND/ACTIEND default: decoder-only models need ≥10 words left of the target.
MIN_LEFT_CONTEXT_WORDS = 10


def _find_maskable_mono(
    sent: str,
    lang: str,
    rng: np.random.Generator,
    *,
    lexicon: Optional[Dict[str, str]] = None,
    skip_cognates: bool = True,
    min_prefix_words: int = MIN_LEFT_CONTEXT_WORDS,
) -> Optional[Tuple[str, str, Optional[str]]]:
    """Return ``(prefix, word, alternative_or_none)`` for a monolingual cloze."""
    stops = _STOP.get(str(lang).lower(), set())
    toks = _tokens_with_spans(sent)
    candidates: List[Tuple[str, Optional[str], int]] = []
    for tok, start, _end in toks:
        key = tok.lower()
        if key in stops or len(key) < 3:
            continue
        prefix = sent[:start]
        if len(prefix.split()) < int(min_prefix_words):
            continue
        w_alt: Optional[str] = None
        if lexicon is not None:
            w_alt = lexicon.get(key)
            if not w_alt:
                continue
            if skip_cognates and _fold_token(tok) == _fold_token(w_alt):
                continue
        candidates.append((tok, w_alt, start))
    if not candidates:
        return None
    tok, w_alt, start = candidates[int(rng.integers(0, len(candidates)))]
    return sent[:start], tok, w_alt


def _opus_partner(lang: str) -> Tuple[str, str, str]:
    """``(fac, cf, side)`` so ``side`` of OPUS ``fac-cf`` is ``lang``."""
    lang = str(lang).lower()
    if lang == "en":
        return "en", "fr", "fac"
    return "en", lang, "cf"


def load_monolingual_sentences(
    lang: str,
    *,
    max_n: int = 200_000,
) -> List[str]:
    """Real OPUS/Tatoeba sentences in ``lang`` (one side of a bitext pair)."""
    fac, cf, side = _opus_partner(lang)
    pairs = load_parallel_sentences(fac, cf, max_pairs=max_n)
    if side == "fac":
        sents = [a for a, _ in pairs]
    else:
        sents = [b for _, b in pairs]
    min_words = MIN_LEFT_CONTEXT_WORDS + 2
    return [s for s in sents if len(s.split()) >= min_words]


def _min_unique(total: int) -> int:
    if total < 200:
        return max(1, min(total, 50))
    return max(50, total // 5)


def _fill_splits(
    collected: Sequence[dict],
    rows_per_split: Dict[str, int],
    rng: np.random.Generator,
) -> List[dict]:
    items = list(collected)
    needed = int(sum(rows_per_split.values()))
    if len(items) < needed:
        raise RuntimeError(
            f"Need {needed} unique rows for disjoint splits, but only collected {len(items)}. "
            "Increase the source scan instead of cycling examples across splits."
        )
    rng.shuffle(items)
    rows: List[dict] = []
    cursor = 0
    for split, n in rows_per_split.items():
        for _ in range(int(n)):
            item = dict(items[cursor])
            item["split"] = split
            rows.append(item)
            cursor += 1
    return rows














# Offline templates for tests/smoke only. Full builds use OPUS (see docstring).
_FIXTURE_CONCEPTS: List[Dict[str, str]] = [
    {"en": "book", "fr": "livre", "de": "Buch", "es": "libro", "it": "libro", "nl": "boek", "pt": "livro"},
    {"en": "house", "fr": "maison", "de": "Haus", "es": "casa", "it": "casa", "nl": "huis", "pt": "casa"},
    {"en": "water", "fr": "eau", "de": "Wasser", "es": "agua", "it": "acqua", "nl": "water", "pt": "água"},
    {"en": "friend", "fr": "ami", "de": "Freund", "es": "amigo", "it": "amico", "nl": "vriend", "pt": "amigo"},
    {"en": "school", "fr": "école", "de": "Schule", "es": "escuela", "it": "scuola", "nl": "school", "pt": "escola"},
    {"en": "city", "fr": "ville", "de": "Stadt", "es": "ciudad", "it": "città", "nl": "stad", "pt": "cidade"},
    {"en": "night", "fr": "nuit", "de": "Nacht", "es": "noche", "it": "notte", "nl": "nacht", "pt": "noite"},
    {"en": "child", "fr": "enfant", "de": "Kind", "es": "niño", "it": "bambino", "nl": "kind", "pt": "criança"},
    {"en": "river", "fr": "rivière", "de": "Fluss", "es": "río", "it": "fiume", "nl": "rivier", "pt": "rio"},
    {"en": "bread", "fr": "pain", "de": "Brot", "es": "pan", "it": "pane", "nl": "brood", "pt": "pão"},
    {"en": "window", "fr": "fenêtre", "de": "Fenster", "es": "ventana", "it": "finestra", "nl": "raam", "pt": "janela"},
    {"en": "garden", "fr": "jardin", "de": "Garten", "es": "jardín", "it": "giardino", "nl": "tuin", "pt": "jardim"},
    {"en": "market", "fr": "marché", "de": "Markt", "es": "mercado", "it": "mercato", "nl": "markt", "pt": "mercado"},
    {"en": "teacher", "fr": "professeur", "de": "Lehrer", "es": "profesor", "it": "insegnante", "nl": "leraar", "pt": "professor"},
    {"en": "mountain", "fr": "montagne", "de": "Berg", "es": "montaña", "it": "montagna", "nl": "berg", "pt": "montanha"},
    {"en": "library", "fr": "bibliothèque", "de": "Bibliothek", "es": "biblioteca", "it": "biblioteca", "nl": "bibliotheek", "pt": "biblioteca"},
    {"en": "hospital", "fr": "hôpital", "de": "Krankenhaus", "es": "hospital", "it": "ospedale", "nl": "ziekenhuis", "pt": "hospital"},
    {"en": "station", "fr": "gare", "de": "Bahnhof", "es": "estación", "it": "stazione", "nl": "station", "pt": "estação"},
    {"en": "university", "fr": "université", "de": "Universität", "es": "universidad", "it": "università", "nl": "universiteit", "pt": "universidade"},
]

_FIXTURE_TEMPLATES: Dict[str, List[str]] = {
    "en": [
        "Yesterday after the long meeting downtown they finally bought a brand new {w} near the river.",
        "This morning she slowly walked past the old {w} after lunch near the crowded square.",
        "For several hours we talked about the local {w} with our neighbors after dinner.",
        "Earlier today I clearly saw the large {w} near the old bridge by the market.",
        "Every morning before work he quietly reads about a famous {w} in the paper.",
    ],
    "fr": [
        "Hier après la longue réunion en ville ils ont enfin acheté un tout nouveau {w} près du fleuve.",
        "Ce matin elle est lentement passée devant le vieux {w} après le déjeuner près de la place.",
        "Pendant plusieurs heures nous avons parlé du {w} local avec nos voisins après le dîner.",
        "Plus tôt aujourd'hui j'ai clairement vu le grand {w} près du vieux pont du marché.",
        "Chaque matin avant le travail il lit tranquillement un célèbre {w} dans le journal.",
    ],
    "de": [
        "Gestern nach der langen Sitzung in der Stadt kauften sie endlich ein ganz neues {w} am Fluss.",
        "Heute Morgen ging sie nach dem Mittagessen langsam am alten {w} nahe dem Platz vorbei.",
        "Mehrere Stunden lang sprachen wir nach dem Abendessen mit den Nachbarn über das lokale {w}.",
        "Heute früh sah ich deutlich das große {w} in der Nähe der alten Brücke am Markt.",
        "Jeden Morgen vor der Arbeit liest er ruhig über ein berühmtes {w} in der Zeitung.",
    ],
    "es": [
        "Ayer después de la larga reunión en el centro compraron un {w} nuevo cerca del río.",
        "Esta mañana ella pasó despacio junto al viejo {w} después del almuerzo cerca de la plaza.",
        "Durante varias horas hablamos del {w} local con los vecinos después de la cena.",
        "Hoy temprano vi con claridad el gran {w} cerca del puente viejo del mercado.",
        "Cada mañana antes del trabajo lee en el periódico sobre un {w} famoso de la ciudad.",
    ],
    "it": [
        "Ieri dopo la lunga riunione in centro hanno comprato un nuovo {w} vicino al fiume.",
        "Stamattina lei è passata lentamente davanti al vecchio {w} dopo pranzo vicino alla piazza.",
        "Per diverse ore abbiamo parlato del {w} locale con i vicini dopo cena in cucina.",
        "Oggi presto ho visto chiaramente il grande {w} vicino al vecchio ponte del mercato.",
        "Ogni mattina prima del lavoro legge sul giornale di un {w} famoso della città.",
    ],
    "nl": [
        "Gisteren na de lange vergadering in de stad kochten ze eindelijk een nieuwe {w} bij de rivier.",
        "Vanmorgen liep ze na de lunch langzaam langs de oude {w} bij het drukke plein.",
        "Urenlang praatten we na het avondeten met de buren over de lokale {w} in het dorp.",
        "Vanochtend zag ik duidelijk de grote {w} bij de oude brug aan de markt.",
        "Elke ochtend voor het werk leest hij rustig over een beroemde {w} in de krant.",
    ],
    "pt": [
        "Ontem depois da longa reunião no centro eles compraram um {w} novo perto do rio.",
        "Esta manhã ela passou devagar pelo velho {w} depois do almoço perto da praça.",
        "Durante várias horas falamos do {w} local com os vizinhos depois do jantar.",
        "Hoje cedo vi com clareza o grande {w} perto da ponte velha do mercado.",
        "Todas as manhãs antes do trabalho ele lê no jornal sobre um {w} famoso da cidade.",
    ],
}


def fixture_language_id(
    classes: Sequence[str],
    neutral_langs: Sequence[str],
    n: int = 400,
    *,
    seed: int = 0,
) -> Tuple[Dict[str, List[str]], Dict[Tuple[str, str], Dict[str, str]]]:
    """Offline sentences + MUSE-like lexica for tests / smoke (not the study cache)."""
    rng = np.random.default_rng(seed)
    langs = sorted({str(x).lower() for x in list(classes) + list(neutral_langs)})
    pad = {
        "en": "At the beginning of this recorded account today",
        "fr": "Au tout début de ce récit enregistré aujourd'hui",
        "de": "Am Anfang dieser aufgezeichneten kurzen Geschichte heute früh",
        "es": "Al comienzo de este relato registrado hoy mismo",
        "it": "All'inizio di questo racconto registrato oggi stesso",
        "nl": "Aan het begin van dit opgenomen verhaal van vandaag",
        "pt": "No começo deste relato registado hoje cedo mesmo",
    }
    sentences: Dict[str, List[str]] = {lang: [] for lang in langs}
    for lang in langs:
        tmpls = _FIXTURE_TEMPLATES.get(lang)
        if not tmpls:
            raise KeyError(f"No offline fixture templates for {lang}")
        words = [c[lang] for c in _FIXTURE_CONCEPTS if lang in c]
        lead = pad.get(lang, pad["en"])
        for i in range(int(n)):
            w = words[i % len(words)]
            tmpl = tmpls[int(rng.integers(0, len(tmpls)))]
            # Smoke fixtures must obey the same no-pseudoreplication contract
            # as production data. The record id supplies distinct context
            # without changing the language or prediction target.
            sentences[lang].append(f"{lead} record {i}: {tmpl.format(w=w)}")
    lexica: Dict[Tuple[str, str], Dict[str, str]] = {}
    for src in classes:
        src = str(src).lower()
        for tgt in classes:
            tgt = str(tgt).lower()
            if src == tgt:
                continue
            lexica[(src, tgt)] = {
                c[src].lower(): c[tgt]
                for c in _FIXTURE_CONCEPTS
                if src in c and tgt in c
            }
    return sentences, lexica


def build_language_id(
    *,
    classes: Optional[Sequence[str]] = None,
    neutral_langs: Optional[Sequence[str]] = None,
    rows_per_split: Optional[Dict[str, int]] = None,
    seed: int = 0,
    use_fixture: bool = False,
    max_parallel_scan: int = 200_000,
    max_muse_entries: int = 80_000,
    sentences_by_lang: Optional[Dict[str, Sequence[str]]] = None,
    lexica: Optional[Dict[Tuple[str, str], Dict[str, str]]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Three-way language-ID cloze + other-language neutrals.

    Returns ``(labeled_df, neutral_df)``. Labeled rows are real OPUS sides
    unless ``use_fixture=True`` (tests/smoke).
    """
    classes = [str(c).lower() for c in (classes or DEFAULT_CLASSES)]
    neutral_langs = [str(c).lower() for c in (neutral_langs or DEFAULT_NEUTRAL_LANGS)]
    overlap = set(classes) & set(neutral_langs)
    if overlap:
        raise ValueError(f"neutral_langs overlap classes: {sorted(overlap)}")
    if len(classes) < 3:
        raise ValueError(f"language task needs ≥3 classes (got {classes!r})")
    for lang in classes + neutral_langs:
        if lang not in LANG_NAMES:
            raise ValueError(f"Unsupported language {lang!r}")

    rows_per_split = _split_counts(rows_per_split)
    per_class = int(sum(rows_per_split.values()))
    rng = np.random.default_rng(seed)
    source = "fixture" if use_fixture else "opus_muse"

    if sentences_by_lang is None or lexica is None:
        if use_fixture:
            n_fix = max(400, per_class * 2)
            sentences_by_lang, lexica = fixture_language_id(
                classes, neutral_langs, n=n_fix, seed=seed
            )
        else:
            sentences_by_lang = {}
            lexica = {}
            for lang in classes + neutral_langs:
                sentences_by_lang[lang] = load_monolingual_sentences(
                    lang, max_n=max_parallel_scan
                )
            for src in classes:
                for tgt in classes:
                    if src == tgt:
                        continue
                    lexica[(src, tgt)] = load_muse_lexicon(
                        src, tgt, max_entries=max_muse_entries
                    )

    labeled_parts: List[pd.DataFrame] = []
    for cls in classes:
        rivals = [c for c in classes if c != cls]
        sents = list(sentences_by_lang.get(cls) or [])
        if not sents:
            raise RuntimeError(f"No sentences for class language {cls}")
        collected: List[dict] = []
        seen_keys = set()
        order = rng.permutation(len(sents))
        for idx in order:
            if len(collected) >= per_class * 3:
                break
            rival = rivals[int(len(collected)) % len(rivals)]
            lex = {str(k).lower(): str(v) for k, v in (lexica.get((cls, rival)) or {}).items()}
            hit = _find_maskable_mono(sents[int(idx)], cls, rng, lexicon=lex)
            if hit is None:
                continue
            prefix, w_fac, w_cf = hit
            if not w_cf:
                continue
            masked = _format_cloze(lang=cls, prefix=prefix)
            key = masked.casefold()
            if key in seen_keys:
                continue
            seen_keys.add(key)
            collected.append(
                {
                    "masked": masked,
                    "label": w_fac,
                    "label_class": cls,
                    "alternative": w_cf,
                    "alternative_class": rival,
                    "source": source,
                    "pair": f"{cls}-{rival}",
                }
            )
        min_ok = _min_unique(per_class)
        if len(collected) < min_ok:
            raise RuntimeError(
                f"Only {len(collected)} cloze rows for class {cls} "
                f"(need ≥{min_ok} for ~{per_class} requested)."
            )
        part = pd.DataFrame(_fill_splits(collected, rows_per_split, rng))
        print(
            f"  language: class {cls} n={len(part)} unique~{len(seen_keys)}",
            flush=True,
        )
        labeled_parts.append(part)

    labeled = pd.concat(labeled_parts, ignore_index=True)

    neu_need = per_class
    neu_collected: List[dict] = []
    seen_neu = set()
    for lang in neutral_langs:
        sents = list(sentences_by_lang.get(lang) or [])
        if not sents:
            raise RuntimeError(f"No sentences for neutral language {lang}")
        order = rng.permutation(len(sents))
        quota = max(neu_need // len(neutral_langs) + 8, 8)
        got = 0
        for idx in order:
            if got >= quota * 3:
                break
            raw_sentence = sents[int(idx)]
            hit = _find_maskable_mono(raw_sentence, lang, rng, lexicon=None)
            if hit is None:
                continue
            prefix, w_fac, _alt = hit
            masked = _format_cloze(lang=lang, prefix=prefix)
            key = str(raw_sentence).strip().casefold()
            if key in seen_neu:
                continue
            seen_neu.add(key)
            neu_collected.append(
                {
                    # Deliberately no "masked" column, and "text" keeps the
                    # "[lang] " tag (other code/tests key off it to identify
                    # source language) but drops the "[MASK]" placeholder and
                    # the prefix-only truncation _format_cloze applies for
                    # cloze training pairs. This is the external neutral
                    # corpus, not a training/eval cloze pair -- gradiend's
                    # _encode_neutral_dataset_rows expects genuinely
                    # unmasked text so it can derive its own mask
                    # position/target (create_masked_pair_from_text raises if
                    # handed text that already contains "[MASK]" -- confirmed
                    # live 2026-08-22 on gpt2-small/language's fair encoder
                    # eval). The previous version reused the templated cloze
                    # string (built for the labeled rows' dict shape above,
                    # which does need a real cloze pair) for "text" too.
                    "text": f"[{str(lang).lower()}] {raw_sentence}",
                    "label": w_fac,
                    "label_class": "neutral",
                    "source_lang": lang,
                    "source": source,
                }
            )
            got += 1
    if len(neu_collected) < _min_unique(neu_need):
        raise RuntimeError(
            f"Only {len(neu_collected)} other-language neutrals "
            f"(need ≥{_min_unique(neu_need)})."
        )
    neutrals = pd.DataFrame(_fill_splits(neu_collected, rows_per_split, rng))
    print(
        f"  language: neutrals n={len(neutrals)} langs={neutral_langs} "
        f"unique~{len(seen_neu)}",
        flush=True,
    )
    return labeled, neutrals
