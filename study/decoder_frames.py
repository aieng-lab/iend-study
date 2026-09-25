"""Task-level inputs for package decoder evaluation."""

from __future__ import annotations

from typing import Any, Sequence, Tuple

import pandas as pd


def decoder_eval_frames_for_study(
    labeled_df: pd.DataFrame,
    neutral_df: pd.DataFrame | None,
    *,
    target_classes: Sequence[str],
    max_size: int,
    split: Any,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build bounded task-level decoder frames, including all factual poles.

    This deliberately does not reuse a one-pole trainer's internally filtered
    training frame. Decoder strengthen selection needs the rival factual panel
    even though the one-pole feature has only one claim class.
    """
    split_key = str(split or "test").lower()
    frame = labeled_df.copy()
    if "split" in frame.columns:
        if split_key in {"validation", "val"}:
            subset = frame[frame["split"].astype(str).str.lower().isin({"validation", "val"})]
        else:
            subset = frame[frame["split"].astype(str).str.lower() == split_key]
        if not subset.empty:
            frame = subset.copy()
    if "label_class" in frame.columns:
        pre_filter_counts = frame["label_class"].astype(str).value_counts().to_dict()
        frame = frame[
            frame["label_class"].astype(str).isin({str(c) for c in target_classes})
        ].copy()
        post_filter_counts = frame["label_class"].astype(str).value_counts().to_dict()
        missing_classes = {str(c) for c in target_classes} - set(post_filter_counts)
        if missing_classes:
            print(
                f"  WARNING decoder_eval_frames_for_study: target_classes "
                f"{sorted(missing_classes)} have zero rows after split={split_key!r} "
                f"filtering (available label_class values pre-filter: {sorted(pre_filter_counts)}) "
                "-- any decoder-eval panel needing these classes will come up empty downstream.",
                flush=True,
            )
        # Keep the package's max-size contract: it applies per factual group.
        if not frame.empty:
            frame = pd.concat(
                [
                    group.sample(min(len(group), max_size), random_state=0)
                    for _, group in frame.groupby("label_class", sort=True)
                ],
                ignore_index=True,
            )

    # Package row-wise decoder scoring (compute_probability_shift_score_row_wise)
    # hard-requires columns named 'factual'/'factual_id'/'alternative_id' — this
    # study's frames carry the same data under 'label'/'label_class'/
    # 'alternative_class' (see causal_eval.py::meta_rows_to_frame, which aliases
    # the same way for the SAE/CAA causal path).
    if "label" in frame.columns:
        frame["factual"] = frame["label"]
    if "label_class" in frame.columns:
        frame["factual_id"] = frame["label_class"]
    if "alternative_class" in frame.columns:
        frame["alternative_id"] = frame["alternative_class"]

    neutral = neutral_df.copy() if neutral_df is not None else pd.DataFrame()
    if not neutral.empty and "split" in neutral.columns:
        if split_key in {"validation", "val"}:
            subset = neutral[
                neutral["split"].astype(str).str.lower().isin({"validation", "val"})
            ]
        else:
            subset = neutral[neutral["split"].astype(str).str.lower() == split_key]
        if not subset.empty:
            neutral = subset.copy()
    if len(neutral) > max_size:
        neutral = neutral.sample(max_size, random_state=0)
    return frame.reset_index(drop=True), neutral.reset_index(drop=True)
