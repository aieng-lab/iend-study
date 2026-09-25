"""PoC ablation inventory — expected method-id patterns from the gender reference.

Used to document and sanity-check that the deep pipeline covers the same set
as ``study.sae_engine`` (encode + causal).
"""

from __future__ import annotations



# SAE fixed-k bag sizes (encode + causal) — same as gender PoC.
SAE_FIXED_KS = (1, 2, 4, 8, 16, 32, 64, 128)

# ACTIEND causal ungated ablations (default gate is separate).

CAA_ACT_POLICIES = ("prediction", "mean", "last")




