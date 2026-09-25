"""Report how often validation selects a single layer over all layers.

This is the concise entry point for the layer-selection decision statistic.
It delegates to :mod:`analysis.layer_selection_appendix`, which also writes
the audit table with one row per target-class/contrast decision.

Example
-------
python analysis/layer_selection_performance.py --source gpt2-small
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.layer_selection_appendix import main  # noqa: E402


if __name__ == "__main__":
    main()
