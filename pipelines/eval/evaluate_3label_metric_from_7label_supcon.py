#!/usr/bin/env python3
"""Evaluate a 7-label SupCon metric checkpoint on the canonical 3-label task."""
from __future__ import annotations

import sys

from evaluate_3label_metric_from_7label import main


if __name__ == "__main__":
    if "--experiment-name" not in sys.argv:
        sys.argv.extend(["--experiment-name", "metric_7label_supcon_to_3label"])
    main()
