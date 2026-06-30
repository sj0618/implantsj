#!/usr/bin/env python3
"""Alias for evaluating a 6-label SupCon metric checkpoint on the 3-label task."""
from __future__ import annotations

import sys

from evaluate_3label_metric_from_7label import main


if __name__ == "__main__":
    if "--experiment-name" not in sys.argv:
        sys.argv.extend(["--experiment-name", "metric_6label_supcon_to_3label"])
    main()
