#!/usr/bin/env python3
"""Print the TileGen environment probe as JSON."""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tilegen.env import probe  # noqa: E402


def main() -> int:
    print(json.dumps(probe(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
