"""gol-stats: analyze a save dir's metrics and events.

Metrics/events logging lands in M2; this CLI is a stub until then.
"""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    print("gol-stats: metrics and events arrive with milestone M2", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
