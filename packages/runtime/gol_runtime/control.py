"""gol-ctl: control a running world (pause, speed, checkpoint).

The HTTP control endpoint lands in M1; this CLI is a stub until then.
"""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    print("gol-ctl: the control API arrives with milestone M1", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
