"""Control a running world: a tiny HTTP API and the gol-ctl CLI.

The server runs on a daemon thread inside gol-run; gol-ctl is a thin client.

    gol-ctl status
    gol-ctl pause | resume
    gol-ctl speed 4
    gol-ctl checkpoint
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import urllib.error
import urllib.request
from typing import TYPE_CHECKING, Any

from aiohttp import web

if TYPE_CHECKING:
    from gol_runtime.loop import SimLoop

DEFAULT_PORT = 7301


class ControlServer:
    def __init__(self, loop: SimLoop, port: int) -> None:
        self.sim = loop
        self.port = port
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------- handlers

    async def _status(self, request: web.Request) -> web.Response:
        world = self.sim.world
        return web.json_response(
            {
                "tick": world.tick,
                "day_fraction": round(world.day_fraction, 4),
                "light_level": round(world.light_level, 3),
                "population": len(world.robots),
                "paused": self.sim.paused,
                "speed": self.sim.speed,
            }
        )

    async def _pause(self, request: web.Request) -> web.Response:
        self.sim.paused = True
        return web.json_response({"paused": True})

    async def _resume(self, request: web.Request) -> web.Response:
        self.sim.paused = False
        return web.json_response({"paused": False})

    async def _speed(self, request: web.Request) -> web.Response:
        body = await request.json()
        value = float(body["value"])
        self.sim.speed = max(0.1, min(64.0, value))
        return web.json_response({"speed": self.sim.speed})

    async def _checkpoint(self, request: web.Request) -> web.Response:
        self.sim.checkpoint_requested = True
        return web.json_response({"requested": True, "tick": self.sim.world.tick})

    # --------------------------------------------------------------- server

    def start(self) -> None:
        app = web.Application()
        app.router.add_get("/status", self._status)
        app.router.add_post("/pause", self._pause)
        app.router.add_post("/resume", self._resume)
        app.router.add_post("/speed", self._speed)
        app.router.add_post("/checkpoint", self._checkpoint)

        def run() -> None:
            web.run_app(
                app,
                host="127.0.0.1",
                port=self.port,
                print=None,
                handle_signals=False,
            )

        self._thread = threading.Thread(target=run, daemon=True, name="control-api")
        self._thread.start()


# -------------------------------------------------------------------- client


def _request(port: int, method: str, path: str, body: dict[str, Any] | None = None) -> Any:
    url = f"http://127.0.0.1:{port}{path}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    if data:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=5) as resp:
        return json.loads(resp.read())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gol-ctl", description=__doc__)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status")
    sub.add_parser("pause")
    sub.add_parser("resume")
    speed = sub.add_parser("speed")
    speed.add_argument("value", type=float)
    sub.add_parser("checkpoint")
    args = parser.parse_args(argv)

    try:
        if args.command == "status":
            out = _request(args.port, "GET", "/status")
        elif args.command == "speed":
            out = _request(args.port, "POST", "/speed", {"value": args.value})
        else:
            out = _request(args.port, "POST", f"/{args.command}")
    except (urllib.error.URLError, ConnectionError) as exc:
        print(f"error: no world listening on port {args.port} ({exc})", file=sys.stderr)
        return 1
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
