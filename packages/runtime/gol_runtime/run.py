"""gol-run: create or resume a persistent world and run it.

gol-run saves/alpha                      # resume if it exists, else create
gol-run --new saves/alpha                # create (error if it exists)
gol-run --resume saves/alpha             # resume (error if it doesn't)
gol-run saves/alpha --headless --ticks 5000
gol-run saves/alpha --set world.seed=11 --set tick_rate=40
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

from gol_world import persistence
from gol_world.world import World

from gol_runtime.config import load_run_config
from gol_runtime.control import ControlServer
from gol_runtime.loop import SimLoop
from gol_runtime.scheduler import Population


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gol-run", description=__doc__)
    parser.add_argument("save_dir", type=Path, help="save directory (the world's life)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--new", action="store_true", help="create a new world")
    mode.add_argument("--resume", action="store_true", help="resume an existing world")
    parser.add_argument("--config", default="configs/run/local_m1.yaml", help="run config YAML")
    parser.add_argument(
        "--set",
        dest="sets",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override config (prefix world. for world config), repeatable",
    )
    parser.add_argument("--headless", action="store_true", help="no viewer, unpaced")
    parser.add_argument("--ticks", type=int, default=None, help="stop after N ticks")
    parser.add_argument(
        "--rrd", type=Path, default=None, help="record to a .rrd file instead of spawning a viewer"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    save_dir: Path = args.save_dir
    exists = (save_dir / "manifest.json").exists()

    if args.new and exists:
        print(f"error: {save_dir} already contains a world (use --resume)", file=sys.stderr)
        return 1
    if args.resume and not exists:
        print(f"error: {save_dir} has no world to resume", file=sys.stderr)
        return 1

    run_cfg, world_cfg = load_run_config(args.config, args.sets)

    if exists:
        world = persistence.load_world(save_dir)
        print(f"resuming {save_dir} at tick {world.tick}")
    else:
        persistence.create_save(save_dir, world_cfg, run_config=dataclasses.asdict(run_cfg))
        world = World.new(world_cfg)
        print(f"created {save_dir} (seed {world_cfg.seed}, size {world_cfg.size})")

    population = Population(world, run_cfg)
    if exists:
        ckpt = persistence.latest_checkpoint(save_dir)
        if ckpt is not None:
            population.restore_brain_states(persistence.load_brain_states(ckpt))

    log_frame = None
    if run_cfg.observability.rerun and (not args.headless or args.rrd):
        from gol_obs.rerun_log import RerunLogger

        logger = RerunLogger(
            world,
            tick_rate=run_cfg.tick_rate,
            spawn=not args.headless and args.rrd is None,
            save_path=args.rrd,
        )

        def log_frame(w: World) -> None:
            logger.log_frame(w, obs=population.last_obs, introspection=population.introspection())

    loop = SimLoop(
        world,
        save_dir,
        run_cfg,
        log_frame=log_frame,
        brain_states=population.brain_states,
        act_step=population.act_step,
    )
    ControlServer(loop, port=run_cfg.control_port).start()
    try:
        loop.run(max_ticks=args.ticks, paced=not args.headless)
    except KeyboardInterrupt:
        print(f"\nstopping; checkpointing at tick {world.tick}...")
        loop.checkpoint()
    print(f"world is at tick {world.tick} ({world.tick / run_cfg.tick_rate:.0f}s of sim time)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
