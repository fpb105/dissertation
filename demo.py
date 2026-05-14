"""
Visual scenarios for the battle simulator.

Each scenario sets up a hand-crafted situation and runs the sim in a
Tk window using your existing WorldView. Good for spot-checking
behaviour and for supervisor demos.

Usage:
    python demo_scenarios.py                  # list scenarios
    python demo_scenarios.py move             # formation marches across the map
    python demo_scenarios.py widen            # formation widens at tick 80
    python demo_scenarios.py shrink           # formation shrinks at tick 80
    python demo_scenarios.py wall             # red raises shield wall mid-fight
    python demo_scenarios.py cavalry          # cavalry charges infantry
    python demo_scenarios.py battle           # full random battle via populate_team

Assumes your view file is called `view.py` and defines `WorldView`.
If you named it differently, update the import below.
"""

from __future__ import annotations

import sys
import tkinter as tk
import numpy as np

from model import (
    World, Team, Formation,
    Infantry, Archer, Cavalry,
    Terrain,
)
from view import WorldView


TICK_RATE = 30
FRAME_MS = 33  # ~30 FPS render/tick cadence


# ============================================================
# Shared setup
# ============================================================

def flat_world(width: int = 120, height: int = 60, seed: int = 0) -> World:
    """A deterministic, flat plains world with zero height."""
    np.random.seed(seed)
    w = World(width, height, tick_rate=TICK_RATE, curriculum_stage=0)
    w.terrain_map[:] = Terrain.PLAINS
    w.height_map[:] = 0
    w.tiles[..., 0] = w.terrain_map
    w.tiles[..., 1] = w.height_map
    return w


def run(root: tk.Tk, world: World, view: WorldView,
        step_hook=None, duration_ticks: int = 600):
    """
    Drive the sim with a tkinter `after` loop.

    `step_hook(tick_index)` is called before each tick; use it to
    change orders at specific times (e.g. raise shield wall at t=60).
    """
    state = {"t": 0}

    def loop():
        if step_hook is not None:
            step_hook(state["t"])
        world.tick()
        view.render()
        state["t"] += 1
        if state["t"] < duration_ticks:
            root.after(FRAME_MS, loop)

    root.after(FRAME_MS, loop)


def new_window(title: str) -> tk.Tk:
    root = tk.Tk()
    root.title(f"battle-sim: {title}")
    return root


# ============================================================
# Helpers to build test armies
# ============================================================

def add_formation(team: Team, unit_cls, count: int,
                  px, py, sx, sy) -> Formation:
    units = [unit_cls() for _ in range(count)]
    for u in units:
        u.set_team(team.team_id)
    f = Formation(units)
    f.move(px=px, py=py, sx=sx, sy=sy)
    team.add_formation(f)
    return f


# ============================================================
# Scenarios
# ============================================================

def scenario_move():
    """Single infantry formation marches left to right."""
    world = flat_world(width=120, height=60)
    team = Team(team_id=False)
    formation = add_formation(team, Infantry, count=12,
                              px=10, py=25, sx=10, sy=45)
    world.add_team(team)

    def step(t):
        if t == 60:
            formation.move(px=100, py=25, sx=100, sy=45)

    root = new_window("move")
    view = WorldView(root, world, tile_size=7)
    run(root, world, view, step_hook=step, duration_ticks=600)
    root.mainloop()


def scenario_widen():
    """Narrow column → wide line at tick 80."""
    world = flat_world(width=120, height=70)
    team = Team(team_id=False)
    formation = add_formation(team, Archer, count=25,
                              px=60, py=30, sx=60, sy=40)
    world.add_team(team)

    def step(t):
        if t == 80:
            formation.move(px=60, py=5, sx=60, sy=65)

    root = new_window("widen")
    view = WorldView(root, world, tile_size=7)
    run(root, world, view, step_hook=step, duration_ticks=600)
    root.mainloop()


def scenario_shrink():
    """Wide line → narrow column at tick 80 (watch ranks stack up)."""
    world = flat_world(width=120, height=70)
    team = Team(team_id=False)
    formation = add_formation(team, Archer, count=25,
                              px=60, py=5, sx=60, sy=65)
    world.add_team(team)

    def step(t):
        if t == 80:
            formation.move(px=60, py=30, sx=60, sy=40)

    root = new_window("shrink")
    view = WorldView(root, world, tile_size=7)
    run(root, world, view, step_hook=step, duration_ticks=600)
    root.mainloop()


def scenario_wall():
    """
    Two infantry lines fight. Red raises shield wall at tick 60.
    You should see red dealing less damage but also taking less
    from the walled-infantry side of the exchange — and crucially,
    archers would do almost nothing if you add them.
    """
    world = flat_world(width=120, height=60)

    red = Team(team_id=False)
    red_form = add_formation(red, Infantry, count=10,
                             px=40, py=15, sx=40, sy=45)

    blue = Team(team_id=True)
    add_formation(blue, Infantry, count=10,
                  px=60, py=15, sx=60, sy=45)
    # Add an archer screen behind blue infantry to make the
    # archer-vs-wall damage reduction visible.
    add_formation(blue, Archer, count=8,
                  px=80, py=20, sx=80, sy=40)

    world.add_team(red)
    world.add_team(blue)

    def step(t):
        if t == 1:
            print(">>> red raises shield wall")
            for u in red_form.get_units():
                u.set_wall(True)

    root = new_window("shield wall")
    view = WorldView(root, world, tile_size=7)
    run(root, world, view, step_hook=step, duration_ticks=1200)
    root.mainloop()


def scenario_cavalry():
    """
    Cavalry charges infantry. Demonstrates can_move_while_attacking:
    Cavalry keeps advancing through targets, while ordinary infantry
    would freeze the moment they start attacking.
    """
    world = flat_world(width=160, height=60)

    red = Team(team_id=False)
    cav_form = add_formation(red, Cavalry, count=8,
                             px=20, py=25, sx=20, sy=35)

    blue = Team(team_id=True)
    add_formation(blue, Infantry, count=15,
                  px=110, py=15, sx=110, sy=45)

    world.add_team(red)
    world.add_team(blue)

    def step(t):
        if t == 20:
            print(">>> cavalry charges")
            cav_form.move(px=100, py=25, sx=100, sy=35)

    root = new_window("cavalry charge")
    view = WorldView(root, world, tile_size=6)
    run(root, world, view, step_hook=step, duration_ticks=1200)
    root.mainloop()


def scenario_battle():
    """Two auto-populated random armies. Sanity check for the full pipeline."""
    world = flat_world(width=200, height=100, seed=12)
    world.populate_team(Team(team_id=False))
    world.populate_team(Team(team_id=True))

    root = new_window("battle")
    view = WorldView(root, world, tile_size=5)
    run(root, world, view, duration_ticks=2000)
    root.mainloop()


# ============================================================
# CLI
# ============================================================

SCENARIOS = {
    "move":    scenario_move,
    "widen":   scenario_widen,
    "shrink":  scenario_shrink,
    "wall":    scenario_wall,
    "cavalry": scenario_cavalry,
    "battle":  scenario_battle,
}


def _usage():
    print("Available scenarios:")
    for name in SCENARIOS:
        print(f"  python demo_scenarios.py {name}")


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in SCENARIOS:
        _usage()
        sys.exit(0)
    SCENARIOS[sys.argv[1]]()