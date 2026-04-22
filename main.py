"""
Visual command tester.

Two infantry formations face each other. The LEFT formation
executes the current command. The RIGHT formation holds still.
Press SPACE to cycle to the next command. Press R to restart
the current command.

Controls:
    SPACE  - next command
    R      - restart current command
    ESC    - quit
"""

import tkinter as tk
import numpy as np
from model import World, Team, Formation, Infantry
from view import WorldView
from formation_commands import (
    execute_command, CMD_HOLD, CMD_ADVANCE, CMD_RETREAT,
    CMD_FLANK_LEFT, CMD_FLANK_RIGHT, CMD_FACE_CLOSEST
)

MAP_W, MAP_H = 300, 300
TICK_RATE = 10
UNIT_COUNT = 20
FORMATION_WIDTH = 40
MS_PER_FRAME = 50

COMMANDS = [
    (CMD_HOLD,         "HOLD"),
    (CMD_ADVANCE,      "ADVANCE"),
    (CMD_RETREAT,      "RETREAT"),
    (CMD_FLANK_LEFT,   "FLANK LEFT"),
    (CMD_FLANK_RIGHT,  "FLANK RIGHT"),
    (CMD_FACE_CLOSEST, "FACE CLOSEST"),
]


def make_formation(team_id, x, y, width=FORMATION_WIDTH):
    units = [Infantry() for _ in range(UNIT_COUNT)]
    for u in units:
        u.set_team(team_id)
    f = Formation(units)
    f.set_port(x, y - width / 2)
    f.set_starboard(x, y + width / 2)
    f.place_units()
    return f


class CommandTester:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.command_index = 0
        self.tick_count = 0
        self.running = False

        self.world = None
        self.friendly_team = None
        self.enemy_team = None
        self.active_formation = None
        self.static_formation = None
        self.view = None

        self._setup_world()

        self.view = WorldView(root, self.world, tile_size=3)

        root.bind('<space>', lambda e: self._next_command())
        root.bind('r', lambda e: self._restart_command())
        root.bind('<Escape>', lambda e: root.quit())

        self._update_title()
        self.view.render()
        self._start()

    def _setup_world(self):
        """Create fresh world with two formations."""
        self.world = World(MAP_W, MAP_H, TICK_RATE, curriculum_stage=0)

        self.friendly_team = Team(team_id=False)
        self.enemy_team = Team(team_id=True)

        # Left formation (this one moves)
        self.active_formation = make_formation(False, 100, MAP_H // 2)
        self.friendly_team.add_formation(self.active_formation)

        # Right formation (stays still)
        self.static_formation = make_formation(True, 200, MAP_H // 2)
        self.enemy_team.add_formation(self.static_formation)

        self.world.add_team(self.friendly_team)
        self.world.add_team(self.enemy_team)

        self.tick_count = 0

    def _reset_formations(self):
        """Reset both formations to starting positions."""
        self.world.teams = []

        self.friendly_team = Team(team_id=False)
        self.enemy_team = Team(team_id=True)

        self.active_formation = make_formation(False, 100, MAP_H // 2)
        self.friendly_team.add_formation(self.active_formation)

        self.static_formation = make_formation(True, 200, MAP_H // 2)
        self.enemy_team.add_formation(self.static_formation)

        self.world.add_team(self.friendly_team)
        self.world.add_team(self.enemy_team)

        self.tick_count = 0

    def _next_command(self):
        """Cycle to the next command and reset."""
        self.command_index = (self.command_index + 1) % len(COMMANDS)
        self._restart_command()

    def _restart_command(self):
        """Reset formations and restart current command."""
        self._reset_formations()
        self._update_title()
        self.view.render()

    def _update_title(self):
        cmd_id, cmd_name = COMMANDS[self.command_index]
        self.root.title(
            f"Command: {cmd_name} ({self.command_index + 1}/{len(COMMANDS)})  "
            f"Tick: {self.tick_count}  |  "
            f"SPACE=next  R=restart  ESC=quit"
        )

    def _do_tick(self):
        cmd_id, cmd_name = COMMANDS[self.command_index]

        # Active formation executes the command
        execute_command(
            self.active_formation, cmd_id, TICK_RATE,
            facing_right=True,
            enemy_formations=[self.static_formation]
        )

        # Static formation does nothing (hold)
        # No execute_command call needed

        # Advance simulation
        self.world.tick()
        self.tick_count += 1

        # Log position
        living = self.active_formation.get_living_units()
        if living:
            ax = sum(u.get_x() for u in living) / len(living)
            ay = sum(u.get_y() for u in living) / len(living)
            px, py = self.active_formation.get_port()
            sx, sy = self.active_formation.get_starboard()
            print(f"  [{cmd_name:12s}] tick={self.tick_count:3d}  "
                  f"avg=({ax:.1f},{ay:.1f})  "
                  f"port=({px:.1f},{py:.1f})  "
                  f"star=({sx:.1f},{sy:.1f})")

        self.view.render()
        self._update_title()

    def _start(self):
        """Auto-run ticks continuously."""
        self.running = True
        self._loop()

    def _loop(self):
        if not self.running:
            return
        self._do_tick()
        self.root.after(MS_PER_FRAME, self._loop)


if __name__ == "__main__":
    root = tk.Tk()
    tester = CommandTester(root)
    root.mainloop()