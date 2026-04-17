import tkinter as tk
from model import World, Team
from view import WorldView


class Controller:
    def __init__(self, world: World, view: WorldView, root: tk.Tk,
                 ms_per_frame: int = 50):
        self.world = world
        self.view = view
        self.root = root
        self.ms_per_frame = ms_per_frame
        self.running = False
        self.tick_count = 0

        self.root.bind('<space>', lambda e: self.toggle_pause())
        self.root.bind('<Escape>', lambda e: self.root.quit())
        self.root.bind('r', lambda e: self.regenerate())
        self.root.bind('<Right>', lambda e: self.step())

    def toggle_pause(self):
        self.running = not self.running
        if self.running:
            self._loop()

    def step(self):
        """Advance a single tick (useful when paused)."""
        self.running = False
        self._do_tick()

    def _do_tick(self):
        self.world.tick()
        self.tick_count += 1
        self.view.render()
        self.root.title(f"Tick {self.tick_count}")

        # Check win condition
        teams = self.world.get_teams()
        for team in teams:
            if team.is_defeated():
                self.running = False
                winner = "Left" if team.team_id else "Right"
                self.root.title(
                    f"Tick {self.tick_count} — {winner} wins!")
                return True  # battle over
        return False

    def _loop(self):
        if not self.running:
            return
        over = self._do_tick()
        if not over:
            self.root.after(self.ms_per_frame, self._loop)

    def regenerate(self):
        self.running = False
        self.tick_count = 0
        self.world.teams = []
        self.world.terrain_map = self.world._generate_terrain_map()
        self.world.height_map  = self.world._generate_height_map()
        import numpy as np
        self.world.tiles = np.stack(
            [self.world.terrain_map, self.world.height_map,
             np.zeros((self.world.height, self.world.width), dtype=int)],
            axis=-1
        )
        self.view.rebuild_terrain()
        self.view.render()
        self.root.title("Regenerated — populate teams and press Space")


if __name__ == "__main__":
    world = World(width=100, height=100, tick_rate=10, curriculum_stage=0)

    team_a = Team(team_id=False)
    team_b = Team(team_id=True)
    world.populate_team(team_a, min_formations=2, max_formations=3,
                        min_units=20, max_units=50)
    world.populate_team(team_b, min_formations=2, max_formations=3,
                        min_units=20, max_units=50)

    for unit in team_a.get_all_units():
        unit.set_destination(world.width - 10, unit.get_y())
    for unit in team_b.get_all_units():
        unit.set_destination(10, unit.get_y())

    root = tk.Tk()
    root.title("Battle — Space=play/pause, Right=step, R=regen, Esc=quit")

    view = WorldView(root, world, tile_size=8)
    controller = Controller(world, view, root, ms_per_frame=50)

    view.render()
    root.mainloop()