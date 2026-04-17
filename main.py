import tkinter as tk
from model import World, Team
from view import WorldView
from controller import Controller


def main():
    world = World(width=100, height=100, tick_rate=10, curriculum_stage=0)

    team_a = Team(team_id=False)
    team_b = Team(team_id=True)
    world.populate_team(team_a, min_formations=1, max_formations=3,
                        min_units=5, max_units=10)
    world.populate_team(team_b, min_formations=1, max_formations=3,
                        min_units=5, max_units=10)

    for unit in team_a.get_all_units():
        unit.set_destination(world.width - 10, unit.get_y())
    for unit in team_b.get_all_units():
        unit.set_destination(10, unit.get_y())

    root = tk.Tk()
    root.title("Space=play/pause, Right=step, R=regen, Esc=quit")

    view = WorldView(root, world, tile_size=8)
    controller = Controller(world, view, root, ms_per_frame=50)

    view.render()
    root.mainloop()


if __name__ == "__main__":
    main()