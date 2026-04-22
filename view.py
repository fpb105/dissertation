import tkinter as tk
import numpy as np
from PIL import Image, ImageTk
from model import World


class WorldView:
    TERRAIN_COLORS = {
        0: (0, 200, 0),      # Plains - green
        1: (0, 100, 255),    # Water - blue
        2: (139, 69, 19),    # Forest - brown
    }

    UNIT_COLORS = {
        'Infantry': '#ff0000',
        'Archer':   '#ffff00',
        'Cavalry':  '#ff8800',
    }

    def __init__(self, root: tk.Tk, world: World, tile_size: int = 10):
        self.world = world
        self.tile_size = tile_size

        canvas_width = world.width * tile_size
        canvas_height = world.height * tile_size
        self.canvas = tk.Canvas(root, width=canvas_width, height=canvas_height)
        self.canvas.pack()

        self._build_terrain_image()

    def _build_terrain_image(self):
        img = Image.new('RGB', (self.world.width, self.world.height))
        pixels = img.load()
        for y in range(self.world.height):
            for x in range(self.world.width):
                terrain = int(self.world.tiles[y, x, 0])
                height  = int(self.world.tiles[y, x, 1])
                base = self.TERRAIN_COLORS[terrain]
                brightness = 0.4 + (height / 5) * 0.6
                pixels[x, y] = tuple(int(c * brightness) for c in base)
        img = img.resize(
            (self.world.width * self.tile_size,
             self.world.height * self.tile_size),
            Image.NEAREST
        )
        self._terrain_photo = ImageTk.PhotoImage(img)

    def render(self):
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor='nw', image=self._terrain_photo)

        ts = self.tile_size
        half = ts // 2
        for unit in self.world.get_all_units():
            if unit.is_dead():
                continue
            unit_type = type(unit).__name__
            color = self.UNIT_COLORS.get(unit_type, '#ffffff')
            outline = '#0000ff' if unit.get_team() else '#ff0000'

            cx = unit.get_x() * ts
            cy = unit.get_y() * ts

            if unit_type == 'Infantry':
                self.canvas.create_rectangle(
                    cx - half, cy - half, cx + half, cy + half,
                    fill=color, outline=outline, width=2
                )
            elif unit_type == 'Archer':
                self.canvas.create_oval(
                    cx - half, cy - half, cx + half, cy + half,
                    fill=color, outline=outline, width=2
                )
            elif unit_type == 'Cavalry':
                # Triangle pointing up
                self.canvas.create_polygon(
                    cx, cy - half,           # top
                    cx - half, cy + half,    # bottom-left
                    cx + half, cy + half,    # bottom-right
                    fill=color, outline=outline, width=2
                )
            else:
                self.canvas.create_oval(
                    cx - half, cy - half, cx + half, cy + half,
                    fill=color, outline=outline, width=2
                )

    def rebuild_terrain(self):
        self._build_terrain_image()