import tkinter as tk
import numpy as np
from model import World

class WorldView:
    TERRAIN_COLORS = {
        0: '#00c800',    # Grass - green
        1: '#0064ff',    # Water - blue  
        2: '#8b4513'     # Forest - brown
    }

    UNIT_COLORS = {
        'Infantry': '#ff0000',   # Red
        'Archer':   '#ffff00',   # Yellow
        'Cavalry':  '#ff8800',   # Orange
    }
    
    def __init__(self, world: World, tile_size: int = 10):
        self.world = world
        self.tile_size = tile_size
        
        self.root = tk.Tk()
        self.root.title("World Map")
        
        canvas_width = world.width * tile_size
        canvas_height = world.height * tile_size
        self.canvas = tk.Canvas(self.root, width=canvas_width, height=canvas_height)
        self.canvas.pack()
        
        self.root.bind('<Escape>', lambda e: self.root.quit())
        self.root.bind('r', self.regenerate)
    
    def _hex_to_rgb(self, hex_color: str) -> tuple:
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    def _rgb_to_hex(self, rgb: tuple) -> str:
        return f'#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}'
    
    def _get_tile_color(self, terrain: int, height: int) -> str:
        base_hex = self.TERRAIN_COLORS[terrain]
        base_rgb = self._hex_to_rgb(base_hex)
        brightness_factor = 0.4 + (height / 5) * 0.6
        rgb = tuple(int(c * brightness_factor) for c in base_rgb)
        return self._rgb_to_hex(rgb)
    
    def _render_tiles(self):
        """Render all terrain tiles from tiles array"""
        for y in range(self.world.height):
            for x in range(self.world.width):
                # Read from unified tiles array
                terrain = self.world.tiles[y, x, 0]
                height  = self.world.tiles[y, x, 1]
                
                color = self._get_tile_color(terrain, height)
                
                x0 = x * self.tile_size
                y0 = y * self.tile_size
                x1 = x0 + self.tile_size
                y1 = y0 + self.tile_size
                
                self.canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline='')
    
    def _render_units(self):
        """Render all units as circles on top of terrain"""
        for unit in self.world.units:
            # Get unit type name for color lookup
            unit_type = type(unit).__name__
            color = self.UNIT_COLORS.get(unit_type, '#ffffff')
            
            # Team affects the outline color
            outline = '#0000ff' if unit.get_team() else '#ff0000'  # Blue = team 1, Red = team 0
            
            x0 = unit.get_x() * self.tile_size
            y0 = unit.get_y() * self.tile_size
            x1 = x0 + self.tile_size
            y1 = y0 + self.tile_size
            
            # Draw circle for unit
            self.canvas.create_oval(x0, y0, x1, y1, fill=color, outline=outline, width=2)
    
    def render(self):
        """Render full world - tiles then units on top"""
        self.canvas.delete("all")
        self._render_tiles()
        self._render_units()
    
    def regenerate(self, event=None):
        """Regenerate world on 'r' key"""
        self.world.terrain_map = self.world._generate_terrain_map()
        self.world.height_map  = self.world._generate_height_map()
        self.world.tiles = np.stack([self.world.terrain_map, self.world.height_map], axis=-1)
        self.world.units = []
        self.render()
    
    def run(self):
        self.render()
        self.root.mainloop()