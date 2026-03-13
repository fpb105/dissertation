import numpy as np
from scipy.signal import convolve2d
from scipy import stats
import time
from array import array
from abc import ABC, abstractmethod
import math

class World:
    def __init__(self, width: int, height: int, tick_rate: int):
        self.width = width
        self.height = height
        self.tick_rate = tick_rate
        self.terrain_map = self._generate_terrain_map()
        self.height_map = self._generate_height_map()
        self.units: list = []
        self.tiles = np.stack([self.terrain_map, self.height_map], axis=-1)

    def _generate_perlin_noise(self, scale: float = 50.0, octaves: int = 6, 
                               persistence: float = 0.5, lacunarity: float = 2.0) -> np.ndarray:
        
        """
        Generate Perlin-like noise using numpy (simplified implementation)
        
        Args:
            scale: Controls feature size (larger = bigger features)
            octaves: Number of noise layers (more = more detail)
            persistence: Amplitude multiplier per octave (0-1, lower = smoother)
            lacunarity: Frequency multiplier per octave (typically 2.0)
        """

        start = time.perf_counter()

        noise_map = np.zeros((self.height, self.width))
        
        amplitude = 1.0
        frequency = 1.0
        max_value = 0.0
        
        for octave in range(octaves):
            # Generate smooth noise at this octave's frequency
            octave_noise = self._generate_smooth_noise(scale / frequency)
            
            # Add weighted octave to result
            noise_map += octave_noise * amplitude
            
            max_value += amplitude
            amplitude *= persistence
            frequency *= lacunarity
        
        # Normalize to [0, 1]
        noise_map = noise_map / max_value
        
        elapsed = time.perf_counter() - start
        print(f"Took {elapsed:.4f} seconds")

        return noise_map
    
    def _generate_smooth_noise(self, scale: float) -> np.ndarray:

        start = time.perf_counter()

        # Generate low-resolution random grid
        low_res_h = int(self.height / scale) + 2
        low_res_w = int(self.width / scale) + 2
        random_grid = np.random.rand(low_res_h, low_res_w)

        # --- THIS IS THE NEW BIT ---

        # 1. Create coordinate arrays for every pixel
        xs = np.arange(self.width)   # [0, 1, 2, ..., 511]
        ys = np.arange(self.height)  # [0, 1, 2, ..., 511]

        # 2. Map pixel coords to grid coords (all at once)
        grid_x = xs / scale  # e.g. [0.0, 0.033, 0.066, ...]
        grid_y = ys / scale

        # 3. Get integer parts (which grid cell each pixel falls in)
        x0 = grid_x.astype(int)  # floor
        y0 = grid_y.astype(int)
        x1 = x0 + 1
        y1 = y0 + 1

        # 4. Get fractional parts (where within that cell)
        sx = grid_x - x0  # 0.0 to 1.0
        sy = grid_y - y0

        # 5. Cosine interpolation (smooth the fractions)
        sx = (1 - np.cos(sx * np.pi)) / 2
        sy = (1 - np.cos(sy * np.pi)) / 2

        # 6. Use meshgrid to expand 1D arrays into 2D grids
        #    sx_2d has shape (height, width) — sx repeated down rows
        #    sy_2d has shape (height, width) — sy repeated across columns
        sx_2d, sy_2d = np.meshgrid(sx, sy)
        x0_2d, y0_2d = np.meshgrid(x0, y0)
        x1_2d, y1_2d = np.meshgrid(x1, y1)

        # 7. Look up all four corner values at once
        top_left     = random_grid[y0_2d, x0_2d]   # shape (height, width)
        top_right    = random_grid[y0_2d, x1_2d]
        bottom_left  = random_grid[y1_2d, x0_2d]
        bottom_right = random_grid[y1_2d, x1_2d]

        # 8. Bilinear interpolation (all pixels simultaneously)
        top    = top_left * (1 - sx_2d) + top_right * sx_2d
        bottom = bottom_left * (1 - sx_2d) + bottom_right * sx_2d
        smooth = top * (1 - sy_2d) + bottom * sy_2d

        elapsed = time.perf_counter() - start
        print(f"Took {elapsed:.4f} seconds")

        return smooth
    
    def _apply_cellular_automata(self, grid: np.ndarray, iterations: int = 3) -> np.ndarray:

        start = time.perf_counter()

        smoothed = np.copy(grid)

        # Kernel that selects the 3x3 neighborhood
        kernel = np.ones((3, 3), dtype=int)

        for iteration in range(iterations):
            # For each unique terrain value, count how many neighbors have that value
            unique_vals = np.unique(smoothed)
            counts = np.zeros((len(unique_vals), self.height, self.width))

            for i, val in enumerate(unique_vals):
                # Create binary mask: 1 where terrain == val, 0 elsewhere
                mask = (smoothed == val).astype(int)

                # Convolve with 3x3 kernel = count neighbors with this value
                counts[i] = convolve2d(mask, kernel, mode='same', boundary='fill')

            # Pick the value with the highest count at each pixel
            winner_indices = np.argmax(counts, axis=0)
            smoothed = unique_vals[winner_indices]

        elapsed = time.perf_counter() - start
        print(f"Took {elapsed:.4f} seconds")

        return smoothed
    
    def _generate_terrain_map(self) -> np.ndarray:
        """
        Generate terrain map using Perlin-like noise + Cellular Automata
        
        Pipeline:
        1. Generate smooth noise (Perlin-like with fBm)
        2. Threshold into terrain types
        3. Apply CA to create clustered regions
        """

        start = time.perf_counter()

        # Step 1: Generate noise with fBm
        noise = self._generate_perlin_noise(scale=30.0, octaves=6, persistence=0.5)
        
        # Step 2: Threshold into terrain types
        terrain_map = np.zeros((self.height, self.width), dtype=int)
        terrain_map[noise < 0.35] = 1  # Water
        terrain_map[(noise >= 0.35) & (noise < 0.65)] = 0  # Grass
        terrain_map[noise >= 0.65] = 2  # Forest
        
        # Step 3: Apply Cellular Automata for region clustering
        terrain_map = self._apply_cellular_automata(terrain_map, iterations=5)
        
        elapsed = time.perf_counter() - start
        print(f"Took {elapsed:.4f} seconds")

        return terrain_map
    
    def _generate_height_map(self) -> np.ndarray:
        """
        Generate height map using noise + CA, then set water to height 0
        
        Pipeline:
        1. Generate noise for height
        2. Convert to discrete heights (0-5)
        3. Apply CA to smooth
        4. Set all water tiles to height 0
        """

        start = time.perf_counter()

        # Step 1: Generate noise for height
        noise = self._generate_perlin_noise(scale=25.0, octaves=4, persistence=0.6)
        
        # Step 2: Convert to discrete heights (0-5)
        height_map = (noise * 5).astype(int)
        height_map = np.clip(height_map, 0, 5)  # Ensure within range
        
        # Step 3: Apply CA to create smoother height transitions
        height_map = self._apply_cellular_automata(height_map, iterations=3)
        
        # Step 4: Set all water tiles to height 0
        height_map[self.terrain_map == 1] = 0
        
        elapsed = time.perf_counter() - start
        print(f"Took {elapsed:.4f} seconds")

        return height_map
    
    def get_units_around_point(self, x: int, y: int, radius: int) -> list:
        """Return list of units within radius of (x, y)"""
        nearby_units = []
        for unit in self.units:
            if abs(unit.get_x() - x) <= radius and abs(unit.get_y() - y) <= radius:
                nearby_units.append(unit)
        return nearby_units

    def add_unit(self, unit: 'Unit'):
        self.units.append(unit)
        self.tiles[unit.get_y(), unit.get_x(), 0] = "o"  # Mark tile as occupied by unit

    def remove_unit(self, unit: 'Unit'):
        self.units.remove(unit)

    def tick(self):
        """Advance world state by one tick (move units, resolve combat, etc.)"""
        # Placeholder for future logic
        pass

# diddy seperation of diddy classes because im diddy

class Unit(ABC):
    def __init__(self):
        self.hp: int
        self.x: int
        self.y: int
        self.damage: int
        self.range:int
        self.team: bool
        self.speed: int

    def get_hp(self) -> int:
        return self.hp
    
    def get_team(self) -> bool:
        return self.team

    def get_x(self) -> int:
        return self.x

    def get_y(self) -> int:
        return self.y

    def get_damage(self) -> int:
        return self.damage

    def get_range(self) -> int:
        return self.range
    
    def set_team(self, team: bool):
        self.team = team

    def set_hp(self, hp: int):
        self.hp = hp
        if self.hp <= 0:
            self = None

    def set_x(self, x: int):
        self.x = x

    def set_y(self, y: int):
        self.y = y

    def move(self, destination: tuple[int, int], world: World, speed: int):
        dx = destination[0] - self.x
        dy = destination[1] - self.y
        dist = math.hypot(dx, dy)
        if dist > 0:
            self.x += (dx / dist) * speed
            self.y += (dy / dist) * speed

    def attack(self, target: 'Unit', world: World):
        world.get_units_around_point(self.x, self.y, self.range)
        target.set_hp(target.get_hp() - self.damage)

class Infantry(Unit):
    def __init__(self):
        super().__init__()
        self.hp = 100
        self.damage = 10
        self.range = 1
        self.speed = 1

class Archer(Unit):
    def __init__(self):
        super().__init__()
        self.hp = 80
        self.damage = 15
        self.range = 3
        self.speed = 1.2

class Cavalry(Unit):
    def __init__(self):
        super().__init__()
        self.hp = 120
        self.damage = 7
        self.range = 1
        self.speed = 3
