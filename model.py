from __future__ import annotations

import enum
import numpy as np
from scipy.signal import convolve2d
import math
from abc import ABC, abstractmethod


class Unit(ABC):
    def __init__(self):
        self.hp: int = 0
        self.x: float = 0.0
        self.y: float = 0.0
        self.damage: int = 0
        self.range: int = 0
        self.speed: float = 0.0
        self.team: bool = False
        self.destination: tuple[float, float] | None = None

    def get_hp(self) -> int:
        return self.hp

    def get_team(self) -> bool:
        return self.team

    def get_x(self) -> float:
        return self.x

    def get_y(self) -> float:
        return self.y

    def get_damage(self) -> int:
        return self.damage

    def get_range(self) -> int:
        return self.range

    def set_team(self, team: bool):
        self.team = team

    def set_hp(self, hp: int):
        self.hp = hp

    def set_x(self, x: float):
        self.x = x

    def set_y(self, y: float):
        self.y = y

    def is_dead(self) -> bool:
        return self.hp <= 0

    def move(self, destination: tuple[float, float], tick_rate: int):
        """
        Step toward destination by (speed / tick_rate) units.
        tick_rate is passed directly — Unit no longer needs a World reference.
        """
        dx = destination[0] - self.x
        dy = destination[1] - self.y
        dist = math.hypot(dx, dy)
        if dist > 0:
            step = self.speed / tick_rate
            ratio = min(step / dist, 1.0)
            self.x += dx * ratio
            self.y += dy * ratio

    def attack(self, target: Unit):
        """
        Deal damage to target. World reference removed — targeting logic
        lives in Formation/World, not here.
        """
        target.set_hp(target.get_hp() - self.damage)

    def on_tick(self, tick_rate: int):
        """
        Called each tick by Formation. Moves toward destination if one is set.
        tick_rate passed directly instead of a World object.
        """
        if self.destination is not None:
            self.move(self.destination, tick_rate)
            if math.hypot(self.destination[0] - self.x, self.destination[1] - self.y) < 0.01:
                self.destination = None


class Infantry(Unit):
    def __init__(self):
        super().__init__()
        self.hp = 100
        self.damage = 10
        self.range = 1
        self.speed = 1.0


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
        self.speed = 3.0


class Formation:
    def __init__(self, units: list[Unit]):
        self.units = units
        self.portx: float = 0.0
        self.porty: float = 0.0
        self.starboardx: float = 0.0
        self.starboardy: float = 0.0

    def place_units(self):
        """
        Directly set each unit's (x, y) along the port-to-starboard line.
        Uses parametric form so vertical lines don't cause division by zero:
            P(t) = port + t * (starboard - port),  t in [0, 1]
        """
        dx = self.starboardx - self.portx
        dy = self.starboardy - self.porty
        n = len(self.units)
        for i, unit in enumerate(self.units):
            t = i / (n - 1) if n > 1 else 0.5
            unit.set_x(self.portx + t * dx)
            unit.set_y(self.porty + t * dy)

    def move(self, portx: float, porty: float,
             starboardx: float, starboardy: float,
             tick_rate: int):
        """
        Order all units to step toward their new positions along the given line.
        World reference replaced with tick_rate passed directly.
        """
        dx = starboardx - portx
        dy = starboardy - porty
        n = len(self.units)
        for i, unit in enumerate(self.units):
            t = i / (n - 1) if n > 1 else 0.5
            target_x = portx + t * dx
            target_y = porty + t * dy
            unit.move((target_x, target_y), tick_rate)

    def on_tick(self, tick_rate: int):
        """Called by World.tick() — propagates tick down to each unit."""
        for unit in self.units:
            unit.on_tick(tick_rate)

    def get_units(self) -> list[Unit]:
        return self.units

    def get_living_units(self) -> list[Unit]:
        return [u for u in self.units if not u.is_dead()]

    def get_units_around_point(self, x: float, y: float, radius: float) -> list[Unit]:
        return [u for u in self.units
                if math.hypot(u.get_x() - x, u.get_y() - y) <= radius]

    def add_unit(self, unit: Unit, x: float, y: float):
        unit.set_x(x)
        unit.set_y(y)
        self.units.append(unit)

    def remove_unit(self, unit: Unit):
        self.units.remove(unit)

    def set_port(self, x: float, y: float):
        self.portx = x
        self.porty = y

    def set_starboard(self, x: float, y: float):
        self.starboardx = x
        self.starboardy = y

    def get_port(self) -> tuple[float, float]:
        return (self.portx, self.porty)

    def get_starboard(self) -> tuple[float, float]:
        return (self.starboardx, self.starboardy)


class World:

    class Units(enum.IntEnum):
        Infantry = 10
        Archer = 11
        Cavalry = 12

    def __init__(self, width: int, height: int, tick_rate: int):
        self.width = width
        self.height = height
        self.tick_rate = tick_rate
        # Plain data attributes first — if terrain generation throws,
        # the object isn't left in a half-constructed state without these.
        self.units: list[Unit] = []
        self.formations: list[Formation] = []
        self.terrain_map = self._generate_terrain_map()
        self.height_map = self._generate_height_map()
        self.tiles = np.stack(
            [self.terrain_map, self.height_map,
             np.zeros((self.height, self.width), dtype=int)],
            axis=-1
        )

    # ------------------------------------------------------------------
    # Terrain generation (unchanged)
    # ------------------------------------------------------------------

    def _generate_perlin_noise(self, scale: float = 50.0, octaves: int = 6,
                               persistence: float = 0.5, lacunarity: float = 2.0) -> np.ndarray:
        noise_map = np.zeros((self.height, self.width))
        amplitude = 1.0
        frequency = 1.0
        max_value = 0.0
        for octave in range(octaves):
            octave_noise = self._generate_smooth_noise(scale / frequency)
            noise_map += octave_noise * amplitude
            max_value += amplitude
            amplitude *= persistence
            frequency *= lacunarity
        return noise_map / max_value

    def _generate_smooth_noise(self, scale: float) -> np.ndarray:
        low_res_h = int(self.height / scale) + 2
        low_res_w = int(self.width / scale) + 2
        random_grid = np.random.rand(low_res_h, low_res_w)
        xs = np.arange(self.width)
        ys = np.arange(self.height)
        grid_x = xs / scale
        grid_y = ys / scale
        x0 = grid_x.astype(int)
        y0 = grid_y.astype(int)
        x1 = x0 + 1
        y1 = y0 + 1
        sx = grid_x - x0
        sy = grid_y - y0
        sx = (1 - np.cos(sx * np.pi)) / 2
        sy = (1 - np.cos(sy * np.pi)) / 2
        sx_2d, sy_2d = np.meshgrid(sx, sy)
        x0_2d, y0_2d = np.meshgrid(x0, y0)
        x1_2d, y1_2d = np.meshgrid(x1, y1)
        top_left     = random_grid[y0_2d, x0_2d]
        top_right    = random_grid[y0_2d, x1_2d]
        bottom_left  = random_grid[y1_2d, x0_2d]
        bottom_right = random_grid[y1_2d, x1_2d]
        top    = top_left  * (1 - sx_2d) + top_right  * sx_2d
        bottom = bottom_left * (1 - sx_2d) + bottom_right * sx_2d
        return top * (1 - sy_2d) + bottom * sy_2d

    def _apply_cellular_automata(self, grid: np.ndarray, iterations: int = 3) -> np.ndarray:
        smoothed = np.copy(grid)
        kernel = np.ones((3, 3), dtype=int)
        for _ in range(iterations):
            unique_vals = np.unique(smoothed)
            counts = np.zeros((len(unique_vals), self.height, self.width))
            for i, val in enumerate(unique_vals):
                mask = (smoothed == val).astype(int)
                counts[i] = convolve2d(mask, kernel, mode='same', boundary='fill')
            winner_indices = np.argmax(counts, axis=0)
            smoothed = unique_vals[winner_indices]
        return smoothed

    def _generate_terrain_map(self) -> np.ndarray:
        noise = self._generate_perlin_noise(scale=30.0, octaves=6, persistence=0.5)
        terrain_map = np.zeros((self.height, self.width), dtype=int)
        terrain_map[noise < 0.35] = 1
        terrain_map[(noise >= 0.35) & (noise < 0.65)] = 0
        terrain_map[noise >= 0.65] = 2
        return self._apply_cellular_automata(terrain_map, iterations=5)

    def _generate_height_map(self) -> np.ndarray:
        noise = self._generate_perlin_noise(scale=25.0, octaves=4, persistence=0.6)
        height_map = np.clip((noise * 5).astype(int), 0, 5)
        height_map = self._apply_cellular_automata(height_map, iterations=3)
        height_map[self.terrain_map == 1] = 0
        return height_map

    # ------------------------------------------------------------------
    # Population
    # ------------------------------------------------------------------

    def populate_teams(self):
        """
        Formations are spread evenly down the map vertically.
        Team False = left side, Team True = right side.
        """
        num_formations = 1
        units_per_formation = 50
        half_height = 40

        for i in range(num_formations):
            y_centre = int((i + 0.5) * self.height / num_formations)
            y_top    = max(0, y_centre - half_height)
            y_bottom = min(self.height - 1, y_centre + half_height)

            f1 = Formation([Infantry() for _ in range(units_per_formation)])
            f2 = Formation([Infantry() for _ in range(units_per_formation)])

            f1.set_port(10, y_top)
            f1.set_starboard(10, y_bottom)
            f2.set_port(self.width - 10, y_top)
            f2.set_starboard(self.width - 10, y_bottom)

            f1.place_units()
            f2.place_units()

            self.add_formation(f1, team=False)
            self.add_formation(f2, team=True)

    # ------------------------------------------------------------------
    # World management
    # ------------------------------------------------------------------

    def add_formation(self, formation: Formation, team: bool | None = None):
        self.formations.append(formation)
        for unit in formation.get_units():
            if team is not None:
                unit.set_team(team)
            self.units.append(unit)

    def remove_formation(self, formation: Formation):
        self.formations.remove(formation)
        for unit in formation.get_units():
            if unit in self.units:
                self.units.remove(unit)

    def add_unit(self, unit: Unit, x: float, y: float):
        unit.set_x(x)
        unit.set_y(y)
        self.units.append(unit)

    def remove_unit(self, unit: Unit):
        self.units.remove(unit)

    def tick(self):
        """
        Advance world state by one tick.
        World passes tick_rate down to formations; formations pass it to units.
        No object passes itself downward.
        """
        for formation in self.formations:
            formation.on_tick(self.tick_rate)