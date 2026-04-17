from __future__ import annotations

import enum
import numpy as np
from scipy.signal import convolve2d
from scipy.spatial import cKDTree
import math
from abc import ABC, abstractmethod


# ──────────────────────────────────────────────────────────────────────
# Terrain constants
# ──────────────────────────────────────────────────────────────────────

class Terrain(enum.IntEnum):
    PLAINS = 0
    WATER  = 1
    FOREST = 2


TERRAIN_SPEED = {
    Terrain.PLAINS: 1.0,
    Terrain.WATER:  0.3,
    Terrain.FOREST: 0.7,
}

FOREST_DETECTION_RANGE = 2.0
HEIGHT_RANGE_BONUS_PER_LEVEL = 1


# ──────────────────────────────────────────────────────────────────────
# Unit hierarchy
# ──────────────────────────────────────────────────────────────────────

class Unit(ABC):
    def __init__(self):
        self.hp: int = 0
        self.x: float = 0.0
        self.y: float = 0.0
        self.damage: int = 0
        self.range: int = 0
        self.speed: float = 0.0
        self.radius: float = 0.5
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

    def set_destination(self, x: float, y: float):
        self.destination = (x, y)

    def get_destination(self) -> tuple[float, float] | None:
        return self.destination

    # ── movement ──────────────────────────────────────────────────────

    def move(self, destination: tuple[float, float], tick_rate: int,
             speed_modifier: float = 1.0):
        dx = destination[0] - self.x
        dy = destination[1] - self.y
        dist = math.hypot(dx, dy)
        if dist > 0:
            step = (self.speed * speed_modifier) / tick_rate
            ratio = min(step / dist, 1.0)
            self.x += dx * ratio
            self.y += dy * ratio

    def on_tick(self, tick_rate: int, speed_modifier: float = 1.0):
        if self.destination is not None:
            self.move(self.destination, tick_rate, speed_modifier)
            if math.hypot(self.destination[0] - self.x,
                          self.destination[1] - self.y) < 0.01:
                self.destination = None

    # ── combat ────────────────────────────────────────────────────────

    def attack(self, target: Unit):
        target.set_hp(target.get_hp() - self.damage)


class Infantry(Unit):
    def __init__(self):
        super().__init__()
        self.hp = 100
        self.damage = 10
        self.range = 1
        self.speed = 1.0
        self.radius = 0.5


class Archer(Unit):
    def __init__(self):
        super().__init__()
        self.hp = 80
        self.damage = 15
        self.range = 3
        self.speed = 1.2
        self.radius = 0.5


class Cavalry(Unit):
    def __init__(self):
        super().__init__()
        self.hp = 120
        self.damage = 7
        self.range = 1
        self.speed = 3.0
        self.radius = 0.7


# ──────────────────────────────────────────────────────────────────────
# Formation
# ──────────────────────────────────────────────────────────────────────

class Formation:
    def __init__(self, units: list[Unit]):
        self.units = units
        self.portx: float = 0.0
        self.porty: float = 0.0
        self.starboardx: float = 0.0
        self.starboardy: float = 0.0

    def place_units(self):
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
        dx = starboardx - portx
        dy = starboardy - porty
        n = len(self.units)
        for i, unit in enumerate(self.units):
            t = i / (n - 1) if n > 1 else 0.5
            target_x = portx + t * dx
            target_y = porty + t * dy
            unit.move((target_x, target_y), tick_rate)

    def on_tick(self, tick_rate: int):
        for unit in self.units:
            unit.on_tick(tick_rate)

    def get_units(self) -> list[Unit]:
        return self.units

    def get_living_units(self) -> list[Unit]:
        return [u for u in self.units if not u.is_dead()]

    def get_units_around_point(self, x: float, y: float,
                               radius: float) -> list[Unit]:
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


# ──────────────────────────────────────────────────────────────────────
# Team
# ──────────────────────────────────────────────────────────────────────

UNIT_TYPES = {
    "infantry": Infantry,
    "archer":   Archer,
    "cavalry":  Cavalry,
}


class Team:
    def __init__(self, team_id: bool):
        self.team_id = team_id
        self.formations: list[Formation] = []

    def add_formation(self, formation: Formation):
        self.formations.append(formation)

    def get_formations(self) -> list[Formation]:
        return self.formations

    def get_all_units(self) -> list[Unit]:
        return [u for f in self.formations for u in f.get_units()]

    def get_living_units(self) -> list[Unit]:
        return [u for f in self.formations for u in f.get_living_units()]

    def is_defeated(self) -> bool:
        return len(self.get_living_units()) == 0


# ──────────────────────────────────────────────────────────────────────
# World — terrain-aware with curriculum gating
# ──────────────────────────────────────────────────────────────────────

class World:

    def __init__(self, width: int, height: int, tick_rate: int,
                 curriculum_stage: int = 0):
        self.width = width
        self.height = height
        self.tick_rate = tick_rate
        self.curriculum_stage = curriculum_stage

        self.teams: list[Team] = []
        self.terrain_map = self._generate_terrain_map()
        self.height_map = self._generate_height_map()
        self.tiles = np.stack(
            [self.terrain_map, self.height_map,
             np.zeros((self.height, self.width), dtype=int)],
            axis=-1
        )

    # ==================================================================
    # Team management
    # ==================================================================

    def add_team(self, team: Team):
        self.teams.append(team)

    def remove_team(self, team: Team):
        self.teams.remove(team)

    def get_teams(self) -> list[Team]:
        return self.teams

    def get_all_units(self) -> list[Unit]:
        return [u for team in self.teams
                for f in team.get_formations()
                for u in f.get_units()]

    def get_living_units(self) -> list[Unit]:
        return [u for team in self.teams
                for f in team.get_formations()
                for u in f.get_living_units()]

    def get_all_formations(self) -> list[Formation]:
        return [f for team in self.teams
                for f in team.get_formations()]

    # ==================================================================
    # Terrain queries
    # ==================================================================

    def _clamp_grid(self, x: float, y: float) -> tuple[int, int]:
        gx = int(np.clip(round(x), 0, self.width - 1))
        gy = int(np.clip(round(y), 0, self.height - 1))
        return gx, gy

    def get_terrain_at(self, x: float, y: float) -> int:
        gx, gy = self._clamp_grid(x, y)
        return int(self.terrain_map[gy, gx])

    def get_height_at(self, x: float, y: float) -> int:
        gx, gy = self._clamp_grid(x, y)
        return int(self.height_map[gy, gx])

    # ==================================================================
    # Curriculum-gated mechanics
    # ==================================================================

    def get_speed_modifier(self, x: float, y: float) -> float:
        if self.curriculum_stage < 1:
            return 1.0
        terrain = self.get_terrain_at(x, y)
        return TERRAIN_SPEED.get(terrain, 1.0)

    def is_concealed(self, unit: Unit) -> bool:
        if self.curriculum_stage < 2:
            return False
        return self.get_terrain_at(unit.x, unit.y) == Terrain.FOREST

    def can_detect(self, observer: Unit, target: Unit) -> bool:
        if self.curriculum_stage < 2:
            return True
        if not self.is_concealed(target):
            return True
        return (math.hypot(target.x - observer.x,
                           target.y - observer.y) <= FOREST_DETECTION_RANGE)

    def get_effective_range(self, attacker: Unit) -> float:
        return float(attacker.range)

    def get_effective_range_against(self, attacker: Unit,
                                    target: Unit) -> float:
        base = float(attacker.range)
        if self.curriculum_stage < 3:
            return base
        h_att = self.get_height_at(attacker.x, attacker.y)
        h_tgt = self.get_height_at(target.x, target.y)
        delta = (h_att - h_tgt) * HEIGHT_RANGE_BONUS_PER_LEVEL
        return max(1.0, base + delta)

    def has_line_of_sight(self, a: Unit, b: Unit) -> bool:
        if self.curriculum_stage < 3:
            return True

        ax, ay = self._clamp_grid(a.x, a.y)
        bx, by = self._clamp_grid(b.x, b.y)

        h_a = self.height_map[ay, ax]
        h_b = self.height_map[by, bx]
        max_endpoint = max(h_a, h_b)

        for gx, gy in self._bresenham(ax, ay, bx, by):
            if (gx, gy) == (ax, ay) or (gx, gy) == (bx, by):
                continue
            if self.height_map[gy, gx] > max_endpoint:
                return False
        return True

    @staticmethod
    def _bresenham(x0: int, y0: int, x1: int, y1: int):
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        while True:
            yield (x0, y0)
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

    # ==================================================================
    # Visibility
    # ==================================================================

    def can_see(self, observer: Unit, target: Unit) -> bool:
        if target.is_dead():
            return False
        if not self.can_detect(observer, target):
            return False
        if not self.has_line_of_sight(observer, target):
            return False
        return True

    def get_visible_enemies(self, friendly: Team, enemy: Team) -> list[Unit]:
        friendly_living = friendly.get_living_units()
        enemy_living = enemy.get_living_units()

        if not friendly_living:
            return []

        visible = []
        for enemy_unit in enemy_living:
            for ally in friendly_living:
                if self.can_see(ally, enemy_unit):
                    visible.append(enemy_unit)
                    break
        return visible

    # ==================================================================
    # Single-pair targeting (kept for BattleEnv / external callers)
    # ==================================================================

    def in_range(self, attacker: Unit, target: Unit) -> bool:
        eff_range = self.get_effective_range_against(attacker, target)
        dist = math.hypot(target.x - attacker.x, target.y - attacker.y)
        return dist <= eff_range

    def can_target(self, attacker: Unit, target: Unit) -> bool:
        if target.is_dead():
            return False
        if target.get_team() == attacker.get_team():
            return False
        if not self.can_detect(attacker, target):
            return False
        if not self.has_line_of_sight(attacker, target):
            return False
        if not self.in_range(attacker, target):
            return False
        return True

    # ==================================================================
    # Formation-based combat
    # ==================================================================

    def _combat_phase(self, living: list[Unit]):
        """
        Formation-based combat using a KDTree for nearest-enemy lookup.

        Builds one KDTree per team from all living enemies, then queries
        all attackers at once — replacing the brute-force inner loop.
        """
        for team in self.teams:
            # Gather ALL living enemies into a single pool
            enemy_units = []
            for et in self.teams:
                if et.team_id == team.team_id:
                    continue
                for f in et.get_formations():
                    enemy_units.extend(f.get_living_units())

            if not enemy_units:
                continue

            # One tree for all enemies this team can target
            enemy_pos = np.array([[u.x, u.y] for u in enemy_units])
            enemy_tree = cKDTree(enemy_pos)

            # Gather all living attackers on this team
            attackers = []
            for f in team.get_formations():
                attackers.extend(f.get_living_units())

            if not attackers:
                continue

            attacker_pos = np.array([[u.x, u.y] for u in attackers])

            # Single vectorised call: nearest enemy for every attacker
            dists, indices = enemy_tree.query(attacker_pos)

            for i, attacker in enumerate(attackers):
                if attacker.is_dead():
                    continue

                if dists[i] > attacker.range:
                    continue

                candidate = enemy_units[indices[i]]
                if candidate.is_dead():
                    continue
                if self.curriculum_stage >= 2 and not self.can_detect(attacker, candidate):
                    continue
                if self.curriculum_stage >= 3 and not self.has_line_of_sight(attacker, candidate):
                    continue

                attacker.attack(candidate)

    # ==================================================================
    # Collision separation
    # ==================================================================

    def _resolve_collisions(self, living: list[Unit],
                            iterations: int = 3):
        """
        Push overlapping units apart over several iterations.

        Uses cKDTree to find all pairs within the maximum possible
        collision distance, then applies a separation vector to each
        overlapping pair — half the overlap to each unit.
        """
        if len(living) < 2:
            return

        max_radius = max(u.radius for u in living)
        query_radius = max_radius * 2

        for _ in range(iterations):
            positions = np.array([[u.x, u.y] for u in living])
            tree = cKDTree(positions)
            pairs = tree.query_pairs(query_radius)

            if not pairs:
                break  # no overlaps — nothing to resolve

            resolved_any = False
            for i, j in pairs:
                a = living[i]
                b = living[j]

                dx = b.x - a.x
                dy = b.y - a.y
                dist = math.hypot(dx, dy)

                min_dist = a.radius + b.radius
                if dist >= min_dist:
                    continue

                resolved_any = True
                overlap = min_dist - dist

                if dist > 0:
                    nx = dx / dist
                    ny = dy / dist
                else:
                    # Exactly overlapping — pick arbitrary direction
                    nx = 1.0
                    ny = 0.0

                push = overlap / 2

                a.x -= nx * push
                a.y -= ny * push
                b.x += nx * push
                b.y += ny * push

            # Clamp to world bounds
            for u in living:
                u.x = max(0, min(self.width - 1, u.x))
                u.y = max(0, min(self.height - 1, u.y))

            if not resolved_any:
                break  # pairs were in query range but none actually overlapped

    # ==================================================================
    # Tick
    # ==================================================================

    def tick(self):
        living = self.get_living_units()

        # ── movement phase ────────────────────────────────────────────
        if self.curriculum_stage < 1:
            for formation in self.get_all_formations():
                formation.on_tick(self.tick_rate)
        else:
            for unit in living:
                modifier = self.get_speed_modifier(unit.x, unit.y)
                unit.on_tick(self.tick_rate, modifier)

        # ── collision separation ──────────────────────────────────────
        self._resolve_collisions(living)

        # ── combat phase ──────────────────────────────────────────────
        self._combat_phase(living)

    # ==================================================================
    # Population
    # ==================================================================

    def populate_team(self, team: Team,
                      min_formations: int = 1, max_formations: int = 3,
                      min_units: int = 10, max_units: int = 50):
        num_formations = np.random.randint(min_formations, max_formations + 1)

        margin = 5
        available_height = self.height - 2 * margin
        slot_height = available_height / num_formations

        if not team.team_id:
            x_start, x_end = 10, 30
        else:
            x_start, x_end = self.width - 30, self.width - 10

        for i in range(num_formations):
            unit_type = np.random.choice(list(UNIT_TYPES.keys()))
            amount = np.random.randint(min_units, max_units + 1)

            UnitClass = UNIT_TYPES[unit_type]
            units = [UnitClass() for _ in range(amount)]

            for unit in units:
                unit.set_team(team.team_id)

            y_top = margin + i * slot_height
            y_bottom = margin + (i + 1) * slot_height

            formation = Formation(units)
            formation.set_port(x_start, y_top)
            formation.set_starboard(x_end, y_bottom)
            formation.place_units()

            team.add_formation(formation)

        self.add_team(team)

    # ==================================================================
    # Terrain generation
    # ==================================================================

    def _generate_perlin_noise(self, scale: float = 50.0, octaves: int = 6,
                               persistence: float = 0.5,
                               lacunarity: float = 2.0) -> np.ndarray:
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
        top    = top_left * (1 - sx_2d) + top_right * sx_2d
        bottom = bottom_left * (1 - sx_2d) + bottom_right * sx_2d
        return top * (1 - sy_2d) + bottom * sy_2d

    def _apply_cellular_automata(self, grid: np.ndarray,
                                 iterations: int = 3) -> np.ndarray:
        smoothed = np.copy(grid)
        kernel = np.ones((3, 3), dtype=int)
        for _ in range(iterations):
            unique_vals = np.unique(smoothed)
            counts = np.zeros((len(unique_vals), self.height, self.width))
            for i, val in enumerate(unique_vals):
                mask = (smoothed == val).astype(int)
                counts[i] = convolve2d(mask, kernel, mode='same',
                                       boundary='fill')
            winner_indices = np.argmax(counts, axis=0)
            smoothed = unique_vals[winner_indices]
        return smoothed

    def _generate_terrain_map(self) -> np.ndarray:
        noise = self._generate_perlin_noise(scale=30.0, octaves=6,
                                            persistence=0.5)
        terrain_map = np.zeros((self.height, self.width), dtype=int)
        terrain_map[noise < 0.35] = Terrain.WATER
        terrain_map[(noise >= 0.35) & (noise < 0.65)] = Terrain.PLAINS
        terrain_map[noise >= 0.65] = Terrain.FOREST
        return self._apply_cellular_automata(terrain_map, iterations=5)

    def _generate_height_map(self) -> np.ndarray:
        noise = self._generate_perlin_noise(scale=25.0, octaves=4,
                                            persistence=0.6)
        height_map = np.clip((noise * 5).astype(int), 0, 5)
        height_map = self._apply_cellular_automata(height_map, iterations=3)
        height_map[self.terrain_map == Terrain.WATER] = 0
        return height_map