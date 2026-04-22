from __future__ import annotations

from asyncio.windows_events import NULL
from asyncio.windows_events import NULL
import enum
import numpy as np
from scipy.signal import convolve2d
from scipy.spatial import cKDTree
import math
from abc import ABC, abstractmethod

ARMY_PRESETS = [
    # ── Balanced compositions ──────────────────────────────────
    {
        "name": "Balanced",
        "formations": [
            {"type": "infantry", "count": 30},
            {"type": "archer",   "count": 20},
            {"type": "cavalry",  "count": 15},
        ],
    },
    {
        "name": "Infantry core",
        "formations": [
            {"type": "infantry", "count": 40},
            {"type": "infantry", "count": 25},
            {"type": "archer",   "count": 15},
        ],
    },
    # ── Ranged heavy ───────────────────────────────────────────
    {
        "name": "Archer line",
        "formations": [
            {"type": "archer",   "count": 35},
            {"type": "archer",   "count": 25},
            {"type": "infantry", "count": 15},
        ],
    },
    {
        "name": "Ranged screen",
        "formations": [
            {"type": "infantry", "count": 15},
            {"type": "archer",   "count": 40},
        ],
    },
    # ── Cavalry focused ────────────────────────────────────────
    {
        "name": "Cavalry hammer",
        "formations": [
            {"type": "infantry", "count": 25},
            {"type": "cavalry",  "count": 30},
        ],
    },
    {
        "name": "Double envelopment",
        "formations": [
            {"type": "cavalry",  "count": 20},
            {"type": "infantry", "count": 30},
            {"type": "cavalry",  "count": 20},
        ],
    },
    # ── Extremes ───────────────────────────────────────────────
    {
        "name": "Horde",
        "formations": [
            {"type": "infantry", "count": 50},
            {"type": "infantry", "count": 50},
        ],
    },
    {
        "name": "Elite cavalry",
        "formations": [
            {"type": "cavalry",  "count": 25},
            {"type": "cavalry",  "count": 25},
        ],
    },
    {
        "name": "Combined arms",
        "formations": [
            {"type": "cavalry",  "count": 15},
            {"type": "infantry", "count": 30},
            {"type": "archer",   "count": 25},
            {"type": "cavalry",  "count": 15},
        ],
    },
    {
        "name": "Skirmish line",
        "formations": [
            {"type": "archer",   "count": 10},
            {"type": "archer",   "count": 10},
            {"type": "archer",   "count": 10},
            {"type": "archer",   "count": 10},
        ],
    },
]

DEFAULT_RANK_WIDTH = 10
DEFAULT_RANK_SPACING = 1.5

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
SIGHT_RANGE = 50.0


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
        self.is_attacking: bool = False

    @property
    def can_move_while_attacking(self) -> bool:
        """Override in subclasses that should move during combat."""
        return False

    def attack(self, target: 'Unit'):
        target.set_hp(target.get_hp() - self.damage)
        self.is_attacking = True

    def set_destination(self, x, y):
        self.destination[0] = x
        self.destination[1] = y

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

    def move(self, destination: tuple[float, float], tick_rate: int, speed_modifier: float = 1.0):
        dx = destination[0] - self.x
        dy = destination[1] - self.y
        dist = math.hypot(dx, dy)
        if dist > 0:
            step = (self.speed * speed_modifier) / tick_rate
            ratio = min(step / dist, 1.0)
            self.x += dx * ratio
            self.y += dy * ratio


    def on_tick(self, tick_rate: int, speed_modifier: float = 1.0):
        if self.is_attacking and not self.can_move_while_attacking:
            return
        if self.destination is not None:
            self.move(self.destination, tick_rate, speed_modifier)
            if math.hypot(self.destination[0] - self.x,
                          self.destination[1] - self.y) < 0.01:
                self.destination = None

class Infantry(Unit):
    def __init__(self):
        super().__init__()
        self.hp = 100
        self.damage = 10
        self.range = 1
        self.speed = 3.0
        self.radius = 0.5


class Archer(Unit):
    def __init__(self):
        super().__init__()
        self.hp = 80
        self.damage = 5
        self.range = 50
        self.speed = 3.6
        self.radius = 0.5


class Cavalry(Unit):
    def __init__(self):
        super().__init__()
        self.hp = 120
        self.damage = 7
        self.range = 1
        self.speed = 9.0
        self.radius = 0.7

    @property
    def can_move_while_attacking(self) -> bool:
        return True

# ──────────────────────────────────────────────────────────────────────
# Formation
# ──────────────────────────────────────────────────────────────────────

class Formation:
    def __init__(self, units: list):  # list[Unit]
        self.units = units
        self.initial_amount = len(units)
        self.portx: float = 0.0
        self.porty: float = 0.0
        self.starboardx: float = 0.0
        self.starboardy: float = 0.0

    def _rank_positions(
            self,
            portx: float, porty: float,
            starboardx: float, starboardy: float,
            n: int,
            rank_width: int = DEFAULT_RANK_WIDTH,
            rank_spacing: float = DEFAULT_RANK_SPACING,
            facing_right: bool = True,
        ) -> list[tuple[float, float]]:
        """
        Compute (x, y) positions for *n* units in a ranked formation.

        The front rank lies along the port → starboard line.
        Subsequent ranks are placed *behind* the line (away from
        the enemy), determined by `facing_right`.

        Parameters
        ----------
        portx, porty, starboardx, starboardy
            Endpoints of the front-rank line.
        n
            Number of units to place.
        rank_width
            Maximum units per rank (row width).
        rank_spacing
            World-unit gap between successive ranks.
        facing_right
            True  → enemy is to the right,  depth extends left.
            False → enemy is to the left,   depth extends right.
        """
        if n == 0:
            return []

        dx = starboardx - portx
        dy = starboardy - porty
        line_len = math.hypot(dx, dy)

        if line_len < 1e-9:
            return [(portx, porty)] * n

        # Unit vectors: along the front line
        along_x = dx / line_len
        along_y = dy / line_len

        # Two perpendicular candidates
        perp_a = (-along_y, along_x)
        perp_b = (along_y, -along_x)

        # Pick the one pointing away from the enemy
        if facing_right:
            # "Behind" is toward negative-x (our spawn side)
            depth_dir = perp_a if perp_a[0] <= 0 else perp_b
        else:
            # "Behind" is toward positive-x
            depth_dir = perp_a if perp_a[0] >= 0 else perp_b

        positions: list[tuple[float, float]] = []
        for idx in range(n):
            rank = idx // rank_width        # row index (0 = front)
            file_idx = idx % rank_width     # position within row

            units_in_this_rank = min(rank_width, n - rank * rank_width)

            # Interpolate along the front line
            if units_in_this_rank > 1:
                t = file_idx / (units_in_this_rank - 1)
            else:
                t = 0.5

            front_x = portx + t * dx
            front_y = porty + t * dy

            # Offset backward by rank depth
            x = front_x + rank * rank_spacing * depth_dir[0]
            y = front_y + rank * rank_spacing * depth_dir[1]

            positions.append((x, y))

        return positions
    
        # ── CHANGED: place_units now uses ranks ───────────────────────
    def place_units(
            self,
            rank_width: int = DEFAULT_RANK_WIDTH,
            rank_spacing: float = DEFAULT_RANK_SPACING,
            facing_right: bool = True,
        ):
        """Snap all units into rank formation at current port/starboard."""
        positions = self._rank_positions(
            self.portx, self.porty,
            self.starboardx, self.starboardy,
            len(self.units),
            rank_width, rank_spacing, facing_right,
        )
        for unit, (x, y) in zip(self.units, positions):
            unit.set_x(x)
            unit.set_y(y)
    
        # ── CHANGED: move now uses ranks and updates port/starboard ───
    def move(self, portx: float, porty: float, starboardx: float, starboardy: float,
            rank_width: int = DEFAULT_RANK_WIDTH,
            rank_spacing: float = DEFAULT_RANK_SPACING,
            facing_right: bool = True,
        ):
        """
        Set unit destinations along a new port/starboard line,
        arranged in ranks.

        Also updates the formation's stored port/starboard so that
        cohesion measurement stays consistent with the target.
        """
        self.portx = portx
        self.porty = porty
        self.starboardx = starboardx
        self.starboardy = starboardy

        living = self.get_living_units()
        positions = self._rank_positions(
            portx, porty, starboardx, starboardy,
            len(living),
            rank_width, rank_spacing, facing_right,
        )
        for unit, (x, y) in zip(living, positions):
            unit.set_destination(x, y)

    def on_tick(self, tick_rate: int, speed_modifier: float = 1.0):
        """Move all units one step toward their destinations."""
        for unit in self.units:
            unit.on_tick(tick_rate, speed_modifier)

    def get_units(self) -> list:
        return self.units

    def get_living_units(self) -> list:
        return [u for u in self.units if not u.is_dead()]

    def get_initial_amount(self) -> int:
        return self.initial_amount

    def set_port(self, x: float, y: float):
        self.portx = x
        self.porty = y

    def set_starboard(self, x: float, y: float):
        self.starboardx = x
        self.starboardy = y

    def get_port(self) -> tuple:
        return (self.portx, self.porty)

    def get_starboard(self) -> tuple:
        return (self.starboardx, self.starboardy)

    def add_unit(self, unit, x: float, y: float):
        unit.set_x(x)
        unit.set_y(y)
        self.units.append(unit)

    def remove_unit(self, unit):
        self.units.remove(unit)

    def get_units_around_point(self, x: float, y: float,
                               radius: float) -> list:
        import math
        return [u for u in self.units
                if math.hypot(u.get_x() - x, u.get_y() - y) <= radius]
    
    def measure_formation_cohesion(self) -> float:
        """
        How close a formation's living units are to a perfect line.

        Returns float in [0, 1].  1 = perfect line, 0 = total disorder.

        Two components, equally weighted:
        - perp_score:    mean perpendicular distance from the line
        - spacing_score: how evenly units are distributed along it

        Sorted projection comparison makes this robust to unit deaths.
        """
        living = self.get_living_units()
        n = len(living)
        if n <= 1:
            return 1.0

        px, py = self.get_port()
        sx, sy = self.get_starboard()
        ldx = sx - px
        ldy = sy - py
        line_len = math.hypot(ldx, ldy)

        if line_len < 1e-9:
            return 0.5

        # Unit vectors: along line and normal
        along_x = ldx / line_len
        along_y = ldy / line_len
        norm_x = -along_y
        norm_y =  along_x

        perp_distances = []
        projections = []

        for u in living:
            dx = u.x - px
            dy = u.y - py
            t = (dx * along_x + dy * along_y) / line_len   # [0,1] ideally
            perp = abs(dx * norm_x + dy * norm_y)
            projections.append(t)
            perp_distances.append(perp)

        # ── Perpendicular score ───────────────────────────────────
        mean_perp = sum(perp_distances) / n
        # Normalise against half the line length — if mean drift equals
        # half the formation width, cohesion is essentially gone.
        perp_score = max(0.0, 1.0 - mean_perp / (line_len * 0.5))

        # ── Spacing score ─────────────────────────────────────────
        projections.sort()
        ideal = [i / (n - 1) for i in range(n)]
        spacing_error = sum(abs(projections[i] - ideal[i]) for i in range(n)) / n
        # An average error of 0.5 (half the line) ⇒ score ~ 0
        spacing_score = max(0.0, 1.0 - spacing_error * 2)

        return (perp_score + spacing_score) / 2

    def get_diagonal_length(self):
        return math.hypot(self.starboardx - self.portx, self.starboardy - self.porty)

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
        self.tick_count = 0
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
        # Clear last tick's attack flags — attack() will re-set
        # them for units that attack this tick
        for u in living:
            u.is_attacking = False

        for team in self.teams:
            enemy_units = []
            for et in self.teams:
                if et.team_id == team.team_id:
                    continue
                for f in et.get_formations():
                    enemy_units.extend(f.get_living_units())

            if not enemy_units:
                continue

            enemy_pos = np.array([[u.x, u.y] for u in enemy_units])
            enemy_tree = cKDTree(enemy_pos)

            attackers = []
            for f in team.get_formations():
                attackers.extend(f.get_living_units())

            if not attackers:
                continue

            attacker_pos = np.array([[u.x, u.y] for u in attackers])
            dists, indices = enemy_tree.query(attacker_pos)

            for i, attacker in enumerate(attackers):
                if attacker.is_dead():
                    continue

                target = enemy_units[indices[i]]
                if target.is_dead():
                    continue

                dist = dists[i]
                eff_range = self.get_effective_range_against(attacker, target)

                if dist <= eff_range and self.can_target(attacker, target):
                    attacker.attack(target)
                elif dist <= SIGHT_RANGE and self.can_see(attacker, target):
                    attacker.set_destination(target.x, target.y)

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
        for unit in living:
            speed_mod = self.get_speed_modifier(unit.x, unit.y)
            unit.on_tick(self.tick_rate, speed_mod)

        # ── collision separation ──────────────────────────────────────
        self._resolve_collisions(living)

        # ── combat phase ──────────────────────────────────────────────
        self._combat_phase(living)

        self.increment_tick()

    def get_tick_count(self) -> int:
        return self.tick_count
    
    def increment_tick(self):
        self.tick_count += 1

    # ==================================================================
    # Population
    # ==================================================================

    def populate_team(self, team: Team):
        """Pick a random preset and place its formations."""
        preset = ARMY_PRESETS[np.random.randint(len(ARMY_PRESETS))]

        num_formations = len(preset["formations"])
        margin = 5
        available_height = self.height - 2 * margin
        slot_height = available_height / num_formations

        if not team.team_id:
            x_start, x_end = 80, 100
        else:
            x_start, x_end = self.width - 100, self.width - 80

        for i, spec in enumerate(preset["formations"]):
            UnitClass = UNIT_TYPES[spec["type"]]
            units = [UnitClass() for _ in range(spec["count"])]

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
    
    # ==================================================================
    # Small uttility methods
    # ==================================================================

    def get_width(self) -> int:
        return self.width
    
    def get_height(self) -> int:
        return self.height
    
    def get_diagonal_length(self) -> float:
        return math.hypot(self.width, self.height)