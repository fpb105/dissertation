"""
Test suite for the battle simulator.

Covers: unit mechanics, formation behaviour, team aggregation,
terrain generation, combat targeting, collision resolution,
line of sight, cohesion measurement, and visibility.

Run with: python -m pytest test_model.py -v
"""

import math
import numpy as np
import pytest
import model


# ══════════════════════════════════════════════════════════════════════
# 7.1  Unit Tests — Unit Types and Stats
# ══════════════════════════════════════════════════════════════════════

class TestUnitStats:
    """Verify each unit type is initialised with correct stats."""

    def test_infantry_stats(self):
        u = model.Infantry()
        assert u.hp == 100
        assert u.damage == 10
        assert u.speed == 1
        assert u.radius == 1.0
        assert u.range == 4  # (1.0 * 2) + 2

    def test_archer_stats(self):
        u = model.Archer()
        assert u.hp == 80
        assert u.damage == 5
        assert u.speed == 1.2
        assert u.radius == 0.5
        assert u.range == 51  # (0.5 * 2) + 50

    def test_cavalry_stats(self):
        u = model.Cavalry()
        assert u.hp == 80
        assert u.damage == 7
        assert u.speed == 3
        assert u.radius == 2.0
        assert u.range == 5  # (2.0 * 2) + 1


class TestUnitDeath:
    """Verify unit death detection at boundary values."""

    def test_unit_dies_at_zero_hp(self):
        u = model.Infantry()
        u.set_hp(0)
        assert u.is_dead()

    def test_unit_dies_below_zero_hp(self):
        u = model.Infantry()
        u.set_hp(-10)
        assert u.is_dead()

    def test_unit_alive_at_one_hp(self):
        u = model.Infantry()
        u.set_hp(1)
        assert not u.is_dead()


class TestUnitMovement:
    """Verify units move toward destinations correctly."""

    def test_unit_moves_toward_destination(self):
        u = model.Infantry()
        u.set_x(0.0)
        u.set_y(0.0)
        u.set_destination(100.0, 0.0)
        u.on_tick(tick_rate=1, speed_modifier=1.0)
        assert u.get_x() > 0.0, "Unit should have moved in +x direction"

    def test_unit_does_not_overshoot(self):
        u = model.Infantry()
        u.set_x(0.0)
        u.set_y(0.0)
        u.set_destination(0.5, 0.0)
        u.on_tick(tick_rate=1, speed_modifier=1.0)
        assert u.get_x() <= 0.5, "Unit should not overshoot destination"

    def test_cavalry_faster_than_infantry(self):
        inf = model.Infantry()
        inf.set_x(0.0); inf.set_y(0.0)
        inf.set_destination(100.0, 0.0)

        cav = model.Cavalry()
        cav.set_x(0.0); cav.set_y(0.0)
        cav.set_destination(100.0, 0.0)

        inf.on_tick(tick_rate=10)
        cav.on_tick(tick_rate=10)

        assert cav.get_x() > inf.get_x(), "Cavalry should move faster"

    def test_attacking_unit_stops_moving(self):
        u = model.Infantry()
        u.set_x(0.0); u.set_y(0.0)
        u.set_destination(100.0, 0.0)
        u.is_attacking = True
        u.on_tick(tick_rate=10)
        assert u.get_x() == 0.0, "Attacking infantry should not move"

    def test_cavalry_moves_while_attacking(self):
        u = model.Cavalry()
        u.set_x(0.0); u.set_y(0.0)
        u.set_destination(100.0, 0.0)
        u.is_attacking = True
        u.on_tick(tick_rate=10)
        assert u.get_x() > 0.0, "Cavalry should move even while attacking"


# ══════════════════════════════════════════════════════════════════════
# 7.2  Unit Tests — Shield Wall Mechanics
# ══════════════════════════════════════════════════════════════════════

class TestShieldWall:
    """Verify shield wall modifies damage and movement."""

    def test_shield_wall_halves_outgoing_damage(self):
        attacker = model.Infantry()
        attacker.set_wall(True)
        target = model.Infantry()
        initial_hp = target.get_hp()
        attacker.attack(target)
        damage_dealt = initial_hp - target.get_hp()
        assert damage_dealt == 5, f"Shield wall should halve damage (10→5), got {damage_dealt}"

    def test_no_wall_full_damage(self):
        attacker = model.Infantry()
        attacker.set_wall(False)
        target = model.Infantry()
        initial_hp = target.get_hp()
        attacker.attack(target)
        damage_dealt = initial_hp - target.get_hp()
        assert damage_dealt == 10

    def test_archer_vs_shield_wall_95_percent_reduction(self):
        archer = model.Archer()
        target = model.Infantry()
        target.set_wall(True)
        initial_hp = target.get_hp()
        archer.attack(target)
        damage_dealt = initial_hp - target.get_hp()
        expected = 5 * 0.05  # 0.25
        assert abs(damage_dealt - expected) < 0.01, \
            f"Archer vs shield wall should deal {expected}, got {damage_dealt}"

    def test_archer_vs_no_wall_full_damage(self):
        archer = model.Archer()
        target = model.Infantry()
        target.set_wall(False)
        initial_hp = target.get_hp()
        archer.attack(target)
        damage_dealt = initial_hp - target.get_hp()
        assert damage_dealt == 5

    def test_archer_vs_non_infantry_full_damage(self):
        archer = model.Archer()
        target = model.Cavalry()
        initial_hp = target.get_hp()
        archer.attack(target)
        damage_dealt = initial_hp - target.get_hp()
        assert damage_dealt == 5

    def test_shield_wall_halves_movement_speed(self):
        u1 = model.Infantry()
        u1.set_x(0.0); u1.set_y(0.0)
        u1.set_destination(100.0, 0.0)
        u1.set_wall(False)

        u2 = model.Infantry()
        u2.set_x(0.0); u2.set_y(0.0)
        u2.set_destination(100.0, 0.0)
        u2.set_wall(True)

        u1.on_tick(tick_rate=10)
        u2.on_tick(tick_rate=10)

        assert abs(u2.get_x() - u1.get_x() / 2) < 0.01, \
            "Shield wall infantry should move at half speed"


# ══════════════════════════════════════════════════════════════════════
# 7.3  Unit Tests — Formation
# ══════════════════════════════════════════════════════════════════════

class TestFormation:
    """Verify formation destination assignment and structure."""

    def test_units_receive_destinations(self):
        units = [model.Infantry() for _ in range(10)]
        f = model.Formation(units)
        f.move(0, 0, 50, 0)
        for u in units:
            assert u.get_destination() is not None, \
                "Every unit should have a destination after formation.move()"

    def test_destinations_within_bounds(self):
        units = [model.Infantry() for _ in range(30)]
        f = model.Formation(units)
        f.move(10, 10, 90, 10)
        for u in units:
            dx, dy = u.get_destination()
            assert dx >= 0, f"Destination x={dx} should not be negative"
            assert dy >= 0, f"Destination y={dy} should not be negative"

    def test_rank_count_increases_with_depth(self):
        units = [model.Infantry() for _ in range(30)]
        f = model.Formation(units)
        f.move(0, 0, 10, 0)  # short line — forces multiple ranks
        assert f.get_rank_count() > 1, "Narrow line should produce multiple ranks"

    def test_single_unit_formation(self):
        units = [model.Infantry()]
        f = model.Formation(units)
        f.move(0, 0, 50, 0)
        assert units[0].get_destination() is not None

    def test_empty_formation_no_crash(self):
        f = model.Formation([])
        f.move(0, 0, 50, 0)  # should not raise

    def test_living_units_excludes_dead(self):
        units = [model.Infantry() for _ in range(5)]
        units[0].set_hp(0)
        units[2].set_hp(0)
        f = model.Formation(units)
        assert len(f.get_living_units()) == 3
        assert len(f.get_units()) == 5


# ══════════════════════════════════════════════════════════════════════
# 7.4  Unit Tests — Team
# ══════════════════════════════════════════════════════════════════════

class TestTeam:
    """Verify team aggregation and defeat detection."""

    def test_team_defeated_when_all_dead(self):
        team = model.Team(team_id=False)
        units = [model.Infantry() for _ in range(5)]
        for u in units:
            u.set_hp(0)
        team.add_formation(model.Formation(units))
        assert team.is_defeated()

    def test_team_not_defeated_with_living(self):
        team = model.Team(team_id=False)
        units = [model.Infantry() for _ in range(5)]
        units[0].set_hp(0)
        team.add_formation(model.Formation(units))
        assert not team.is_defeated()

    def test_get_living_units_across_formations(self):
        team = model.Team(team_id=False)
        f1 = model.Formation([model.Infantry() for _ in range(3)])
        f2 = model.Formation([model.Archer() for _ in range(4)])
        f1.get_units()[0].set_hp(0)
        team.add_formation(f1)
        team.add_formation(f2)
        assert len(team.get_living_units()) == 6  # 2 + 4


# ══════════════════════════════════════════════════════════════════════
# 7.5  Unit Tests — Terrain Generation
# ══════════════════════════════════════════════════════════════════════

class TestTerrainGeneration:
    """Verify terrain maps are valid and within expected bounds."""

    @pytest.fixture
    def world(self):
        return model.World(100, 100, tick_rate=10, curriculum_stage=3)

    def test_terrain_map_shape(self, world):
        assert world.terrain_map.shape == (100, 100)

    def test_height_map_shape(self, world):
        assert world.height_map.shape == (100, 100)

    def test_terrain_values_valid(self, world):
        unique = set(np.unique(world.terrain_map))
        valid = {model.Terrain.PLAINS, model.Terrain.WATER, model.Terrain.FOREST}
        assert unique.issubset(valid), f"Unexpected terrain values: {unique - valid}"

    def test_height_values_in_range(self, world):
        assert world.height_map.min() >= 0
        assert world.height_map.max() <= 5

    def test_water_tiles_height_zero(self, world):
        water_mask = world.terrain_map == model.Terrain.WATER
        if water_mask.any():
            water_heights = world.height_map[water_mask]
            assert np.all(water_heights == 0), \
                "All water tiles should have height 0"

    def test_terrain_deterministic_with_seed(self):
        np.random.seed(42)
        w1 = model.World(50, 50, tick_rate=10)
        np.random.seed(42)
        w2 = model.World(50, 50, tick_rate=10)
        assert np.array_equal(w1.terrain_map, w2.terrain_map)
        assert np.array_equal(w1.height_map, w2.height_map)


# ══════════════════════════════════════════════════════════════════════
# 7.6  Unit Tests — Speed Modifiers (Curriculum Gating)
# ══════════════════════════════════════════════════════════════════════

class TestSpeedModifiers:
    """Verify terrain speed modifiers respect curriculum stage."""

    def test_stage0_always_returns_1(self):
        world = model.World(50, 50, tick_rate=10, curriculum_stage=0)
        # Regardless of terrain type, stage 0 should return 1.0
        for y in range(50):
            for x in range(50):
                assert world.get_speed_modifier(x, y) == 1.0

    def test_stage1_water_slows(self):
        world = model.World(100, 100, tick_rate=10, curriculum_stage=1)
        # Find a water tile
        water_positions = np.argwhere(world.terrain_map == model.Terrain.WATER)
        if len(water_positions) > 0:
            wy, wx = water_positions[0]
            assert world.get_speed_modifier(wx, wy) == 0.3

    def test_stage1_plains_normal(self):
        world = model.World(100, 100, tick_rate=10, curriculum_stage=1)
        plains_positions = np.argwhere(world.terrain_map == model.Terrain.PLAINS)
        if len(plains_positions) > 0:
            py, px = plains_positions[0]
            assert world.get_speed_modifier(px, py) == 1.0


# ══════════════════════════════════════════════════════════════════════
# 7.7  Unit Tests — Concealment and Detection
# ══════════════════════════════════════════════════════════════════════

class TestConcealment:
    """Verify forest concealment and detection range."""

    def test_stage0_no_concealment(self):
        world = model.World(100, 100, tick_rate=10, curriculum_stage=0)
        u = model.Infantry()
        # Even if on forest, stage 0 means no concealment
        assert not world.is_concealed(u)

    def test_stage2_forest_conceals(self):
        world = model.World(100, 100, tick_rate=10, curriculum_stage=2)
        forest_positions = np.argwhere(world.terrain_map == model.Terrain.FOREST)
        if len(forest_positions) > 0:
            fy, fx = forest_positions[0]
            u = model.Infantry()
            u.set_x(float(fx)); u.set_y(float(fy))
            assert world.is_concealed(u)

    def test_concealed_unit_detected_within_range(self):
        world = model.World(100, 100, tick_rate=10, curriculum_stage=2)
        forest_positions = np.argwhere(world.terrain_map == model.Terrain.FOREST)
        if len(forest_positions) > 0:
            fy, fx = forest_positions[0]
            target = model.Infantry()
            target.set_x(float(fx)); target.set_y(float(fy))

            observer = model.Infantry()
            observer.set_x(float(fx) + 10); observer.set_y(float(fy))
            # 10 units away, within FOREST_DETECTION_RANGE (50)
            assert world.can_detect(observer, target)

    def test_concealed_unit_not_detected_outside_range(self):
        world = model.World(200, 200, tick_rate=10, curriculum_stage=2)
        forest_positions = np.argwhere(world.terrain_map == model.Terrain.FOREST)
        if len(forest_positions) > 0:
            fy, fx = forest_positions[0]
            target = model.Infantry()
            target.set_x(float(fx)); target.set_y(float(fy))

            observer = model.Infantry()
            observer.set_x(float(fx) + 60); observer.set_y(float(fy))
            # 60 units away, outside FOREST_DETECTION_RANGE (50)
            assert not world.can_detect(observer, target)


# ══════════════════════════════════════════════════════════════════════
# 7.8  Unit Tests — Line of Sight
# ══════════════════════════════════════════════════════════════════════

class TestLineOfSight:
    """Verify Bresenham-based line of sight checks."""

    def test_los_always_true_below_stage3(self):
        world = model.World(50, 50, tick_rate=10, curriculum_stage=2)
        a = model.Infantry(); a.set_x(0); a.set_y(0)
        b = model.Infantry(); b.set_x(49); b.set_y(49)
        assert world.has_line_of_sight(a, b)

    def test_los_blocked_by_peak(self):
        world = model.World(50, 50, tick_rate=10, curriculum_stage=3)
        # Force a peak between two units
        world.height_map[:] = 1
        world.height_map[25, 25] = 5  # tall peak in the middle

        a = model.Infantry(); a.set_x(0); a.set_y(25)
        b = model.Infantry(); b.set_x(49); b.set_y(25)
        assert not world.has_line_of_sight(a, b), \
            "Peak of height 5 should block LOS between height-1 units"

    def test_los_not_blocked_by_equal_height(self):
        world = model.World(50, 50, tick_rate=10, curriculum_stage=3)
        world.height_map[:] = 3
        a = model.Infantry(); a.set_x(0); a.set_y(25)
        b = model.Infantry(); b.set_x(49); b.set_y(25)
        assert world.has_line_of_sight(a, b), \
            "Flat terrain should not block LOS"

    def test_los_high_ground_sees_over_low_peak(self):
        world = model.World(50, 50, tick_rate=10, curriculum_stage=3)
        world.height_map[:] = 1
        world.height_map[25, 25] = 3  # mid peak
        world.height_map[25, 0] = 4   # observer on high ground
        world.height_map[25, 49] = 4  # target on high ground

        a = model.Infantry(); a.set_x(0); a.set_y(25)
        b = model.Infantry(); b.set_x(49); b.set_y(25)
        assert world.has_line_of_sight(a, b), \
            "High ground units should see over lower peaks"


# ══════════════════════════════════════════════════════════════════════
# 7.9  Unit Tests — Height Range Bonus
# ══════════════════════════════════════════════════════════════════════

class TestHeightRangeBonus:
    """Verify height affects effective range at stage 3."""

    def test_higher_ground_increases_range(self):
        world = model.World(50, 50, tick_rate=10, curriculum_stage=3)
        world.height_map[:] = 1
        world.height_map[10, 10] = 5  # attacker on high ground

        attacker = model.Archer()
        attacker.set_x(10); attacker.set_y(10)
        target = model.Infantry()
        target.set_x(10); target.set_y(20)

        base_range = float(attacker.range)
        eff_range = world.get_effective_range_against(attacker, target)
        assert eff_range == base_range + 4, \
            "4 levels higher should add 4 range"

    def test_lower_ground_decreases_range(self):
        world = model.World(50, 50, tick_rate=10, curriculum_stage=3)
        world.height_map[:] = 5
        world.height_map[10, 10] = 1  # attacker on low ground

        attacker = model.Archer()
        attacker.set_x(10); attacker.set_y(10)
        target = model.Infantry()
        target.set_x(10); target.set_y(20)

        base_range = float(attacker.range)
        eff_range = world.get_effective_range_against(attacker, target)
        assert eff_range == base_range - 4

    def test_no_range_bonus_before_stage3(self):
        world = model.World(50, 50, tick_rate=10, curriculum_stage=2)
        world.height_map[:] = 1
        world.height_map[10, 10] = 5

        attacker = model.Archer()
        attacker.set_x(10); attacker.set_y(10)
        target = model.Infantry()
        target.set_x(10); target.set_y(20)

        assert world.get_effective_range_against(attacker, target) == float(attacker.range)


# ══════════════════════════════════════════════════════════════════════
# 7.10  Unit Tests — Combat Targeting
# ══════════════════════════════════════════════════════════════════════

class TestCombat:
    """Verify combat phase resolves attacks correctly."""

    def test_units_in_range_deal_damage(self):
        world = model.World(100, 100, tick_rate=10, curriculum_stage=0)
        team_a = model.Team(team_id=False)
        team_b = model.Team(team_id=True)

        attacker = model.Infantry()
        attacker.set_team(False); attacker.set_x(50); attacker.set_y(50)

        target = model.Infantry()
        target.set_team(True); target.set_x(52); target.set_y(50)
        # Distance = 2, infantry range = 4 — should hit

        team_a.add_formation(model.Formation([attacker]))
        team_b.add_formation(model.Formation([target]))
        world.add_team(team_a)
        world.add_team(team_b)

        initial_hp = target.get_hp()
        world._combat_phase(world.get_living_units())
        assert target.get_hp() < initial_hp, "Target should take damage"

    def test_units_out_of_range_no_damage(self):
        world = model.World(300, 300, tick_rate=10, curriculum_stage=0)
        team_a = model.Team(team_id=False)
        team_b = model.Team(team_id=True)

        attacker = model.Infantry()
        attacker.set_team(False); attacker.set_x(50); attacker.set_y(50)

        target = model.Infantry()
        target.set_team(True); target.set_x(200); target.set_y(50)
        # Distance = 150, infantry range = 4 — should not hit

        team_a.add_formation(model.Formation([attacker]))
        team_b.add_formation(model.Formation([target]))
        world.add_team(team_a)
        world.add_team(team_b)

        initial_hp = target.get_hp()
        world._combat_phase(world.get_living_units())
        assert target.get_hp() == initial_hp

    def test_dead_units_do_not_attack(self):
        world = model.World(100, 100, tick_rate=10, curriculum_stage=0)
        team_a = model.Team(team_id=False)
        team_b = model.Team(team_id=True)

        attacker = model.Infantry()
        attacker.set_team(False); attacker.set_x(50); attacker.set_y(50)
        attacker.set_hp(0)  # dead

        target = model.Infantry()
        target.set_team(True); target.set_x(52); target.set_y(50)

        team_a.add_formation(model.Formation([attacker]))
        team_b.add_formation(model.Formation([target]))
        world.add_team(team_a)
        world.add_team(team_b)

        initial_hp = target.get_hp()
        world._combat_phase(world.get_living_units())
        assert target.get_hp() == initial_hp


# ══════════════════════════════════════════════════════════════════════
# 7.11  Unit Tests — Collision Resolution
# ══════════════════════════════════════════════════════════════════════

class TestCollisionResolution:
    """Verify overlapping units are pushed apart."""

    def test_overlapping_units_separated(self):
        world = model.World(100, 100, tick_rate=10, curriculum_stage=0)
        u1 = model.Infantry(); u1.set_x(50); u1.set_y(50)
        u2 = model.Infantry(); u2.set_x(50); u2.set_y(50)

        world._resolve_collisions([u1, u2])
        dist = math.hypot(u1.get_x() - u2.get_x(), u1.get_y() - u2.get_y())
        min_dist = u1.radius + u2.radius
        assert dist >= min_dist - 0.01, \
            f"Units should be at least {min_dist} apart, got {dist}"

    def test_non_overlapping_units_unchanged(self):
        world = model.World(100, 100, tick_rate=10, curriculum_stage=0)
        u1 = model.Infantry(); u1.set_x(10); u1.set_y(50)
        u2 = model.Infantry(); u2.set_x(90); u2.set_y(50)
        x1_before, x2_before = u1.get_x(), u2.get_x()

        world._resolve_collisions([u1, u2])
        assert u1.get_x() == x1_before
        assert u2.get_x() == x2_before

    def test_units_clamped_to_world_bounds(self):
        world = model.World(100, 100, tick_rate=10, curriculum_stage=0)
        u1 = model.Infantry(); u1.set_x(0); u1.set_y(0)
        u2 = model.Infantry(); u2.set_x(0); u2.set_y(0)

        world._resolve_collisions([u1, u2])
        for u in [u1, u2]:
            assert u.get_x() >= 0 and u.get_x() <= 99
            assert u.get_y() >= 0 and u.get_y() <= 99


# ══════════════════════════════════════════════════════════════════════
# 7.12  Unit Tests — Formation Cohesion
# ══════════════════════════════════════════════════════════════════════

class TestCohesion:
    """Verify cohesion metric produces valid values."""

    def test_cohesion_in_valid_range(self):
        units = [model.Infantry() for _ in range(20)]
        f = model.Formation(units)
        f.move(0, 0, 50, 0)
        # Place units at their destinations
        for u in units:
            dx, dy = u.get_destination()
            u.set_x(dx); u.set_y(dy)
        cohesion = f.measure_formation_cohesion()
        assert 0.0 <= cohesion <= 1.0

    def test_perfect_line_high_cohesion(self):
        units = [model.Infantry() for _ in range(10)]
        f = model.Formation(units)
        f.move(0, 50, 50, 50)
        for u in units:
            dx, dy = u.get_destination()
            u.set_x(dx); u.set_y(dy)
        cohesion = f.measure_formation_cohesion()
        assert cohesion > 0.8, f"Perfectly placed formation should have high cohesion, got {cohesion}"

    def test_single_unit_cohesion_is_one(self):
        units = [model.Infantry()]
        f = model.Formation(units)
        f.move(0, 0, 50, 0)
        assert f.measure_formation_cohesion() == 1.0

    def test_scattered_units_low_cohesion(self):
        units = [model.Infantry() for _ in range(10)]
        f = model.Formation(units)
        f.move(0, 0, 50, 0)
        # Scatter units randomly
        np.random.seed(123)
        for u in units:
            u.set_x(np.random.uniform(0, 200))
            u.set_y(np.random.uniform(0, 200))
        cohesion = f.measure_formation_cohesion()
        assert cohesion < 0.5, f"Scattered units should have low cohesion, got {cohesion}"


# ══════════════════════════════════════════════════════════════════════
# 7.13  Unit Tests — Team Population
# ══════════════════════════════════════════════════════════════════════

class TestPopulation:
    """Verify team population creates valid armies."""

    def test_populated_team_has_formations(self):
        world = model.World(300, 300, tick_rate=10)
        team = model.Team(team_id=False)
        world.populate_team(team)
        assert len(team.get_formations()) > 0

    def test_populated_team_has_living_units(self):
        world = model.World(300, 300, tick_rate=10)
        team = model.Team(team_id=False)
        world.populate_team(team)
        assert len(team.get_living_units()) > 0

    def test_all_units_have_correct_team_id(self):
        world = model.World(300, 300, tick_rate=10)
        team = model.Team(team_id=True)
        world.populate_team(team)
        for u in team.get_all_units():
            assert u.get_team() == True

    def test_units_within_world_bounds(self):
        world = model.World(300, 300, tick_rate=10)
        team = model.Team(team_id=False)
        world.populate_team(team)
        for u in team.get_all_units():
            assert 0 <= u.get_x() <= 300, f"Unit x={u.get_x()} out of bounds"
            assert 0 <= u.get_y() <= 300, f"Unit y={u.get_y()} out of bounds"

    def test_max_six_formations(self):
        """All presets should have <= 6 formations (MAX_FORMATIONS)."""
        for preset in model.ARMY_PRESETS:
            assert len(preset["formations"]) <= 6, \
                f"Preset '{preset['name']}' has {len(preset['formations'])} formations (max 6)"


# ══════════════════════════════════════════════════════════════════════
# 7.14  Unit Tests — Tick Execution
# ══════════════════════════════════════════════════════════════════════

class TestTick:
    """Verify the tick loop runs without errors."""

    def test_tick_increments_count(self):
        world = model.World(300, 300, tick_rate=10, curriculum_stage=0)
        team_a = model.Team(team_id=False)
        team_b = model.Team(team_id=True)
        world.populate_team(team_a)
        world.populate_team(team_b)
        assert world.get_tick_count() == 0
        world.tick()
        assert world.get_tick_count() == 1

    def test_100_ticks_no_crash(self):
        world = model.World(300, 300, tick_rate=10, curriculum_stage=3)
        team_a = model.Team(team_id=False)
        team_b = model.Team(team_id=True)
        world.populate_team(team_a)
        world.populate_team(team_b)
        for _ in range(100):
            world.tick()
        # If we get here without an exception, the test passes

    def test_battle_eventually_ends(self):
        """At least one team should take damage when formations engage."""
        world = model.World(300, 300, tick_rate=10, curriculum_stage=0)
        team_a = model.Team(team_id=False)
        team_b = model.Team(team_id=True)
        world.populate_team(team_a)
        world.populate_team(team_b)

        # Move all formations toward the centre so they engage
        for f in team_a.get_formations():
            f.move(140, 50, 160, 250)
        for f in team_b.get_formations():
            f.move(140, 50, 160, 250)

        initial_hp = sum(u.get_hp() for u in world.get_all_units())
        for _ in range(2000):
            world.tick()
            if team_a.is_defeated() or team_b.is_defeated():
                break
        final_hp = sum(u.get_hp() for u in world.get_all_units())
        assert final_hp < initial_hp, "Some damage should occur when formations engage"


# ══════════════════════════════════════════════════════════════════════
# 7.15  Unit Tests — Bresenham Algorithm
# ══════════════════════════════════════════════════════════════════════

class TestBresenham:
    """Verify Bresenham's line algorithm produces correct paths."""

    def test_horizontal_line(self):
        points = list(model.World._bresenham(0, 0, 5, 0))
        assert points == [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0)]

    def test_vertical_line(self):
        points = list(model.World._bresenham(0, 0, 0, 3))
        assert points == [(0, 0), (0, 1), (0, 2), (0, 3)]

    def test_single_point(self):
        points = list(model.World._bresenham(5, 5, 5, 5))
        assert points == [(5, 5)]

    def test_includes_start_and_end(self):
        points = list(model.World._bresenham(0, 0, 10, 7))
        assert points[0] == (0, 0)
        assert points[-1] == (10, 7)


# ══════════════════════════════════════════════════════════════════════
# 7.16  Training Environment — Observation and Action Spaces
# ══════════════════════════════════════════════════════════════════════

from gym_wrapper import BattleEnv, OBS_SIZE, MAX_FORMATIONS, ACTIONS_PER_FORMATION


class TestTrainingEnvironment:
    """Verify the Gymnasium environment contract."""

    def test_observation_space_shape(self):
        env = BattleEnv(curriculum_stage=0)
        obs, _ = env.reset()
        assert obs.shape == (OBS_SIZE,), \
            f"Obs shape should be ({OBS_SIZE},), got {obs.shape}"

    def test_observation_values_in_range(self):
        env = BattleEnv(curriculum_stage=0)
        obs, _ = env.reset()
        assert np.all(obs >= 0.0), "All obs values should be >= 0"
        assert np.all(obs <= 1.0), "All obs values should be <= 1"

    def test_action_space_shape(self):
        env = BattleEnv(curriculum_stage=0)
        expected = (MAX_FORMATIONS * ACTIONS_PER_FORMATION,)
        assert env.action_space.shape == expected

    def test_step_returns_correct_tuple(self):
        env = BattleEnv(curriculum_stage=0)
        env.reset()
        action = env.action_space.sample()
        result = env.step(action)
        assert len(result) == 5, "step() should return (obs, reward, terminated, truncated, info)"
        obs, reward, terminated, truncated, info = result
        assert obs.shape == (OBS_SIZE,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_reset_produces_different_maps(self):
        env = BattleEnv(curriculum_stage=0)
        obs1, _ = env.reset(seed=1)
        obs2, _ = env.reset(seed=2)
        assert not np.array_equal(obs1, obs2), \
            "Different seeds should produce different observations"

    def test_dead_formations_zeroed_in_obs(self):
        env = BattleEnv(curriculum_stage=0)
        env.reset()
        # Kill all units in the first friendly formation
        first_formation = env.friendly_team.get_formations()[0]
        for u in first_formation.get_units():
            u.set_hp(0)
        obs = env._build_obs()
        # First formation's slot should be all zeros
        slot = obs[0:14]
        assert np.all(slot == 0.0), \
            "Dead formation should be zeroed in observation"


# ══════════════════════════════════════════════════════════════════════
# 7.17  Self-Play Mirroring Validation
# ══════════════════════════════════════════════════════════════════════

class TestSelfPlayMirroring:
    """Verify the opponent observation is correctly mirrored."""

    def test_mirrored_obs_same_shape(self):
        env = BattleEnv(curriculum_stage=0)
        env.reset()
        friendly_obs = env._build_obs()
        opponent_obs = env._build_opponent_obs()
        assert friendly_obs.shape == opponent_obs.shape

    def test_mirrored_obs_swaps_teams(self):
        """Friendly formations in normal obs should appear in enemy
        slot of mirrored obs, and vice versa."""
        env = BattleEnv(curriculum_stage=0)
        env.reset()
        friendly_obs = env._build_obs()
        opponent_obs = env._build_opponent_obs()

        friendly_slot_size = MAX_FORMATIONS * 14

        # In friendly obs: slot 0 = friendly, slot 1 = enemy
        # In opponent obs: slot 0 = enemy (mirrored), slot 1 = friendly (mirrored)
        # The presence flags should swap
        friendly_presence = friendly_obs[0]  # first friendly formation alive?
        opponent_enemy_presence = opponent_obs[friendly_slot_size]  # friendly in enemy slot

        # Both should be 1.0 if the formation is alive
        if friendly_presence == 1.0:
            assert opponent_enemy_presence == 1.0, \
                "Alive friendly formation should appear in opponent's enemy slot"

    def test_mirrored_x_coordinates_sum_to_one(self):
        """For a mirrored observation, normalised x positions should
        satisfy: normal_x + mirrored_x ≈ 1.0."""
        env = BattleEnv(curriculum_stage=0)
        env.reset()
        friendly_obs = env._build_obs()
        opponent_obs = env._build_opponent_obs()

        friendly_slot_size = MAX_FORMATIONS * 14

        # Check first enemy formation's avg_x (index 11 within slot)
        # In friendly obs: enemy is in second block
        enemy_avg_x_normal = friendly_obs[friendly_slot_size + 11]
        # In opponent obs: same team is in first block (mirrored)
        enemy_avg_x_mirrored = opponent_obs[0 + 11]

        if enemy_avg_x_normal > 0:  # only test if formation exists
            total = enemy_avg_x_normal + enemy_avg_x_mirrored
            assert abs(total - 1.0) < 0.05, \
                f"Mirrored x should sum to ~1.0, got {total}"

    def test_global_features_unchanged_by_mirror(self):
        env = BattleEnv(curriculum_stage=0)
        env.reset()
        friendly_obs = env._build_obs()
        opponent_obs = env._build_opponent_obs()

        global_offset = MAX_FORMATIONS * 14 * 2
        assert np.allclose(
            friendly_obs[global_offset:],
            opponent_obs[global_offset:]
        ), "Global features should be identical in both observations"


# ══════════════════════════════════════════════════════════════════════
# 7.18  Reward Validation
# ══════════════════════════════════════════════════════════════════════

class TestRewardValidation:
    """Verify reward signals are bounded and directionally correct."""

    def test_reward_is_finite(self):
        env = BattleEnv(curriculum_stage=0)
        env.reset()
        for _ in range(50):
            action = env.action_space.sample()
            _, reward, terminated, truncated, _ = env.step(action)
            assert np.isfinite(reward), f"Reward should be finite, got {reward}"
            if terminated or truncated:
                break

    def test_victory_gives_positive_terminal_reward(self):
        env = BattleEnv(curriculum_stage=0)
        env.reset()
        # Kill all enemy units
        for f in env.enemy_team.get_formations():
            for u in f.get_units():
                u.set_hp(0)
        # Use zero action to avoid shield wall penalties on non-infantry
        action = np.zeros(MAX_FORMATIONS * ACTIONS_PER_FORMATION, dtype=np.float32)
        _, reward, terminated, _, info = env.step(action)
        assert terminated
        assert info.get("won") == True
        assert reward > 0, f"Victory should produce positive reward, got {reward}"

    def test_defeat_gives_negative_terminal_reward(self):
        env = BattleEnv(curriculum_stage=0)
        env.reset()
        # Kill all friendly units
        for f in env.friendly_team.get_formations():
            for u in f.get_units():
                u.set_hp(0)
        action = env.action_space.sample()
        _, reward, terminated, _, info = env.step(action)
        assert terminated
        assert info.get("won") == False
        assert reward < 0, "Defeat should produce negative reward"

    def test_shield_wall_on_cavalry_penalised(self):
        env = BattleEnv(curriculum_stage=0)
        env.reset()
        # Find a cavalry formation
        cav_index = None
        for i, f in enumerate(env.friendly_team.get_formations()):
            living = f.get_living_units()
            if living and isinstance(living[0], model.Cavalry):
                cav_index = i
                break
        if cav_index is not None:
            action = np.zeros(MAX_FORMATIONS * ACTIONS_PER_FORMATION, dtype=np.float32)
            # Set shield wall toggle to positive for cavalry formation
            action[cav_index * ACTIONS_PER_FORMATION + 4] = 1.0
            _, reward, _, _, _ = env.step(action)
            # Should include -10.0 penalty
            assert reward < -5.0, \
                "Activating shield wall on cavalry should incur heavy penalty"

    def test_episode_truncates_at_max_steps(self):
        env = BattleEnv(curriculum_stage=0)
        env.reset()
        env.current_step = 1999
        action = env.action_space.sample()
        _, _, _, truncated, _ = env.step(action)
        assert truncated, "Episode should truncate at MAX_STEPS"


# ══════════════════════════════════════════════════════════════════════
# 7.19  Performance Testing
# ══════════════════════════════════════════════════════════════════════

import time


class TestPerformance:
    """Verify simulation scales and tick rate is acceptable."""

    def test_tick_performance_small(self):
        """100-unit battle should tick in under 10ms."""
        world = model.World(300, 300, tick_rate=10, curriculum_stage=3)
        team_a = model.Team(team_id=False)
        team_b = model.Team(team_id=True)
        # Use small presets
        np.random.seed(0)
        world.populate_team(team_a)
        world.populate_team(team_b)

        # Warm up
        for _ in range(5):
            world.tick()

        start = time.perf_counter()
        for _ in range(100):
            world.tick()
        elapsed = (time.perf_counter() - start) / 100

        assert elapsed < 0.01, \
            f"Average tick should be under 10ms, got {elapsed*1000:.1f}ms"

    def test_tick_performance_large(self):
        """200+ unit battle should tick in under 50ms."""
        world = model.World(300, 300, tick_rate=10, curriculum_stage=3)
        team_a = model.Team(team_id=False)
        team_b = model.Team(team_id=True)

        # Manually create large armies
        for _ in range(4):
            units = [model.Infantry() for _ in range(50)]
            for u in units:
                u.set_team(False)
                u.set_x(np.random.uniform(50, 150))
                u.set_y(np.random.uniform(50, 250))
            f = model.Formation(units)
            f.move(50, 50, 150, 250)
            team_a.add_formation(f)

        for _ in range(4):
            units = [model.Infantry() for _ in range(50)]
            for u in units:
                u.set_team(True)
                u.set_x(np.random.uniform(150, 250))
                u.set_y(np.random.uniform(50, 250))
            f = model.Formation(units)
            f.move(150, 50, 250, 250)
            team_b.add_formation(f)

        world.add_team(team_a)
        world.add_team(team_b)

        # Warm up
        for _ in range(5):
            world.tick()

        start = time.perf_counter()
        for _ in range(50):
            world.tick()
        elapsed = (time.perf_counter() - start) / 50

        assert elapsed < 0.05, \
            f"Large battle tick should be under 50ms, got {elapsed*1000:.1f}ms"

    def test_env_reset_performance(self):
        """Environment reset (including terrain gen) under 500ms."""
        env = BattleEnv(curriculum_stage=3)

        start = time.perf_counter()
        for _ in range(10):
            env.reset()
        elapsed = (time.perf_counter() - start) / 10

        assert elapsed < 0.5, \
            f"Reset should be under 500ms, got {elapsed*1000:.0f}ms"


# ══════════════════════════════════════════════════════════════════════
# 7.20  Curriculum Callback State Machine
# ══════════════════════════════════════════════════════════════════════

from unittest.mock import MagicMock, patch
from curriculum_callback import CurriculumSelfPlayCallback
import tempfile
import os


class TestCurriculumStateMachine:
    """Verify the curriculum callback transitions correctly
    without requiring a real training run."""

    def _make_callback(self, threshold=0.7, window=10, max_stage=3):
        cb = CurriculumSelfPlayCallback(
            save_path=tempfile.mkdtemp(),
            promotion_threshold=threshold,
            window_size=window,
            max_stage=max_stage,
            verbose=0,
        )
        cb.model = MagicMock()
        # training_env is a property that returns model.get_env()
        mock_env = MagicMock()
        cb.model.get_env.return_value = mock_env
        cb.locals = {"dones": [], "infos": []}
        return cb

    def _feed_outcomes(self, cb, wins: int, losses: int):
        """Simulate a batch of episode outcomes."""
        for _ in range(wins):
            cb.locals = {"dones": [True], "infos": [{"won": True}]}
            cb._on_step()
        for _ in range(losses):
            cb.locals = {"dones": [True], "infos": [{"won": False}]}
            cb._on_step()

    def test_starts_at_stage0_vs_scripted(self):
        cb = self._make_callback()
        assert cb.current_stage == 0
        assert cb.vs_self_play == False
        assert cb.fully_complete == False

    def test_no_promotion_below_threshold(self):
        cb = self._make_callback(threshold=0.7, window=10)
        # 6 wins out of 10 = 60%, below 70%
        self._feed_outcomes(cb, wins=6, losses=4)
        assert cb.current_stage == 0
        assert cb.vs_self_play == False

    def test_promotes_to_self_play_at_threshold(self):
        cb = self._make_callback(threshold=0.7, window=10)
        # 8 wins out of 10 = 80%, above 70%
        self._feed_outcomes(cb, wins=8, losses=2)
        assert cb.vs_self_play == True
        assert cb.current_stage == 0
        # Should have saved a snapshot
        cb.model.save.assert_called_once()
        # Should have called set_opponent_model on the env
        cb.training_env.env_method.assert_any_call(
            "set_opponent_model", cb.model.save.call_args[0][0]
        )

    def test_promotes_to_next_stage_after_self_play(self):
        cb = self._make_callback(threshold=0.7, window=10)
        # Stage 0 vs scripted → self-play
        self._feed_outcomes(cb, wins=8, losses=2)
        assert cb.vs_self_play == True
        assert cb.current_stage == 0
        # Stage 0 vs self-play → Stage 1 vs scripted
        self._feed_outcomes(cb, wins=8, losses=2)
        assert cb.vs_self_play == False
        assert cb.current_stage == 1
        # Should have reset opponent to scripted (None)
        cb.training_env.env_method.assert_any_call("set_opponent_model", None)
        cb.training_env.env_method.assert_any_call("set_curriculum_stage", 1)

    def test_full_curriculum_completion(self):
        cb = self._make_callback(threshold=0.7, window=10, max_stage=3)
        # Walk through all 8 transitions: 4 stages × 2 phases
        for expected_stage in range(4):
            # vs scripted
            assert cb.current_stage == expected_stage
            assert cb.vs_self_play == False
            self._feed_outcomes(cb, wins=8, losses=2)
            # vs self-play
            assert cb.vs_self_play == True
            self._feed_outcomes(cb, wins=8, losses=2)

        assert cb.fully_complete == True

    def test_no_further_transitions_after_complete(self):
        cb = self._make_callback(threshold=0.7, window=10, max_stage=0)
        # Complete the only stage
        self._feed_outcomes(cb, wins=8, losses=2)  # scripted → self
        self._feed_outcomes(cb, wins=8, losses=2)  # self → complete
        assert cb.fully_complete == True
        # Feed more wins — nothing should change
        call_count = cb.model.save.call_count
        self._feed_outcomes(cb, wins=10, losses=0)
        assert cb.model.save.call_count == call_count

    def test_episode_outcomes_cleared_on_promotion(self):
        cb = self._make_callback(threshold=0.7, window=10)
        self._feed_outcomes(cb, wins=8, losses=2)
        # After promotion, outcomes should be cleared
        assert len(cb.episode_outcomes) == 0

    def test_window_uses_most_recent(self):
        cb = self._make_callback(threshold=0.7, window=10)
        # 5 losses then 10 wins — only last 10 count
        self._feed_outcomes(cb, wins=0, losses=5)
        self._feed_outcomes(cb, wins=10, losses=0)
        assert cb.vs_self_play == True  # should have promoted


# ══════════════════════════════════════════════════════════════════════
# 7.21  Tests requiring sb3_contrib (run on training machine)
# ══════════════════════════════════════════════════════════════════════

# These tests are skipped in CI but can be run on the training machine
# where sb3_contrib, torch, and model checkpoints are available.
# Run with: python -m pytest test_model.py -k "sb3" --no-header

try:
    from sb3_contrib import RecurrentPPO
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False


@pytest.mark.skipif(not HAS_SB3, reason="sb3_contrib not installed")
class TestModelLoading:
    """Verify trained model snapshots load and produce valid actions."""

    CHECKPOINT = "battle_agent_finetuned.zip"

    def test_model_loads_and_predicts(self):
        """Load a checkpoint and verify it produces valid actions."""
        if not os.path.exists(self.CHECKPOINT):
            pytest.skip("No checkpoint available")

        loaded_model = RecurrentPPO.load(self.CHECKPOINT)
        env = BattleEnv(curriculum_stage=3)
        obs, _ = env.reset()

        lstm_states = None
        episode_start = np.ones((1,), dtype=bool)
        obs_batch = np.expand_dims(obs, axis=0)

        action, lstm_states = loaded_model.predict(
            obs_batch, state=lstm_states,
            episode_start=episode_start, deterministic=True
        )
        action = action.squeeze(0)
        assert action.shape == (MAX_FORMATIONS * ACTIONS_PER_FORMATION,)
        assert np.all(np.isfinite(action))

    def test_opponent_snapshot_loads_into_env(self):
        """Verify a snapshot can be set as opponent and the env still steps."""
        if not os.path.exists(self.CHECKPOINT):
            pytest.skip("No checkpoint available")

        env = BattleEnv(curriculum_stage=3)
        env.reset()
        env.set_opponent_model(self.CHECKPOINT)

        # Take 10 steps with opponent using the loaded model
        for _ in range(10):
            action = np.zeros(MAX_FORMATIONS * ACTIONS_PER_FORMATION, dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            assert np.all(np.isfinite(obs))
            if terminated or truncated:
                break

    def test_empty_opponent_snapshots_handled(self):
        """Env should handle no opponent model gracefully (scripted fallback)."""
        env = BattleEnv(curriculum_stage=3)
        env.reset()
        env.set_opponent_model(None)
        action = np.zeros(MAX_FORMATIONS * ACTIONS_PER_FORMATION, dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        assert np.all(np.isfinite(obs))
        assert np.isfinite(reward)


@pytest.mark.skipif(not HAS_SB3, reason="sb3_contrib not installed")
class TestEarlyVsLateModel:
    """Compare behaviour of early and late training checkpoints.
    
    Requires two checkpoint files. Update the paths below to match
    your saved checkpoints.
    """

    EARLY_CHECKPOINT = "battle_agent_finetuned.zip"
    LATE_CHECKPOINT = "battle_agent_ultra_finetuned.zip"
    NUM_EVAL_EPISODES = 5
    MAX_EVAL_STEPS = 500  # cap per episode to avoid hanging

    def _evaluate_model(self, model_path, num_episodes):
        """Run evaluation episodes and return win rate and avg reward."""
        loaded_model = RecurrentPPO.load(model_path)
        env = BattleEnv(curriculum_stage=3, frame_skip=5)

        wins = 0
        total_reward = 0.0

        for _ in range(num_episodes):
            obs, _ = env.reset()
            lstm_states = None
            episode_start = np.ones((1,), dtype=bool)
            episode_reward = 0.0
            steps = 0

            while steps < self.MAX_EVAL_STEPS:
                obs_batch = np.expand_dims(obs, axis=0)
                action, lstm_states = loaded_model.predict(
                    obs_batch, state=lstm_states,
                    episode_start=episode_start, deterministic=True
                )
                episode_start = np.zeros((1,), dtype=bool)
                action = action.squeeze(0)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                steps += 1
                if terminated or truncated:
                    break

            if info.get("won"):
                wins += 1
            total_reward += episode_reward

        return wins / num_episodes, total_reward / num_episodes

    def test_late_model_wins_more_than_early(self):
        if not os.path.exists(self.EARLY_CHECKPOINT):
            pytest.skip(f"Early checkpoint not found: {self.EARLY_CHECKPOINT}")
        if not os.path.exists(self.LATE_CHECKPOINT):
            pytest.skip(f"Late checkpoint not found: {self.LATE_CHECKPOINT}")

        early_wr, early_reward = self._evaluate_model(
            self.EARLY_CHECKPOINT, self.NUM_EVAL_EPISODES)
        late_wr, late_reward = self._evaluate_model(
            self.LATE_CHECKPOINT, self.NUM_EVAL_EPISODES)

        print(f"\nEarly model: win rate={early_wr:.0%}, avg reward={early_reward:.1f}")
        print(f"Late model:  win rate={late_wr:.0%}, avg reward={late_reward:.1f}")

        assert late_wr >= early_wr, \
            f"Late model ({late_wr:.0%}) should win at least as often as early ({early_wr:.0%})"

    def test_late_model_higher_avg_reward(self):
        if not os.path.exists(self.EARLY_CHECKPOINT):
            pytest.skip(f"Early checkpoint not found: {self.EARLY_CHECKPOINT}")
        if not os.path.exists(self.LATE_CHECKPOINT):
            pytest.skip(f"Late checkpoint not found: {self.LATE_CHECKPOINT}")

        _, early_reward = self._evaluate_model(
            self.EARLY_CHECKPOINT, self.NUM_EVAL_EPISODES)
        _, late_reward = self._evaluate_model(
            self.LATE_CHECKPOINT, self.NUM_EVAL_EPISODES)

        assert late_reward > early_reward, \
            f"Late model ({late_reward:.1f}) should have higher avg reward than early ({early_reward:.1f})"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])