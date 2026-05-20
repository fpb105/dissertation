import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math

import model

# ── Constants ──────────────────────────────────────────────────────
MAX_FORMATIONS = 6
FEATURES_PER_FORMATION = 14
NUM_GLOBAL_FEATURES = 3
OBS_SIZE = (MAX_FORMATIONS * FEATURES_PER_FORMATION * 2) + NUM_GLOBAL_FEATURES

# Each formation: (port_x, port_y, starboard_x, starboard_y, shield_wall)
ACTIONS_PER_FORMATION = 5

UNIT_TYPE_ENCODING = {
    "infantry": 0.0,
    "archer":   0.5,
    "cavalry":  1.0,
}

UNIT_CLASS_TO_TYPE = {
    model.Infantry: "infantry",
    model.Archer:   "archer",
    model.Cavalry:  "cavalry",
}

MAX_STEPS = 2000

# How far the scripted opponent's target advances per tick
SCRIPTED_ADVANCE_SPEED = 2.0


class BattleEnv(gym.Env):

    def __init__(self, curriculum_stage: int = 0, frame_skip: int = 1):
        super().__init__()

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(MAX_FORMATIONS * ACTIONS_PER_FORMATION,),
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(OBS_SIZE,),
            dtype=np.float32,
        )

        self.curriculum_stage = curriculum_stage
        self.tick_rate = 10
        self.frame_skip = frame_skip
        self.world = None
        self.friendly_team = None
        self.enemy_team = None
        self.current_step = 0

        self.opponent_model = None
        self.opponent_lstm_states = None
        self.opponent_episode_start = None

    # ── Curriculum / opponent management ──────────────────────────
    def set_curriculum_stage(self, stage: int):
        self.curriculum_stage = stage

    def get_curriculum_stage(self) -> int:
        return self.curriculum_stage

    def set_opponent_model(self, path: str | None):
        if path is None:
            self.opponent_model = None
        else:
            from sb3_contrib import RecurrentPPO
            self.opponent_model = RecurrentPPO.load(path)

    # ── Helpers ───────────────────────────────────────────────────
    def _decode_positions(self, raw: np.ndarray) -> np.ndarray:
        decoded = np.empty_like(raw)
        decoded[:, 0] = (raw[:, 0] + 1.0) / 2.0 * self.world.width
        decoded[:, 1] = (raw[:, 1] + 1.0) / 2.0 * self.world.height
        decoded[:, 2] = (raw[:, 2] + 1.0) / 2.0 * self.world.width
        decoded[:, 3] = (raw[:, 3] + 1.0) / 2.0 * self.world.height
        decoded[:, [0, 2]] = np.clip(decoded[:, [0, 2]], 0, self.world.width - 1)
        decoded[:, [1, 3]] = np.clip(decoded[:, [1, 3]], 0, self.world.height - 1)
        return decoded

    # ── Shield wall application ───────────────────────────────────
    def _apply_shield_wall(self, formations, shield_wall_raw):
        reward = 0.0
        for i, formation in enumerate(formations):
            if i >= MAX_FORMATIONS:
                break
            living = formation.get_living_units()
            if not living:
                continue

            wants_wall = shield_wall_raw[i] > 0.0
            is_infantry = isinstance(living[0], model.Infantry)

            if wants_wall:
                if is_infantry:
                    for u in living:
                        u.sheild_wall = True   # direct attribute, no getter overhead
                    reward += 0.5
                else:
                    reward -= 10.0
            else:
                if is_infantry:
                    for u in living:
                        u.sheild_wall = False
        return reward

    # ── Reset ─────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.world = model.World(300, 300, self.tick_rate, self.curriculum_stage)

        self.friendly_team = model.Team(team_id=False)
        self.enemy_team = model.Team(team_id=True)

        self.world.populate_team(self.friendly_team)
        self.world.populate_team(self.enemy_team)

        self.current_step = 0

        self.opponent_lstm_states = None
        self.opponent_episode_start = np.ones((1,), dtype=bool)

        self.prev_friendly_hp = self._total_hp(self.friendly_team)
        self.prev_enemy_hp = self._total_hp(self.enemy_team)
        self.prev_avg_distance = self._avg_distance_to_enemy()

        obs = self._build_obs()
        return obs, {}

    # ── Step ──────────────────────────────────────────────────────
    def step(self, action):
        raw = action.reshape(MAX_FORMATIONS, ACTIONS_PER_FORMATION)
        positions = self._decode_positions(raw[:, :4])
        shield_wall_raw = raw[:, 4]

        total_reward = 0.0
        terminated = False
        truncated = False

        friendly_formations = self.friendly_team.get_formations()
        shield_reward = self._apply_shield_wall(friendly_formations, shield_wall_raw)
        total_reward += shield_reward

        for _ in range(self.frame_skip):
            for i, formation in enumerate(friendly_formations):
                if i >= MAX_FORMATIONS:
                    break
                if not formation.get_living_units():
                    continue
                px, py, sx, sy = positions[i]
                formation.move(px, py, sx, sy)

            self._opponent_act()
            self.world.tick()
            self.current_step += 1

            reward = self._compute_reward()
            total_reward += reward

            terminated = (
                self.friendly_team.is_defeated()
                or self.enemy_team.is_defeated()
            )
            truncated = self.current_step >= MAX_STEPS

            if terminated or truncated:
                break

        obs = self._build_obs()
        info = {}
        if terminated:
            info["won"] = self.enemy_team.is_defeated()

        return obs, total_reward, terminated, truncated, info

    # ── Opponent logic ────────────────────────────────────────────
    def _opponent_act(self):
        enemy_formations = self.enemy_team.get_formations()

        if self.opponent_model is None:
            advance_step = SCRIPTED_ADVANCE_SPEED / self.tick_rate
            for formation in enemy_formations:
                if not formation.get_living_units():
                    continue
                px, py = formation.get_port()
                sx, sy = formation.get_starboard()
                formation.move(px - advance_step, py, sx - advance_step, sy)
        else:
            opponent_obs = self._build_opponent_obs()
            obs_batch = np.expand_dims(opponent_obs, axis=0)

            enemy_action, self.opponent_lstm_states = (
                self.opponent_model.predict(
                    obs_batch,
                    state=self.opponent_lstm_states,
                    episode_start=self.opponent_episode_start,
                    deterministic=False,
                )
            )
            self.opponent_episode_start = np.zeros((1,), dtype=bool)
            enemy_action = enemy_action.squeeze(0)

            raw = enemy_action.reshape(MAX_FORMATIONS, ACTIONS_PER_FORMATION)
            positions = self._decode_positions(raw[:, :4])
            enemy_shield_wall_raw = raw[:, 4]

            for i, formation in enumerate(enemy_formations):
                if i >= MAX_FORMATIONS:
                    break
                living = formation.get_living_units()
                if not living:
                    continue

                if isinstance(living[0], model.Infantry):
                    wall_on = enemy_shield_wall_raw[i] > 0.0
                    for u in living:
                        u.sheild_wall = wall_on   # direct attribute

                px, py, sx, sy = positions[i]
                px = self.world.width - px
                sx = self.world.width - sx
                formation.move(px, py, sx, sy)

    def _build_opponent_obs(self) -> np.ndarray:
        obs = np.zeros(OBS_SIZE, dtype=np.float32)
        self._encode_team_formations(
            obs, self.enemy_team.get_formations(), offset=0, mirror_x=True,
        )
        self._encode_team_formations(
            obs, self.friendly_team.get_formations(),
            offset=MAX_FORMATIONS * FEATURES_PER_FORMATION, mirror_x=True,
        )
        self._encode_global_features(obs, offset=MAX_FORMATIONS * FEATURES_PER_FORMATION * 2)
        return obs

    # ── Observation builder ───────────────────────────────────────
    def _build_obs(self) -> np.ndarray:
        obs = np.zeros(OBS_SIZE, dtype=np.float32)
        self._encode_team_formations(
            obs, self.friendly_team.get_formations(), offset=0, mirror_x=False,
        )
        self._encode_team_formations(
            obs, self.enemy_team.get_formations(),
            offset=MAX_FORMATIONS * FEATURES_PER_FORMATION, mirror_x=False,
        )
        self._encode_global_features(obs, offset=MAX_FORMATIONS * FEATURES_PER_FORMATION * 2)
        return obs

    def _encode_team_formations(self, obs, formations, offset, mirror_x=False):
        for i, formation in enumerate(formations):
            if i >= MAX_FORMATIONS:
                break
            living = formation.get_living_units()
            if not living:
                continue

            base = offset + (i * FEATURES_PER_FORMATION)
            unit_type_name = UNIT_CLASS_TO_TYPE.get(type(living[0]), "infantry")

            xs = [u.x for u in living]
            ys = [u.y for u in living]
            n = len(xs)
            avg_x = sum(xs) / n
            avg_y = sum(ys) / n
            min_x = min(xs)
            max_x = max(xs)
            min_y = min(ys)
            max_y = max(ys)

            if mirror_x:
                avg_x = self.world.width - avg_x
                min_x, max_x = self.world.width - max_x, self.world.width - min_x

            gx = int(max(0, min(round(avg_x), self.world.width - 1)))
            gy = int(max(0, min(round(avg_y), self.world.height - 1)))

            obs[base + 0]  = 1.0
            obs[base + 1]  = UNIT_TYPE_ENCODING.get(unit_type_name, 0.0)
            obs[base + 2]  = min(n / 50.0, 1.0)
            obs[base + 3]  = min_x / self.world.width
            obs[base + 4]  = max_x / self.world.width
            obs[base + 5]  = avg_y / self.world.height
            obs[base + 6]  = self.world.tiles[gy, gx, 1] / 5.0   # height
            obs[base + 7]  = self.world.tiles[gy, gx, 0] / 2.0   # terrain
            if unit_type_name == "infantry":
                obs[base + 8] = 1.0 if living[0].sheild_wall else 0.5  # direct attribute
            else:
                obs[base + 8] = 0.0
            obs[base + 9]  = max_y / self.world.height
            obs[base + 10] = min_y / self.world.height
            obs[base + 11] = avg_x / self.world.width
            obs[base + 12] = formation.get_rank_count() / self.world.diagonal
            obs[base + 13] = formation.get_rank_width() / self.world.diagonal

    def _encode_global_features(self, obs, offset):
        obs[offset + 0] = self.current_step / MAX_STEPS
        obs[offset + 1] = self.world.width / 1000.0
        obs[offset + 2] = self.world.height / 1000.0

    # ── Reward ────────────────────────────────────────────────────
    def _total_hp(self, team) -> int:
        # FIX: was u.health (AttributeError), correct attribute is u.hp
        return sum(u.hp for u in team.get_living_units())

    def _avg_distance_to_enemy(self) -> float:
        # Simplified from O(f²) formation pairs to O(n) army centroid comparison
        friendly_living = self.friendly_team.get_living_units()
        enemy_living = self.enemy_team.get_living_units()

        if not friendly_living or not enemy_living:
            return 0.0

        fx = sum(u.x for u in friendly_living) / len(friendly_living)
        fy = sum(u.y for u in friendly_living) / len(friendly_living)
        ex = sum(u.x for u in enemy_living) / len(enemy_living)
        ey = sum(u.y for u in enemy_living) / len(enemy_living)

        return math.hypot(fx - ex, fy - ey)

    def _compute_reward(self) -> float:
        """Reward components:
        1. HP differential: reward damage to enemy, penalise damage to self.
        2. Approach reward: reward closing distance to enemy formations.
        3. Cohesion: reward keeping units in tight formations.
        4. Terminal bonuses: large reward for victory, penalty for defeat.
        """
        reward = 0.0

        # Compute living units once per formation — reused across all reward components
        friendly_living_by_formation = [
            f.get_living_units() for f in self.friendly_team.get_formations()
        ]
        enemy_living_by_formation = [
            f.get_living_units() for f in self.enemy_team.get_formations()
        ]

        # ── 1. HP differential ────────────────────────────────────
        current_friendly_hp = sum(
            u.hp for living in friendly_living_by_formation for u in living
        )
        current_enemy_hp = sum(
            u.hp for living in enemy_living_by_formation for u in living
        )

        friendly_hp_lost = self.prev_friendly_hp - current_friendly_hp
        enemy_hp_lost    = self.prev_enemy_hp    - current_enemy_hp

        hp_scale = max(self.prev_friendly_hp + self.prev_enemy_hp, 1)
        reward += enemy_hp_lost    / hp_scale * 0.5
        reward -= friendly_hp_lost / hp_scale * 0.1

        self.prev_friendly_hp = current_friendly_hp
        self.prev_enemy_hp    = current_enemy_hp

        # ── 2. Approach reward ────────────────────────────────────
        avg_distance   = self._avg_distance_to_enemy()
        distance_closed = self.prev_avg_distance - avg_distance
        # Use cached diagonal — avoids math.hypot call every tick
        reward += distance_closed / self.world.diagonal * 2.0
        self.prev_avg_distance = avg_distance

        # ── 3. Cohesion ───────────────────────────────────────────
        for living in friendly_living_by_formation:
            if not living:
                continue
            # Find the formation for cohesion metrics — match by first unit
            formation = next(
                f for f in self.friendly_team.get_formations()
                if f.get_living_units() and f.get_living_units()[0] is living[0]
            )
            cohesion = formation.measure_formation_cohesion()
            cohesion_reward = max(0.0, 1.0 - abs(cohesion - 0.7) / 0.3)
            reward += cohesion_reward * 0.5

            n = len(living)
            unit_slot = 2 * living[0].radius + model.DEFAULT_UNIT_GAP  # direct attribute
            expected_len = n * unit_slot
            actual_len = formation.get_diagonal_length()
            if actual_len > expected_len * 2.0:
                reward -= 1.0

        # ── 4. Terminal bonuses ───────────────────────────────────
        if not any(living for living in enemy_living_by_formation):
            reward += 10.0
        elif not any(living for living in friendly_living_by_formation):
            reward -= 10.0

        return reward