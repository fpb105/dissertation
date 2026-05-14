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

        # ── Continuous action space ───────────────────────────────
        # Flat vector of (port_x, port_y, star_x, star_y, shield_wall)
        # per formation. Range [-1, 1].
        # Positions rescaled to world coordinates in step().
        # shield_wall: > 0 = activate, <= 0 = deactivate.
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

        # ── Opponent ──────────────────────────────────────────────
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
        """
        Map raw actions from [-1, 1] to world coordinates.

        Input shape:  (MAX_FORMATIONS, 4)
        Output shape: (MAX_FORMATIONS, 4) — (px, py, sx, sy) in world units

        [-1, 1] → [0, 1] → [0, dimension].
        Clamped to world bounds.
        """
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
        """
        Apply shield wall toggles and return the reward adjustment.

        For each formation with living units:
        - If shield_wall_raw[i] > 0 (toggle on):
            - Infantry: activate wall on all units, reward += 0.5
            - Non-infantry: ignore the action, reward -= 10.0
        - If shield_wall_raw[i] <= 0 (toggle off):
            - Infantry: deactivate wall on all units
            - Non-infantry: no effect
        """
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
                        u.set_wall(True)
                    reward += 0.5
                else:
                    # Penalty for attempting shield wall on non-infantry
                    reward -= 10.0
            else:
                # Deactivate wall if infantry, no-op otherwise
                if is_infantry:
                    for u in living:
                        u.set_wall(False)

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
        # (MAX_FORMATIONS * 5,) → (MAX_FORMATIONS, 5)
        raw = action.reshape(MAX_FORMATIONS, ACTIONS_PER_FORMATION)

        # Separate position columns from shield wall column
        positions = self._decode_positions(raw[:, :4])
        shield_wall_raw = raw[:, 4]

        total_reward = 0.0
        terminated = False
        truncated = False

        # ── Shield wall: apply once per step, not per frame-skip tick ─
        friendly_formations = self.friendly_team.get_formations()
        shield_reward = self._apply_shield_wall(
            friendly_formations, shield_wall_raw,
        )
        total_reward += shield_reward

        for _ in range(self.frame_skip):
            # ── Friendly: set formation targets ───────────────────
            for i, formation in enumerate(friendly_formations):
                if i >= MAX_FORMATIONS:
                    break
                if not formation.get_living_units():
                    continue

                px, py, sx, sy = positions[i]
                formation.move(px, py, sx, sy)

            # ── Opponent acts ─────────────────────────────────────
            self._opponent_act()

            # ── Simulation tick ───────────────────────────────────
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
            # Scripted: slide each formation's target toward the player
            advance_step = SCRIPTED_ADVANCE_SPEED / self.tick_rate
            for formation in enemy_formations:
                if not formation.get_living_units():
                    continue
                px, py = formation.get_port()
                sx, sy = formation.get_starboard()
                formation.move(
                    px - advance_step, py,
                    sx - advance_step, sy,
                )
        else:
            # Self-play: query frozen policy with mirrored obs
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

            # Apply shield wall for opponent infantry (no reward tracking)
            for i, formation in enumerate(enemy_formations):
                if i >= MAX_FORMATIONS:
                    break
                living = formation.get_living_units()
                if not living:
                    continue

                # Shield wall for opponent infantry
                if isinstance(living[0], model.Infantry):
                    wall_on = enemy_shield_wall_raw[i] > 0.0
                    for u in living:
                        u.set_wall(wall_on)

                px, py, sx, sy = positions[i]
                # Mirror x: the opponent's policy sees a mirrored world
                px = self.world.width - px
                sx = self.world.width - sx
                formation.move(
                    px, py, sx, sy,
                )

    def _build_opponent_obs(self) -> np.ndarray:
        obs = np.zeros(OBS_SIZE, dtype=np.float32)
        self._encode_team_formations(
            obs, self.enemy_team.get_formations(),
            offset=0, mirror_x=True,
        )
        self._encode_team_formations(
            obs, self.friendly_team.get_formations(),
            offset=MAX_FORMATIONS * FEATURES_PER_FORMATION,
            mirror_x=True,
        )
        self._encode_global_features(
            obs, offset=MAX_FORMATIONS * FEATURES_PER_FORMATION * 2,
        )
        return obs

    # ── Observation builder ───────────────────────────────────────
    def _build_obs(self) -> np.ndarray:
        obs = np.zeros(OBS_SIZE, dtype=np.float32)
        self._encode_team_formations(
            obs, self.friendly_team.get_formations(),
            offset=0, mirror_x=False,
        )
        self._encode_team_formations(
            obs, self.enemy_team.get_formations(),
            offset=MAX_FORMATIONS * FEATURES_PER_FORMATION,
            mirror_x=False,
        )
        self._encode_global_features(
            obs, offset=MAX_FORMATIONS * FEATURES_PER_FORMATION * 2,
        )
        return obs

    def _encode_team_formations(self, obs, formations, offset,
                                 mirror_x=False):
        for i, formation in enumerate(formations):
            if i >= MAX_FORMATIONS:
                break
            living = formation.get_living_units()
            if not living:
                continue

            base = offset + (i * FEATURES_PER_FORMATION)
            unit_type_name = UNIT_CLASS_TO_TYPE.get(
                type(living[0]), "infantry"
            )

            xs = [u.get_x() for u in living]
            ys = [u.get_y() for u in living]
            avg_x = sum(xs) / len(xs)
            avg_y = sum(ys) / len(ys)
            min_x = min(xs)
            max_x = max(xs)
            min_y = min(ys)
            max_y = max(ys)

            if mirror_x:
                avg_x = self.world.width - avg_x
                min_x, max_x = (
                    self.world.width - max_x,
                    self.world.width - min_x,
                )

            obs[base + 0] = 1.0                                        # alive
            obs[base + 1] = UNIT_TYPE_ENCODING.get(unit_type_name, 0.0) # type
            obs[base + 2] = min(len(living) / 50.0, 1.0)               # count
            obs[base + 3] = min_x / self.world.width                    # left edge
            obs[base + 4] = max_x / self.world.width                    # right edge
            obs[base + 5] = avg_y / self.world.height                   # vertical pos
            obs[base + 6] = self.world.get_height_at(avg_x, avg_y) / 5.0
            obs[base + 7] = self.world.get_terrain_at(avg_x, avg_y) / 2.0
            # Shield wall status
            if unit_type_name == "infantry":
                # Check first living unit — all units in a formation share type
                shield_active = living[0].get_wall()
                obs[base + 8] = 1.0 if shield_active else 0.5
            else:
                obs[base + 8] = 0.0
            obs[base + 9] = max_y / self.world.height 
            obs[base + 10] = min_y / self.world.height  
            obs[base + 11] = avg_x / self.world.width
            obs[base + 12] = formation.get_rank_count() / self.world.get_diagonal_length()
            obs[base + 13] = formation.get_rank_width() / self.world.get_diagonal_length()

    def _encode_global_features(self, obs, offset):
        obs[offset + 0] = self.current_step / MAX_STEPS
        obs[offset + 1] = self.world.get_width() / 1000.0
        obs[offset + 2] = self.world.get_height() / 1000.0

    # ── Reward ────────────────────────────────────────────────────
    def _total_hp(self, team) -> int:
        return sum(u.get_hp() for u in team.get_living_units())

    def _avg_distance_to_enemy(self) -> float:
        friendly_formations = self.friendly_team.get_formations()
        enemy_formations = self.enemy_team.get_formations()

        friendly_centres = []
        for f in friendly_formations:
            living = f.get_living_units()
            if living:
                cx = sum(u.get_x() for u in living) / len(living)
                cy = sum(u.get_y() for u in living) / len(living)
                friendly_centres.append((cx, cy))

        enemy_centres = []
        for f in enemy_formations:
            living = f.get_living_units()
            if living:
                cx = sum(u.get_x() for u in living) / len(living)
                cy = sum(u.get_y() for u in living) / len(living)
                enemy_centres.append((cx, cy))

        if not friendly_centres or not enemy_centres:
            return 0.0

        total = 0.0
        count = 0
        for fx, fy in friendly_centres:
            for ex, ey in enemy_centres:
                total += math.hypot(fx - ex, fy - ey)
                count += 1

        return total / count if count else 0.0

    def _compute_reward(self) -> float:
        """Reward components:
        1. HP differential: reward damage to enemy, penalise damage to self.
        2. Approach reward: reward closing distance to enemy formations.
        3. Cohesion: reward keeping units in tight formations.
        4. Terminal bonuses: large reward for victory, penalty for defeat.
        """
        reward = 0.0

        # ── 1. HP differential ────────────────────────────────────
        current_friendly_hp = self._total_hp(self.friendly_team)
        current_enemy_hp = self._total_hp(self.enemy_team)

        friendly_hp_lost = self.prev_friendly_hp - current_friendly_hp
        enemy_hp_lost = self.prev_enemy_hp - current_enemy_hp

        hp_scale = max(self.prev_friendly_hp + self.prev_enemy_hp, 1)
        reward += enemy_hp_lost / hp_scale * 0.5
        reward -= friendly_hp_lost / hp_scale * 0.1

        self.prev_friendly_hp = current_friendly_hp
        self.prev_enemy_hp = current_enemy_hp

        # ── 2. Approach reward ────────────────────────────────────
        avg_distance = self._avg_distance_to_enemy()
        max_distance = self.world.get_diagonal_length()
        distance_closed = self.prev_avg_distance - avg_distance
        reward += distance_closed / max_distance * 2.0
        self.prev_avg_distance = avg_distance

        # ── 3. Cohesion ───────────────────────────────────────────
        for formation in self.friendly_team.get_formations():
            living = formation.get_living_units()
            if not living:
                continue
            cohesion = formation.measure_formation_cohesion()
            # Peak reward at 0.7, falls off symmetrically
            cohesion_reward = max(0.0, 1.0 - abs(cohesion - 0.7) / 0.3)
            reward += cohesion_reward * 0.5
            # penalise diagonals: expected length vs actual
            n = len(living)
            unit_slot = 2 * living[0].get_radius() + model.DEFAULT_UNIT_GAP
            expected_len = n * unit_slot
            actual_len = formation.get_diagonal_length()
            if actual_len > expected_len * 2.0:
                reward -= 1  # per formation, per tick — adds up fast

        # ── 4. Terminal bonuses ───────────────────────────────────
        if self.enemy_team.is_defeated():
            reward += 10.0
        elif self.friendly_team.is_defeated():
            reward -= 10.0

        return reward