"""
BattleEnv — Gymnasium wrapper around the Team-based World simulator.

Observation (per formation, zero-padded to MAX_FORMATIONS):
  Index   Feature
  ──────  ─────────────────────────────────────────
  0-1     Formation port (x, y), normalised
  2-3     Formation starboard (x, y), normalised
  4       Alive count / starting count
  5       Average HP / MAX_UNIT_HP
  6       Terrain under centroid (0=plains, 0.5=forest, 1=water)
  7       Height under centroid / 5

  Repeated for: friendly formations (0..MAX_F-1),
                enemy formations   (MAX_F..2*MAX_F-1)

  Then 2 global features:
    - friendly alive ratio
    - enemy alive ratio

  Total obs size: 2 * MAX_FORMATIONS * 8 + 2

Action space:
  MultiDiscrete — one 9-way direction per friendly formation.
  0=stay, 1=N, 2=NE, 3=E, 4=SE, 5=S, 6=SW, 7=W, 8=NW
  Padded to MAX_FORMATIONS (extra actions ignored).
"""

from __future__ import annotations

import math
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from model import (
    World, Team, Formation,
    Infantry, Archer, Cavalry, Terrain, Unit,
)

TERRAIN_OBS = {
    Terrain.PLAINS: 0.0,
    Terrain.WATER:  1.0,
    Terrain.FOREST: 0.5,
}

MAX_UNIT_HP = 120  # Cavalry, the highest


class BattleEnv(gym.Env):

    metadata = {"render_modes": ["human"], "render_fps": 10}

    MAX_FORMATIONS = 4
    MOVE_STEP = 5.0

    # Reward weights
    KILL_REWARD = 1.0
    DEATH_PENALTY = 1.0
    WIN_BONUS = 10.0
    LOSE_PENALTY = -10.0
    STEP_PENALTY = -0.01

    def __init__(
        self,
        width: int = 200,
        height: int = 200,
        tick_rate: int = 10,
        ticks_per_step: int = 5,
        max_steps: int = 500,
        curriculum_stage: int = 0,
        min_formations: int = 1,
        max_formations: int = 3,
        min_units: int = 10,
        max_units: int = 50,
        opponent_model=None,
        render_mode: str | None = None,
    ):
        super().__init__()
        self.width = width
        self.height = height
        self.tick_rate = tick_rate
        self.ticks_per_step = ticks_per_step
        self.max_steps = max_steps
        self.curriculum_stage = curriculum_stage
        self.min_formations = min_formations
        self.max_formations = max_formations
        self.min_units = min_units
        self.max_units = max_units
        self.opponent_model = opponent_model
        self.render_mode = render_mode

        # Observation: per-formation features × 2 teams + 2 global
        self.FEATURES_PER_FORMATION = 8
        obs_size = 2 * self.MAX_FORMATIONS * self.FEATURES_PER_FORMATION + 2
        self.OBS_SIZE = obs_size
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )

        # Action: one 9-way direction per friendly formation
        self.action_space = spaces.MultiDiscrete(
            [9] * self.MAX_FORMATIONS
        )

        # State
        self.world: World | None = None
        self.friendly_team: Team | None = None
        self.enemy_team: Team | None = None
        self._step_count = 0
        self._initial_friendly = 0
        self._initial_enemy = 0
        self._prev_friendly_alive = 0
        self._prev_enemy_alive = 0
        self.opponent_lstm_states = None

    # ==================================================================
    # Reset
    # ==================================================================

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.world = World(
            width=self.width,
            height=self.height,
            tick_rate=self.tick_rate,
            curriculum_stage=self.curriculum_stage,
        )

        self.friendly_team = Team(team_id=False)
        self.enemy_team = Team(team_id=True)

        self.world.populate_team(
            self.friendly_team,
            min_formations=self.min_formations,
            max_formations=self.max_formations,
            min_units=self.min_units,
            max_units=self.max_units,
        )
        self.world.populate_team(
            self.enemy_team,
            min_formations=self.min_formations,
            max_formations=self.max_formations,
            min_units=self.min_units,
            max_units=self.max_units,
        )

        self._step_count = 0
        self._initial_friendly = len(self.friendly_team.get_all_units())
        self._initial_enemy = len(self.enemy_team.get_all_units())
        self._prev_friendly_alive = self._initial_friendly
        self._prev_enemy_alive = self._initial_enemy
        self.opponent_lstm_states = None

        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    # ==================================================================
    # Step
    # ==================================================================

    def step(self, action):
        self._step_count += 1

        # Apply agent action to friendly formations
        self._apply_action(self.friendly_team, action, mirror=False)

        # Apply opponent action to enemy formations
        self._opponent_policy()

        # Tick the simulation
        for _ in range(self.ticks_per_step):
            self.world.tick()

        # Count alive
        friendly_alive = len(self.friendly_team.get_living_units())
        enemy_alive = len(self.enemy_team.get_living_units())

        # Reward
        reward = self._compute_reward(friendly_alive, enemy_alive)

        self._prev_friendly_alive = friendly_alive
        self._prev_enemy_alive = enemy_alive

        # Termination
        terminated = friendly_alive == 0 or enemy_alive == 0
        truncated = self._step_count >= self.max_steps

        obs = self._get_observation()
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    # ==================================================================
    # Action interpretation
    # ==================================================================

    # Direction vectors for 9-way movement (0 = stay)
    _DIRS = [
        (0, 0),    # 0: stay
        (0, -1),   # 1: N
        (1, -1),   # 2: NE
        (1, 0),    # 3: E
        (1, 1),    # 4: SE
        (0, 1),    # 5: S
        (-1, 1),   # 6: SW
        (-1, 0),   # 7: W
        (-1, -1),  # 8: NW
    ]

    def _apply_action(self, team: Team, action, mirror: bool = False):
        formations = team.get_formations()
        for i, formation in enumerate(formations):
            if i >= self.MAX_FORMATIONS:
                break

            d = int(action[i])
            dx, dy = self._DIRS[d]

            if mirror:
                dx = -dx  # flip x for opponent's perspective

            px, py = formation.get_port()
            sx, sy = formation.get_starboard()

            step = self.MOVE_STEP
            new_px = np.clip(px + dx * step, 0, self.width)
            new_py = np.clip(py + dy * step, 0, self.height)
            new_sx = np.clip(sx + dx * step, 0, self.width)
            new_sy = np.clip(sy + dy * step, 0, self.height)

            formation.move(new_px, new_py, new_sx, new_sy, self.tick_rate)

    # ==================================================================
    # Opponent
    # ==================================================================

    def _opponent_policy(self):
        if self.opponent_model is not None:
            obs = self._get_mirror_observation()
            action, self.opponent_lstm_states = self.opponent_model.predict(
                obs,
                state=self.opponent_lstm_states,
                episode_start=np.array([False]),
                deterministic=False,
            )
            self._apply_action(self.enemy_team, action, mirror=True)
        else:
            self._heuristic_opponent()

    def _heuristic_opponent(self):
        """Simple: each enemy formation walks toward friendly centroid."""
        friendly_living = self.friendly_team.get_living_units()
        if not friendly_living:
            return

        fx = np.mean([u.get_x() for u in friendly_living])
        fy = np.mean([u.get_y() for u in friendly_living])

        for formation in self.enemy_team.get_formations():
            px, py = formation.get_port()
            sx, sy = formation.get_starboard()
            cx = (px + sx) / 2
            cy = (py + sy) / 2

            dx = fx - cx
            dy = fy - cy
            mag = math.hypot(dx, dy) or 1.0

            move_x = dx / mag * self.MOVE_STEP
            move_y = dy / mag * self.MOVE_STEP

            formation.move(
                np.clip(px + move_x, 0, self.width),
                np.clip(py + move_y, 0, self.height),
                np.clip(sx + move_x, 0, self.width),
                np.clip(sy + move_y, 0, self.height),
                self.tick_rate,
            )

    # ==================================================================
    # Observation
    # ==================================================================

    def _get_formation_features(self, formation: Formation,
                                 starting_count: int) -> np.ndarray:
        """8 features for one formation, all normalised to [0, 1]."""
        feats = np.zeros(self.FEATURES_PER_FORMATION, dtype=np.float32)

        living = formation.get_living_units()
        if not living:
            return feats

        px, py = formation.get_port()
        sx, sy = formation.get_starboard()

        feats[0] = px / self.width
        feats[1] = py / self.height
        feats[2] = sx / self.width
        feats[3] = sy / self.height
        feats[4] = len(living) / max(starting_count, 1)
        feats[5] = np.mean([u.get_hp() for u in living]) / MAX_UNIT_HP

        # Terrain and height under centroid
        cx = np.mean([u.get_x() for u in living])
        cy = np.mean([u.get_y() for u in living])
        terrain = self.world.get_terrain_at(cx, cy)
        feats[6] = TERRAIN_OBS.get(terrain, 0.0)
        feats[7] = self.world.get_height_at(cx, cy) / 5.0

        return feats

    def _build_team_obs(self, team: Team, starting_count: int) -> np.ndarray:
        """Observation block for one team: MAX_FORMATIONS × 8 features."""
        block = np.zeros(
            self.MAX_FORMATIONS * self.FEATURES_PER_FORMATION,
            dtype=np.float32,
        )
        formations = team.get_formations()
        per_formation_start = starting_count // max(len(formations), 1)

        for i, f in enumerate(formations):
            if i >= self.MAX_FORMATIONS:
                break
            start = i * self.FEATURES_PER_FORMATION
            block[start:start + self.FEATURES_PER_FORMATION] = \
                self._get_formation_features(f, per_formation_start)

        return block

    def _get_observation(self) -> np.ndarray:
        obs = np.zeros(self.OBS_SIZE, dtype=np.float32)

        friendly_block = self._build_team_obs(
            self.friendly_team, self._initial_friendly
        )
        enemy_block = self._build_team_obs(
            self.enemy_team, self._initial_enemy
        )

        f_len = self.MAX_FORMATIONS * self.FEATURES_PER_FORMATION
        obs[0:f_len] = friendly_block
        obs[f_len:2 * f_len] = enemy_block

        # Global ratios
        friendly_alive = len(self.friendly_team.get_living_units())
        enemy_alive = len(self.enemy_team.get_living_units())
        obs[-2] = friendly_alive / max(self._initial_friendly, 1)
        obs[-1] = enemy_alive / max(self._initial_enemy, 1)

        return obs

    def _get_mirror_observation(self) -> np.ndarray:
        """
        Observation from the enemy's perspective.
        Enemy sees itself as 'friendly' (first block) and us as 'enemy'.
        X coordinates are mirrored.
        """
        obs = np.zeros(self.OBS_SIZE, dtype=np.float32)

        enemy_block = self._build_team_obs(
            self.enemy_team, self._initial_enemy
        )
        friendly_block = self._build_team_obs(
            self.friendly_team, self._initial_friendly
        )

        f_len = self.MAX_FORMATIONS * self.FEATURES_PER_FORMATION

        # Mirror x coordinates (indices 0, 2 within each formation's 8 features)
        for i in range(self.MAX_FORMATIONS):
            base = i * self.FEATURES_PER_FORMATION
            enemy_block[base + 0] = 1.0 - enemy_block[base + 0]
            enemy_block[base + 2] = 1.0 - enemy_block[base + 2]
            friendly_block[base + 0] = 1.0 - friendly_block[base + 0]
            friendly_block[base + 2] = 1.0 - friendly_block[base + 2]

        obs[0:f_len] = enemy_block          # opponent sees itself first
        obs[f_len:2 * f_len] = friendly_block
        obs[-2] = len(self.enemy_team.get_living_units()) / max(self._initial_enemy, 1)
        obs[-1] = len(self.friendly_team.get_living_units()) / max(self._initial_friendly, 1)

        return obs

    # ==================================================================
    # Reward
    # ==================================================================

    def _compute_reward(self, friendly_alive: int,
                         enemy_alive: int) -> float:
        reward = self.STEP_PENALTY

        # Kills / deaths since last step
        enemy_killed = self._prev_enemy_alive - enemy_alive
        friendly_killed = self._prev_friendly_alive - friendly_alive

        reward += enemy_killed * self.KILL_REWARD
        reward -= friendly_killed * self.DEATH_PENALTY

        # Win / loss bonus
        if enemy_alive == 0 and friendly_alive > 0:
            reward += self.WIN_BONUS
        elif friendly_alive == 0 and enemy_alive > 0:
            reward += self.LOSE_PENALTY

        return reward

    # ==================================================================
    # Info
    # ==================================================================

    def _get_info(self) -> dict:
        return {
            "step": self._step_count,
            "friendly_alive": len(self.friendly_team.get_living_units()),
            "enemy_alive": len(self.enemy_team.get_living_units()),
        }


# ======================================================================
# Smoke test
# ======================================================================

if __name__ == "__main__":
    env = BattleEnv(width=200, height=200, ticks_per_step=10)
    obs, info = env.reset(seed=42)
    print(f"Obs shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Info: {info}")
    print()

    total_reward = 0.0
    for step in range(500):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if step % 50 == 0 or terminated or truncated:
            print(
                f"  step={step:3d}  reward={reward:+.3f}  "
                f"friendly={info['friendly_alive']}  "
                f"enemy={info['enemy_alive']}"
            )

        if terminated or truncated:
            print(f"Episode ended: terminated={terminated} truncated={truncated}")
            break

    print(f"\nTotal reward: {total_reward:.3f}")
    env.close()

    try:
        from stable_baselines3.common.env_checker import check_env
        env2 = BattleEnv()
        check_env(env2, warn=True)
        print("SB3 check_env passed.")
    except ImportError:
        print("(SB3 not installed — skipping check_env)")
    except Exception as e:
        print(f"check_env failed: {e}")