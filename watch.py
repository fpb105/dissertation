"""
Watch the trained agent fight using the existing tkinter view.

Controls:
    SPACE  - play / pause
    RIGHT  - single step
    R      - new battle
    ESC    - quit
"""

import tkinter as tk
import numpy as np
from sb3_contrib import RecurrentPPO
from model import World, Team
from view import WorldView
from formation_commands import execute_command
from gym_wrapper import (
    BattleEnv, MAX_FORMATIONS, FEATURES_PER_FORMATION,
    OBS_SIZE, NUM_GLOBAL_FEATURES, CMD_ADVANCE,
    UNIT_CLASS_TO_TYPE, UNIT_TYPE_ENCODING, MAX_STEPS
)


MODEL_PATH = "battle_agent_final"
CURRICULUM_STAGE = 3
TICK_RATE = 10
MAP_W, MAP_H = 300, 300


class AgentController:

    def __init__(self, world: World, friendly: Team, enemy: Team,
                 view: WorldView, root: tk.Tk,
                 agent, ms_per_frame: int = 50):
        self.world = world
        self.friendly = friendly
        self.enemy = enemy
        self.view = view
        self.root = root
        self.agent = agent
        self.ms_per_frame = ms_per_frame

        self.running = False
        self.tick_count = 0

        self.friendly_lstm_states = None
        self.friendly_episode_start = np.ones((1,), dtype=bool)
        self.enemy_lstm_states = None
        self.enemy_episode_start = np.ones((1,), dtype=bool)

        self.root.bind('<space>', lambda e: self.toggle_pause())
        self.root.bind('<Escape>', lambda e: self.root.quit())
        self.root.bind('r', lambda e: self.regenerate())
        self.root.bind('<Right>', lambda e: self.step())

    def toggle_pause(self):
        self.running = not self.running
        if self.running:
            self._loop()

    def step(self):
        self.running = False
        self._do_tick()

    # ── Observation builders (matching gym_wrapper exactly) ─────────

    def _encode_formations(self, obs, formations, offset, mirror_x=False):
        for i, formation in enumerate(formations):
            if i >= MAX_FORMATIONS:
                break
            living = formation.get_living_units()
            if not living:
                continue

            base = offset + (i * FEATURES_PER_FORMATION)
            unit_type_name = UNIT_CLASS_TO_TYPE.get(type(living[0]), "infantry")

            xs = [u.get_x() for u in living]
            ys = [u.get_y() for u in living]
            avg_x = sum(xs) / len(xs)
            avg_y = sum(ys) / len(ys)
            min_x = min(xs)
            max_x = max(xs)

            if mirror_x:
                avg_x = self.world.width - avg_x
                min_x, max_x = self.world.width - max_x, self.world.width - min_x

            obs[base + 0] = 1.0
            obs[base + 1] = UNIT_TYPE_ENCODING.get(unit_type_name, 0.0)
            obs[base + 2] = min(len(living) / 50.0, 1.0)
            obs[base + 3] = min_x / self.world.width
            obs[base + 4] = max_x / self.world.width
            obs[base + 5] = self.world.get_height_at(avg_x, avg_y) / 5.0
            obs[base + 6] = self.world.get_terrain_at(avg_x, avg_y) / 2.0

    def _encode_global_features(self, obs, offset):
        obs[offset + 0] = self.tick_count / MAX_STEPS
        obs[offset + 1] = self.world.get_width() / 1000.0
        obs[offset + 2] = self.world.get_height() / 1000.0

    def _build_obs(self) -> np.ndarray:
        obs = np.zeros(OBS_SIZE, dtype=np.float32)
        self._encode_formations(obs, self.friendly.get_formations(), offset=0)
        self._encode_formations(obs, self.enemy.get_formations(),
                                offset=MAX_FORMATIONS * FEATURES_PER_FORMATION)
        self._encode_global_features(obs, offset=MAX_FORMATIONS * FEATURES_PER_FORMATION * 2)
        return obs

    def _build_enemy_obs(self) -> np.ndarray:
        obs = np.zeros(OBS_SIZE, dtype=np.float32)
        self._encode_formations(obs, self.enemy.get_formations(),
                                offset=0, mirror_x=True)
        self._encode_formations(obs, self.friendly.get_formations(),
                                offset=MAX_FORMATIONS * FEATURES_PER_FORMATION,
                                mirror_x=True)
        self._encode_global_features(obs, offset=MAX_FORMATIONS * FEATURES_PER_FORMATION * 2)
        return obs

    # ── Tick logic ─────────────────────────────────────────────────

    CMD_NAMES = ["HOLD", "ADVANCE", "RETREAT", "FLANK_L", "FLANK_R", "FACE_CLOSEST"]

    def _do_tick(self):
        # ── Friendly agent decides ─────────────────────────────────
        obs = self._build_obs()
        obs_batch = np.expand_dims(obs, axis=0)

        action, self.friendly_lstm_states = self.agent.predict(
            obs_batch,
            state=self.friendly_lstm_states,
            episode_start=self.friendly_episode_start,
            deterministic=True,
        )
        self.friendly_episode_start = np.zeros((1,), dtype=bool)
        action = action.squeeze(0)

        friendly_formations = self.friendly.get_formations()
        enemy_formations = self.enemy.get_formations()

        for i in range(len(friendly_formations)):
            execute_command(friendly_formations[i], action[i], TICK_RATE,
                            facing_right=True,
                            enemy_formations=enemy_formations)

        # ── Enemy agent decides (same policy, mirrored obs) ────────
        enemy_obs = self._build_enemy_obs()
        enemy_obs_batch = np.expand_dims(enemy_obs, axis=0)

        enemy_action, self.enemy_lstm_states = self.agent.predict(
            enemy_obs_batch,
            state=self.enemy_lstm_states,
            episode_start=self.enemy_episode_start,
            deterministic=True,
        )
        self.enemy_episode_start = np.zeros((1,), dtype=bool)
        enemy_action = enemy_action.squeeze(0)

        for i in range(len(enemy_formations)):
            execute_command(enemy_formations[i], enemy_action[i], TICK_RATE,
                            facing_right=False,
                            enemy_formations=friendly_formations)

        # ── Log ────────────────────────────────────────────────────
        f_cmds = [self.CMD_NAMES[action[i]] for i in range(len(friendly_formations))]
        e_cmds = [self.CMD_NAMES[enemy_action[i]] for i in range(len(enemy_formations))]

        f_positions = []
        for f in friendly_formations:
            living = f.get_living_units()
            if living:
                ax = sum(u.get_x() for u in living) / len(living)
                ay = sum(u.get_y() for u in living) / len(living)
                f_positions.append(f"({ax:.0f},{ay:.0f})")
            else:
                f_positions.append("(dead)")

        e_positions = []
        for f in enemy_formations:
            living = f.get_living_units()
            if living:
                ax = sum(u.get_x() for u in living) / len(living)
                ay = sum(u.get_y() for u in living) / len(living)
                e_positions.append(f"({ax:.0f},{ay:.0f})")
            else:
                e_positions.append("(dead)")

        print(f"Tick {self.tick_count:4d}  "
              f"F: {list(zip(f_cmds, f_positions))}  "
              f"E: {list(zip(e_cmds, e_positions))}")

        # ── Advance simulation ────────────────────────────────────
        self.world.tick()
        self.tick_count += 1
        self.view.render()

        # ── Status / win check ────────────────────────────────────
        friendly_alive = len(self.friendly.get_living_units())
        enemy_alive = len(self.enemy.get_living_units())
        self.root.title(
            f"Tick {self.tick_count}  |  "
            f"Friendly: {friendly_alive}  |  "
            f"Enemy: {enemy_alive}"
        )

        if self.friendly.is_defeated():
            self.running = False
            self.root.title(
                f"Tick {self.tick_count} — ENEMY WINS  "
                f"(friendly: {friendly_alive}, enemy: {enemy_alive})")
            return True
        if self.enemy.is_defeated():
            self.running = False
            self.root.title(
                f"Tick {self.tick_count} — AGENT WINS  "
                f"(friendly: {friendly_alive}, enemy: {enemy_alive})")
            return True
        return False

    def _loop(self):
        if not self.running:
            return
        over = self._do_tick()
        if not over:
            self.root.after(self.ms_per_frame, self._loop)

    def regenerate(self):
        self.running = False
        self.tick_count = 0

        self.world.teams = []
        self.world.terrain_map = self.world._generate_terrain_map()
        self.world.height_map = self.world._generate_height_map()
        self.world.tiles = np.stack(
            [self.world.terrain_map, self.world.height_map,
             np.zeros((self.world.height, self.world.width), dtype=int)],
            axis=-1
        )

        self.friendly = Team(team_id=False)
        self.enemy = Team(team_id=True)
        self.world.populate_team(self.friendly)
        self.world.populate_team(self.enemy)

        self.friendly_lstm_states = None
        self.friendly_episode_start = np.ones((1,), dtype=bool)
        self.enemy_lstm_states = None
        self.enemy_episode_start = np.ones((1,), dtype=bool)

        self.view.rebuild_terrain()
        self.view.render()
        self.root.title("Regenerated — press Space to watch")


if __name__ == "__main__":
    print(f"Loading model from {MODEL_PATH}...")
    agent = RecurrentPPO.load(MODEL_PATH)
    print("Model loaded.")

    world = World(width=MAP_W, height=MAP_H,
                  tick_rate=TICK_RATE,
                  curriculum_stage=CURRICULUM_STAGE)

    friendly = Team(team_id=False)
    enemy = Team(team_id=True)
    world.populate_team(friendly)
    world.populate_team(enemy)

    root = tk.Tk()
    root.title("Agent Battle — Space=play, Right=step, R=regen, Esc=quit")

    view = WorldView(root, world, tile_size=3)
    controller = AgentController(
        world, friendly, enemy, view, root,
        agent, ms_per_frame=50
    )

    view.render()
    root.mainloop()