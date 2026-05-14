from stable_baselines3.common.callbacks import BaseCallback
from sb3_contrib import RecurrentPPO
import numpy as np
import os


class CurriculumSelfPlayCallback(BaseCallback):
    """
    Combined curriculum learning + self-play callback.

    State machine per curriculum stage:
        vs SCRIPTED  ──(90% win rate)──>  vs SELF-PLAY  ──(90% win rate)──>  next stage

    The full progression:
        Stage 0 vs scripted  →  Stage 0 vs self  →
        Stage 1 vs scripted  →  Stage 1 vs self  →
        Stage 2 vs scripted  →  Stage 2 vs self  →
        Stage 3 vs scripted  →  Stage 3 vs self  →  done

    Parameters
    ----------
    save_path : str
        Directory to save opponent snapshots.
    promotion_threshold : float
        Win rate needed to advance (0.0 to 1.0).
    window_size : int
        Number of episodes to measure win rate over.
    max_stage : int
        Highest curriculum stage.
    verbose : int
        Print transitions if >= 1.

    stages:
    0: basic movement
    1: terrain speed modifiers
    2: forest concealment
    3: height + line of sight
    """

    def __init__(self, save_path: str = "./opponent_snapshots",
                 promotion_threshold: float = 0.7,
                 window_size: int = 100,
                 max_stage: int = 3,
                 verbose: int = 1):
        super().__init__(verbose)
        self.save_path = save_path
        self.promotion_threshold = promotion_threshold
        self.window_size = window_size
        self.max_stage = max_stage

        self.current_stage = 0
        self.vs_self_play = False  # False = vs scripted, True = vs self
        self.episode_outcomes = []
        self.fully_complete = False

        os.makedirs(save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.fully_complete:
            return True

        # ── Track episode outcomes ────────────────────────────────
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])

        for i, done in enumerate(dones):
            if done:
                info = infos[i]
                won = info.get("won", False)
                self.episode_outcomes.append(won)

        # ── Check for promotion ───────────────────────────────────
        if len(self.episode_outcomes) >= self.window_size:
            recent = self.episode_outcomes[-self.window_size:]
            win_rate = sum(recent) / len(recent)

            if win_rate >= self.promotion_threshold:
                self._promote(win_rate)

        return True

    def _promote(self, win_rate: float):
        """Handle state transition."""
        self.episode_outcomes.clear()

        if not self.vs_self_play:
            snapshot_path = os.path.join(
                self.save_path,
                f"opponent_stage{self.current_stage}"
            )
            self.model.save(snapshot_path)

            # Pass the PATH, not the model object
            self.training_env.env_method("set_opponent_model", snapshot_path)
            self.vs_self_play = True

            if self.verbose >= 1:
                print(f"\n{'='*60}")
                print(f"  STAGE {self.current_stage}: "
                    f"scripted → SELF-PLAY  (win rate: {win_rate:.0%})")
                print(f"  Opponent snapshot saved: {snapshot_path}")
                print(f"{'='*60}\n")

        else:
            if self.current_stage < self.max_stage:
                self.current_stage += 1
                self.vs_self_play = False

                self.training_env.env_method(
                    "set_curriculum_stage", self.current_stage
                )
                # None is fine — it's a simple type that pickles trivially
                self.training_env.env_method("set_opponent_model", None)

                if self.verbose >= 1:
                    print(f"\n{'='*60}")
                    print(f"  CURRICULUM: Advanced to STAGE {self.current_stage} "
                        f"vs scripted  (win rate: {win_rate:.0%})")
                    stages = [
                        "basic movement",
                        "terrain speed modifiers",
                        "forest concealment",
                        "height + line of sight",
                    ]
                    print(f"  New mechanic: {stages[self.current_stage]}")
                    print(f"{'='*60}\n")

            else:
                self.fully_complete = True

                if self.verbose >= 1:
                    print(f"\n{'='*60}")
                    print(f"  CURRICULUM COMPLETE!")
                    print(f"  Agent beat self-play on all "
                        f"{self.max_stage + 1} stages")
                    print(f"{'='*60}\n")