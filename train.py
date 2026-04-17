"""
train.py — Self-play training loop with model bank and curriculum stages.

Usage:
    python train.py

Trains a RecurrentPPO agent through self-play.  Early generations
fight the heuristic opponent, then increasingly fight past versions
of themselves from a model bank.
"""

import os
import random
from sb3_contrib import RecurrentPPO
from gym_wrapper import BattleEnv


# ── config ────────────────────────────────────────────────────────────

MODEL_BANK_DIR = "model_bank"
LOG_DIR = "tb_logs"

CURRICULUM = [
    # (stage, generations to train, timesteps per generation)
    (0, 20, 50_000),   # flat map, no terrain effects
    (1, 20, 50_000),   # terrain speed modifiers
    (2, 20, 50_000),   # forest concealment
    (3, 40, 50_000),   # height + LOS
]

# how many generations to train vs heuristic before using the bank
HEURISTIC_WARMUP = 3

# RecurrentPPO hyperparameters
PPO_KWARGS = dict(
    n_steps=256,
    batch_size=64,
    n_epochs=10,
    learning_rate=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    verbose=1,
)


# ── helpers ───────────────────────────────────────────────────────────

def get_bank_models() -> list[str]:
    """Return sorted list of model paths in the bank."""
    if not os.path.isdir(MODEL_BANK_DIR):
        return []
    files = [
        os.path.join(MODEL_BANK_DIR, f)
        for f in os.listdir(MODEL_BANK_DIR)
        if f.endswith(".zip")
    ]
    return sorted(files)


def pick_opponent(generation: int):
    """
    Return an opponent model or None (heuristic).

    First HEURISTIC_WARMUP generations: always heuristic (None).
    After that: 20% chance of heuristic, 80% chance of random past self.
    """
    if generation < HEURISTIC_WARMUP:
        return None

    bank = get_bank_models()
    if not bank:
        return None

    if random.random() < 0.2:
        return None  # occasional heuristic to prevent forgetting

    path = random.choice(bank)
    print(f"  Opponent: {os.path.basename(path)}")
    return RecurrentPPO.load(path)


# ── main loop ─────────────────────────────────────────────────────────

def main():
    os.makedirs(MODEL_BANK_DIR, exist_ok=True)

    model = None
    generation = 0

    for stage, n_gens, timesteps in CURRICULUM:
        print(f"\n{'='*60}")
        print(f"CURRICULUM STAGE {stage} — {n_gens} generations × "
              f"{timesteps:,} timesteps")
        print(f"{'='*60}")

        for g in range(n_gens):
            print(f"\n--- Stage {stage}, generation {generation} ---")

            # pick opponent
            opponent = pick_opponent(generation)
            env = BattleEnv(
                curriculum_stage=stage,
                opponent_model=opponent,
            )

            if model is None:
                # first generation: create from scratch
                model = RecurrentPPO(
                    "MlpLstmPolicy",
                    env,
                    tensorboard_log=LOG_DIR,
                    **PPO_KWARGS,
                )
            else:
                # subsequent generations: keep learning, swap env
                model.set_env(env)

            model.learn(
                total_timesteps=timesteps,
                reset_num_timesteps=False,  # keep global step counter
                tb_log_name=f"stage{stage}",
            )

            # save to bank
            save_path = os.path.join(
                MODEL_BANK_DIR, f"gen_{generation:04d}_stage{stage}"
            )
            model.save(save_path)
            print(f"  Saved: {save_path}")

            generation += 1

    # save final model separately
    model.save("battle_agent_final")
    print(f"\nTraining complete. Final model: battle_agent_final.zip")
    print(f"Model bank: {len(get_bank_models())} checkpoints")


if __name__ == "__main__":
    main()