from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from gym_wrapper import BattleEnv
from curriculum_callback import CurriculumSelfPlayCallback


def make_env(stage):
    def _init():
        return BattleEnv(curriculum_stage=stage)
    return _init


if __name__ == "__main__":
    n_envs = 5 #using ryzen 5 3600 with 6 cores, one left for gpu orchestration etc

    env = SubprocVecEnv([make_env(0) for _ in range(n_envs)])

    model = RecurrentPPO(
        "MlpLstmPolicy",
        env,
        n_steps=256,
        batch_size=256,
        n_epochs=5,
        verbose=1,
        tensorboard_log="./tb_logs/",
        device="cuda",
    )

    callback = CurriculumSelfPlayCallback(
        save_path="./opponent_snapshots",
        promotion_threshold=0.9,
        window_size=100,
        max_stage=3,
    )

    model.learn(
        total_timesteps=20_000_000,
        callback=callback,
        progress_bar=True,
    )

    model.save("battle_agent_final")
    print("Training complete.")