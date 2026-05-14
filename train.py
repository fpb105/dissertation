from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from gym_wrapper import BattleEnv
from curriculum_callback import CurriculumSelfPlayCallback
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList


def make_env(stage):
    def _init():
        return BattleEnv(curriculum_stage=stage)
    return _init

if __name__ == "__main__":
    n_envs = 5 

    # Start at whatever stage your trained model reached, not 0
    env = SubprocVecEnv([make_env(3) for _ in range(n_envs)])

    model = RecurrentPPO.load(
        "battle_agent_finetuned.zip",  # path to your pre-trained model
        env=env,
        device="cuda",
        custom_objects={
            "learning_rate": 5e-5,
        },
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=1_000_000,
        save_path="./checkpoints/",
        name_prefix="battle_sim"
    )

    callback = CurriculumSelfPlayCallback(
        save_path="./opponent_snapshots",
        promotion_threshold=0.7,
        window_size=100,
        max_stage=3,
    )

    callback_list = CallbackList([checkpoint_callback, callback])

    model.learn(
        total_timesteps=5_000_000,
        callback=callback_list,
        progress_bar=True,
        reset_num_timesteps=False,  # keeps tensorboard continuous
    )

    model.save("battle_agent_ultra_finetuned.zip")
    print("Fine-tuning complete.")