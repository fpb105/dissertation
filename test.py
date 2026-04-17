import time
from gym_wrapper import BattleEnv

for stage in [0, 3]:
    env = BattleEnv(curriculum_stage=stage)
    obs, _ = env.reset()
    
    start = time.time()
    steps = 1000
    for _ in range(steps):
        obs, r, term, trunc, _ = env.step(env.action_space.sample())
        if term or trunc:
            obs, _ = env.reset()
    elapsed = time.time() - start
    
    per_step = elapsed / steps * 1000
    est_total = (elapsed / steps) * 5_000_000 / 3600
    print(f"Stage {stage}: {per_step:.1f}ms/step → ~{est_total:.1f} hours for full run")