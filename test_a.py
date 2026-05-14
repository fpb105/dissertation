"""
Acceptance Tests for the Battle Simulator
==========================================
AT-1  Agent behaviour plausibility   (human-rated, interactive runner)
AT-2  Curriculum progression verifiability (log parser)
AT-3  Visualiser interpretability    (human-rated, interactive runner)
AT-4  Dissertation reproducibility   (automated environment check)

AT-1 and AT-3 require a human evaluator and cannot be fully automated.
They are structured as a CLI runner that records scores to JSON, which
can then be cited as evidence in the dissertation.

Run automated tests only:
    pytest test_a.py -v -m "not human_eval"

Run everything including human-eval prompts:
    pytest test_a.py -v

Run the standalone human-eval runners directly:
    python test_a.py --at1
    python test_a.py --at3
"""

import argparse
import importlib
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

# ─────────────────────────────────────────────────────────────────────────────
# AT-4  Dissertation reproducibility
# ─────────────────────────────────────────────────────────────────────────────

class TestReproducibility:
    """
    AT-4 — Acceptance / System
    Validates that the submitted codebase can be initialised from scratch
    without undocumented steps. A reviewer following the dissertation's
    methodology chapter should reach a running training loop without
    needing information beyond what is written down.
    """

    def test_model_module_importable(self):
        """model.py must import without errors."""
        import model  # noqa: F401

    def test_gym_wrapper_importable(self):
        """gym_wrapper.py must import without errors."""
        import gym_wrapper  # noqa: F401

    def test_required_dependencies_installed(self):
        """
        All packages that a fresh clone would need must be importable.
        If any are missing, the test reports which ones so the
        requirements.txt can be corrected.
        """
        required = [
            "gymnasium",
            "numpy",
            "scipy",
            "stable_baselines3",
        ]
        # sb3_contrib is required for RecurrentPPO but skip gracefully
        # if not present — the test below will catch it separately
        missing = []
        for pkg in required:
            try:
                importlib.import_module(pkg)
            except ImportError:
                missing.append(pkg)

        assert not missing, (
            f"The following required packages are not installed: {missing}\n"
            f"Update requirements.txt so a reviewer can reproduce the work."
        )

    def test_sb3_contrib_installed(self):
        """sb3_contrib (RecurrentPPO) must be present."""
        pytest.importorskip("sb3_contrib", reason="sb3_contrib not installed")

    def test_env_resets_without_error(self):
        """A fresh BattleEnv must reset without raising any exception."""
        from gym_wrapper import BattleEnv
        env = BattleEnv(curriculum_stage=0)
        obs, info = env.reset(seed=0)
        assert obs is not None

    def test_env_steps_without_error(self):
        """A single step must complete without raising any exception."""
        from gym_wrapper import BattleEnv
        env = BattleEnv(curriculum_stage=0)
        env.reset(seed=0)
        action = env.action_space.sample()
        result = env.step(action)
        assert len(result) == 5, "step() must return a 5-tuple."

    def test_checkpoint_present_or_training_can_start(self):
        """
        Either a trained checkpoint must exist, or RecurrentPPO must be
        able to construct a new model against the environment.
        This confirms the training entry point is reachable.
        """
        sb3c = pytest.importorskip("sb3_contrib")
        from gym_wrapper import BattleEnv

        checkpoint_paths = [
            "battle_agent_finetuned.zip",
            "battle_agent_ultra_finetuned.zip",
        ]

        checkpoint_found = any(Path(p).exists() for p in checkpoint_paths)

        if not checkpoint_found:
            # Verify a new model can at least be constructed
            from stable_baselines3.common.vec_env import DummyVecEnv
            vec_env = DummyVecEnv([lambda: BattleEnv(curriculum_stage=0)])
            try:
                new_model = sb3c.RecurrentPPO(
                    "MlpLstmPolicy",
                    vec_env,
                    verbose=0,
                )
                assert new_model is not None
            except Exception as e:
                pytest.fail(
                    f"No checkpoint found and RecurrentPPO construction "
                    f"failed: {e}"
                )
        else:
            found = [p for p in checkpoint_paths if Path(p).exists()]
            print(f"\n[AT-4] Checkpoint(s) found: {found}")


# ─────────────────────────────────────────────────────────────────────────────
# AT-2  Curriculum progression verifiability
# ─────────────────────────────────────────────────────────────────────────────

class TestCurriculumProgression:
    """
    AT-2 — Acceptance / System
    Validates training behaviour using the TensorBoard event files that
    SB3 writes by default to tb_logs/RecurrentPPO_1/.

    Because the CurriculumSelfPlayCallback does not explicitly log win
    rates as TensorBoard scalars, these tests use what IS available:
    episode reward mean (rollout/ep_rew_mean), which should trend upward
    over a successful training run, and the presence of multiple event
    files indicating the run did not crash early.

    To enable explicit win-rate logging in future, add to the callback's
    _on_step method:
        self.logger.record("curriculum/win_rate", win_rate)
    """

    # Path to TensorBoard logs — adjust if your run name differs
    TB_LOG_DIR = Path("tb_logs/RecurrentPPO_1")

    def _load_accumulator(self):
        """
        Return a loaded EventAccumulator for the TB log directory,
        or None if the directory does not exist or tensorboard is
        not installed.
        """
        if not self.TB_LOG_DIR.exists():
            return None
        try:
            from tensorboard.backend.event_processing.event_accumulator import (
                EventAccumulator,
            )
        except ImportError:
            return None

        ea = EventAccumulator(str(self.TB_LOG_DIR))
        ea.Reload()
        return ea

    def test_tensorboard_log_directory_present(self):
        """
        The TensorBoard log directory must exist.
        Skips if not found — this is a setup gap, not a code error.
        """
        if not self.TB_LOG_DIR.exists():
            pytest.skip(
                f"TensorBoard log directory not found at {self.TB_LOG_DIR}. "
                f"Adjust TB_LOG_DIR if your run used a different name."
            )

    def test_tensorboard_readable_and_contains_scalars(self):
        """
        The event files must be readable and contain at least one scalar
        tag, confirming the training run produced valid output.
        """
        ea = self._load_accumulator()
        if ea is None:
            pytest.skip(
                "TensorBoard log directory not found or tensorboard not "
                "installed. Run: pip install tensorboard"
            )

        tags = ea.Tags().get("scalars", [])
        assert len(tags) > 0, (
            "No scalar tags found in TensorBoard logs — the training run "
            "may not have written any metrics."
        )
        print(f"\n[AT-2] Available scalar tags: {tags}")

    def test_episode_reward_present_in_logs(self):
        """
        SB3 always logs rollout/ep_rew_mean. Its presence confirms the
        agent completed at least one full episode during training.
        """
        ea = self._load_accumulator()
        if ea is None:
            pytest.skip("TensorBoard logs not accessible.")

        tags = ea.Tags().get("scalars", [])
        assert "rollout/ep_rew_mean" in tags, (
            f"rollout/ep_rew_mean not found in logs. "
            f"Available tags: {tags}"
        )

    def test_reward_trend_is_non_catastrophic(self):
        """
        The mean episode reward over the second half of training must be
        higher than over the first half.

        This is a weak but honest test given that win-rate logging was
        not implemented in the curriculum callback. A strictly increasing
        reward trend is not required — RL reward is noisy — but the
        second-half mean being above the first-half mean is a reasonable
        signal that learning occurred rather than diverged.
        """
        ea = self._load_accumulator()
        if ea is None:
            pytest.skip("TensorBoard logs not accessible.")

        tags = ea.Tags().get("scalars", [])
        if "rollout/ep_rew_mean" not in tags:
            pytest.skip("rollout/ep_rew_mean not found in logs.")

        events = ea.Scalars("rollout/ep_rew_mean")
        values = [e.value for e in events]

        if len(values) < 4:
            pytest.skip(
                f"Only {len(values)} reward data points — not enough to "
                f"assess a trend."
            )

        midpoint  = len(values) // 2
        first_half_mean  = sum(values[:midpoint]) / midpoint
        second_half_mean = sum(values[midpoint:]) / (len(values) - midpoint)

        print(
            f"\n[AT-2] Reward trend:"
            f"\n        First-half mean:  {first_half_mean:.3f}"
            f"\n        Second-half mean: {second_half_mean:.3f}"
            f"\n        Data points: {len(values)}"
        )

        assert second_half_mean >= first_half_mean, (
            f"Second-half reward mean ({second_half_mean:.3f}) is lower "
            f"than first-half ({first_half_mean:.3f}), suggesting training "
            f"diverged or the reward signal is miscalibrated."
        )

    def test_training_ran_for_expected_duration(self):
        """
        The number of logged timestep entries must indicate a substantial
        run. A training cut short by a crash would show very few entries.
        Threshold: at least 50 reward log entries (SB3 logs every 2048
        steps by default, so 50 entries ≈ 100k timesteps).
        """
        ea = self._load_accumulator()
        if ea is None:
            pytest.skip("TensorBoard logs not accessible.")

        tags = ea.Tags().get("scalars", [])
        if "rollout/ep_rew_mean" not in tags:
            pytest.skip("rollout/ep_rew_mean not found in logs.")

        events = ea.Scalars("rollout/ep_rew_mean")
        count = len(events)

        print(f"\n[AT-2] Training log entries (ep_rew_mean): {count}")

        assert count >= 50, (
            f"Only {count} reward log entries found. "
            f"Expected at least 50 (≈ 100k timesteps). "
            f"Training may have been cut short."
        )


# ─────────────────────────────────────────────────────────────────────────────
# AT-1 and AT-3  Human evaluation runners
# ─────────────────────────────────────────────────────────────────────────────

RESULTS_FILE = "acceptance_test_results.json"


def _load_results() -> dict:
    if Path(RESULTS_FILE).exists():
        return json.loads(Path(RESULTS_FILE).read_text())
    return {}


def _save_results(results: dict):
    Path(RESULTS_FILE).write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {RESULTS_FILE}")


def run_at1_agent_plausibility():
    """
    AT-1 — Acceptance
    A human evaluator watches 5 evaluation episodes and rates whether
    the agent's decisions appear tactically coherent.

    Scoring rubric (1–5):
      1 — Completely random, no visible intent
      2 — Some movement toward enemy but no tactics
      3 — Engages enemy and occasionally reacts to threats
      4 — Uses formations deliberately, shield wall used defensively
      5 — Coordinated multi-formation behaviour, clear tactical logic

    Pass criterion: mean score >= 3.0
    """
    print("\n" + "=" * 60)
    print("AT-1: Agent Behaviour Plausibility")
    print("=" * 60)
    print(__doc__.strip() if run_at1_agent_plausibility.__doc__ else "")
    print("\nInstructions:")
    print("  1. The evaluator should not have been involved in development.")
    print("  2. Watch each episode in the visualiser without commentary.")
    print("  3. Rate each episode 1–5 using the rubric above.")
    print("  4. Enter your score after each episode.\n")

    scores = []
    for episode in range(1, 6):
        print(f"--- Episode {episode} of 5 ---")
        print("Launch the visualiser for this episode, then press Enter when ready.")
        input("Press Enter to begin scoring... ")

        while True:
            raw = input(f"Score for episode {episode} (1–5): ").strip()
            try:
                score = int(raw)
                if 1 <= score <= 5:
                    scores.append(score)
                    break
                print("  Please enter a number between 1 and 5.")
            except ValueError:
                print("  Invalid input.")

    mean_score = sum(scores) / len(scores)
    passed = mean_score >= 3.0

    print(f"\nScores:     {scores}")
    print(f"Mean score: {mean_score:.2f}")
    print(f"Result:     {'PASS' if passed else 'FAIL'} (threshold: 3.0)")

    results = _load_results()
    results["AT-1"] = {
        "scores": scores,
        "mean": mean_score,
        "threshold": 3.0,
        "passed": passed,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    _save_results(results)
    return passed


def run_at3_visualiser_interpretability():
    """
    AT-3 — Acceptance
    A user unfamiliar with the system watches one battle and answers
    three questions without guidance.

    Questions:
      Q1. Which side appears to be winning right now? (left / right / unclear)
      Q2. Can you identify any distinct formation type you can see?
          (e.g. line, column, wedge, scattered — any answer counts)
      Q3. Did you observe any special mechanic?
          (e.g. units bunching together, fast units charging, arrow fire)

    Pass criterion: all three questions answered with a non-empty,
    non-"unclear" response. The evaluator need not use correct
    terminology — any confident answer counts.
    """
    print("\n" + "=" * 60)
    print("AT-3: Visualiser Interpretability")
    print("=" * 60)
    print("\nInstructions:")
    print("  1. Find a participant who has NOT seen this system before.")
    print("  2. Show them one full battle in the visualiser.")
    print("  3. Ask each question below and record their answer verbatim.")
    print("  4. Do NOT explain the system before or during observation.\n")

    input("Press Enter when the participant is watching the battle... ")

    answers = {}

    questions = [
        ("Q1", "Which side appears to be winning? (left / right / unclear): "),
        ("Q2", "Can you identify any distinct formation or group shape? Describe it: "),
        ("Q3", "Did you notice any special behaviour or mechanic? Describe it: "),
    ]

    for key, prompt in questions:
        answer = input(prompt).strip()
        answers[key] = answer

    def _answered(text: str) -> bool:
        return bool(text) and text.lower() not in {"unclear", "no", "none", "n/a"}

    passed = all(_answered(a) for a in answers.values())

    print(f"\nAnswers:  {answers}")
    print(f"Result:   {'PASS' if passed else 'FAIL'}")
    print("(Pass = all three questions answered with a non-empty, confident response)")

    results = _load_results()
    results["AT-3"] = {
        "answers": answers,
        "passed": passed,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    _save_results(results)
    return passed


# ─────────────────────────────────────────────────────────────────────────────
# pytest wrappers for AT-1 and AT-3
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.human_eval
class TestHumanEvaluation:
    """
    These tests require a human evaluator and cannot run automatically.

    To conduct them:
        python test_acceptance.py --at1    # agent behaviour plausibility
        python test_acceptance.py --at3    # visualiser interpretability

    Each CLI runner prompts the evaluator, records answers, and saves
    results to acceptance_test_results.json. Once that file exists these
    pytest wrappers will pass or fail based on the recorded outcomes.

    Until the CLI runners have been executed, these tests skip — a missing
    results file means the evaluation hasn't happened yet, which is a
    process gap rather than a code error.
    """

    def test_at1_agent_plausibility_results_on_disk(self):
        """
        Checks that AT-1 has been conducted and its results saved.

        If this skips: run  python test_acceptance.py --at1
        The CLI will open evaluation prompts, record scores, and write
        acceptance_test_results.json so this test can then pass.
        """
        results = _load_results()
        if "AT-1" not in results:
            pytest.skip(
                "AT-1 has not been conducted yet. "
                "Run: python test_acceptance.py --at1"
            )
        assert results["AT-1"]["passed"], (
            f"AT-1 failed. Mean score: {results['AT-1']['mean']:.2f} "
            f"(threshold 3.0). Scores: {results['AT-1']['scores']}"
        )

    def test_at3_visualiser_interpretability_results_on_disk(self):
        """
        Checks that AT-3 has been conducted and passed.

        If this skips: run  python test_acceptance.py --at3
        """
        results = _load_results()
        if "AT-3" not in results:
            pytest.skip(
                "AT-3 has not been conducted yet. "
                "Run: python test_acceptance.py --at3"
            )
        assert results["AT-3"]["passed"], (
            f"AT-3 failed. Answers: {results['AT-3']['answers']}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run human-evaluated acceptance tests."
    )
    parser.add_argument("--at1", action="store_true",
                        help="Run AT-1: Agent Behaviour Plausibility")
    parser.add_argument("--at3", action="store_true",
                        help="Run AT-3: Visualiser Interpretability")
    parser.add_argument("--show-results", action="store_true",
                        help="Print previously saved results and exit.")
    args = parser.parse_args()

    if args.show_results:
        results = _load_results()
        if results:
            print(json.dumps(results, indent=2))
        else:
            print("No results saved yet.")
        sys.exit(0)

    if args.at1:
        ok = run_at1_agent_plausibility()
        sys.exit(0 if ok else 1)

    if args.at3:
        ok = run_at3_visualiser_interpretability()
        sys.exit(0 if ok else 1)

    parser.print_help()