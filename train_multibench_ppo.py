"""
CompilerGym optimisation with PPO (stable-baselines3 1.x + gym 0.21).

Install:
    pip install compiler_gym stable-baselines3==1.8.0

CompilerGym pins gym~=0.21.  SB3 >=2.0 requires gymnasium, so we
must stay on the 1.x line of SB3 to avoid breakage.
"""
# CAEN was yelling at me for using the common linux cache
import os
os.environ["COMPILER_GYM_TRANSIENT_CACHE"] = "/tmp/ajprater_scratch_compiler"
from common import *

import gym
import numpy as np
import compiler_gym
import logging
import random
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from pathlib import Path

# ---------------------------------------------------------------------------
# Benchmarks to train across.  Add / remove entries to control the mix.
# Using cbench-v1 programs that cover a range of computational patterns.
# ---------------------------------------------------------------------------
TRAIN_BENCHMARKS = [
    "cbench-v1/sha",
    "cbench-v1/dijkstra",
    "cbench-v1/susan",
    "cbench-v1/jpeg-c",
    "cbench-v1/jpeg-d",
    "cbench-v1/blowfish",
    "cbench-v1/adpcm",
    "cbench-v1/gsm",
    "cbench-v1/stringsearch",
    "cbench-v1/patricia",
]

class EpisodeLogCallback(BaseCallback):
    """Print cumulative reward at the end of each episode."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._episode_reward = 0.0
        self._episode_steps  = 0

    def _on_step(self) -> bool:
        self._episode_reward += self.locals["rewards"][0]
        self._episode_steps  += 1
        if self.locals["dones"][0]:
            logging.info(
                f"[train]  episode finished  |  "
                f"steps={self._episode_steps}  "
                f"reward={self._episode_reward:.4f}"
            )
            self._episode_reward = 0.0
            self._episode_steps  = 0
        return True


def make_multi_benchmark_env(benchmarks: list[str] = TRAIN_BENCHMARKS):
    """
    Return a factory that creates an energy env which randomly selects a new
    benchmark from `benchmarks` on every reset.  This is the core mechanism
    for preventing overfitting to a single program.
    """
    def _factory():
        # Build the base env using the existing helper from common.py, then
        # wrap it so that reset() swaps the benchmark each episode.
        env = make_energy_env()

        class MultiBenchmarkWrapper(gym.Wrapper):
            def reset(self, **kwargs):
                benchmark = random.choice(benchmarks)
                env.unwrapped.benchmark = benchmark
                obs = self.env.reset(**kwargs)
                logging.debug(f"[env] reset → benchmark={benchmark}")
                return obs

        return MultiBenchmarkWrapper(env)

    return _factory


def train(
    total_timesteps: int = 20_000,
    algo: str = "ppo",
    benchmarks: list[str] = TRAIN_BENCHMARKS,
):
    vec_env = DummyVecEnv([make_multi_benchmark_env(benchmarks)])

    common_kwargs = dict(
        policy="MlpPolicy",
        env=vec_env,
        verbose=1,
    )

    if algo == "ppo":
        model = PPO(
            **common_kwargs,
            n_steps=256,
            batch_size=64,
            n_epochs=4,
            learning_rate=3e-4,
            gamma=1.0,
            ent_coef=0.01,
        )
    elif algo == "dqn":
        model = DQN(
            **common_kwargs,
            learning_rate=1e-3,
            buffer_size=50_000,
            learning_starts=500,
            batch_size=64,
            gamma=1.0,
            exploration_fraction=0.3,
            exploration_final_eps=0.02,
        )
    else:
        raise ValueError(f"Unknown algo: {algo}")

    model.learn(
        total_timesteps=total_timesteps,
        callback=EpisodeLogCallback(),
    )

    vec_env.close()
    return model


def main():
    print(f"=== Training PPO across {len(TRAIN_BENCHMARKS)} benchmarks ===\n")
    print(f"Benchmarks: {TRAIN_BENCHMARKS}\n")

    print("=== Phase 1: Energy reward training ===\n")
    model = train(total_timesteps=10_000, algo="ppo", benchmarks=TRAIN_BENCHMARKS)
    model.save(MODEL_SAVE_PATH)
    print(f"\nModel saved to {MODEL_SAVE_PATH}.zip")

    '''
    print("\n=== Phase 2: Perf fine-tuning ===\n")
    perf_env = DummyVecEnv([make_multi_benchmark_env(TRAIN_BENCHMARKS)])
    model.set_env(perf_env)
    model.ent_coef = 0.001
    model.learning_rate = 1e-4
    model.learn(total_timesteps=5_000, callback=EpisodeLogCallback())
    perf_env.close()
    model.save(MODEL_SAVE_PATH)
    print(f"Fine-tuned model saved to {MODEL_SAVE_PATH}.zip")
    '''


if __name__ == "__main__":
    main()