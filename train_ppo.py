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

import gym
import numpy as np
import compiler_gym
from compiler_gym.wrappers import RuntimePointEstimateReward
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from pathlib import Path

# AJP : import custom energy reward 
from energy_reward_ajprater import EnergyReward
from perf_reward import PerfReward

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BENCHMARK         = "benchmark://cbench-v1/sha"
TEST_BENCHMARK    = "benchmark://cbench-v1/ghostscript"
MAX_EPISODE_STEPS = 100
MODEL_SAVE_PATH   = "ppo_energy_sha"   # SB3 appends .zip automatically


# ---------------------------------------------------------------------------
# Wrappers
# ---------------------------------------------------------------------------

class ActionCastWrapper(gym.ActionWrapper):
    """Cast numpy.int64 actions to plain int for CompilerGym compatibility."""
    def action(self, action):
        return int(action)


# ---------------------------------------------------------------------------
# 1.  Environment factories
# ---------------------------------------------------------------------------

def make_train_env():
    env = compiler_gym.make(
        "llvm-v0",
        observation_space="Autophase",
        benchmark=BENCHMARK,
    )
    env.reward.add_space(EnergyReward())
    env.reward_space = "energy"
    return ActionCastWrapper(env)


def make_runtime_train_env():
    env = compiler_gym.make(
        "llvm-v0",
        observation_space="Autophase",
        benchmark=BENCHMARK,
    )
    env.reward.add_space(PerfReward())
    env.reward_space = "perf"
    return ActionCastWrapper(env)


def make_eval_env(runtime_count: int = 30):
    env = compiler_gym.make(
        "llvm-v0",
        observation_space="Autophase",
        reward_space="IrInstructionCountOz",
        benchmark=BENCHMARK,
    )
    env = RuntimePointEstimateReward(
        env, runtime_count=runtime_count, warmup_count=5
    )
    return ActionCastWrapper(env)


# ---------------------------------------------------------------------------
# 2.  Logging callback
# ---------------------------------------------------------------------------

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
            print(
                f"[train]  episode finished  |  "
                f"steps={self._episode_steps}  "
                f"reward={self._episode_reward:.4f}"
            )
            self._episode_reward = 0.0
            self._episode_steps  = 0
        return True


# ---------------------------------------------------------------------------
# 3.  Train
# ---------------------------------------------------------------------------

def train(total_timesteps: int = 20_000, algo: str = "ppo"):
    vec_env = DummyVecEnv([make_train_env])

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


# ---------------------------------------------------------------------------
# 4.  Entry point — train and save only
# ---------------------------------------------------------------------------

def main():
    print(f"=== Training PPO on {BENCHMARK} ===\n")
    print("=== Phase 1: Energy reward training ===\n")
    model = train(total_timesteps=50_000, algo="ppo")

    model.save(MODEL_SAVE_PATH)
    print(f"\nModel saved to {MODEL_SAVE_PATH}.zip")

    ''' TODO : Phase 2
    # Phase 2: fine-tune on runtime with the pretrained policy
    print("\n=== Phase 2: Runtime fine-tuning ===\n")
    runtime_env = DummyVecEnv([make_runtime_train_env])
    model.set_env(runtime_env)
    model.ent_coef = 0.001
    model.learning_rate = 1e-4
    model.learn(total_timesteps=5_000, callback=EpisodeLogCallback())
    runtime_env.close()
    model.save(MODEL_SAVE_PATH)
    print(f"Fine-tuned model saved to {MODEL_SAVE_PATH}.zip")
    '''


if __name__ == "__main__":
    main()
