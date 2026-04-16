"""
CompilerGym optimisation with PPO (stable-baselines3 1.x + gym 0.21).

Install:
    pip install compiler_gym stable-baselines3==1.8.0

CompilerGym pins gym~=0.21.  SB3 >=2.0 requires gymnasium, so we
must stay on the 1.x line of SB3 to avoid breakage.
"""

import gym
import numpy as np
import compiler_gym
from compiler_gym.wrappers import RuntimePointEstimateReward
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

# AJP : import custom energy reward and register it
from energy_reward_ajprater import EnergyReward
compiler_gym.envs.LlvmEnv.reward_space["EnergyRewardAJP"] = EnergyReward

# Add near the top, after imports
class ActionCastWrapper(gym.ActionWrapper):
    """Cast numpy.int64 actions to plain int for CompilerGym compatibility."""

    def action(self, action):
        return int(action)

# ---------------------------------------------------------------------------
# 1.  Environment factory
# ---------------------------------------------------------------------------
# "Autophase" gives a 56-dim feature vector describing the IR — a compact,
# fixed-size observation that works out-of-the-box with MlpPolicy.
#
# For the reward we have two practical choices:
#
#   • IrInstructionCountOz  – fast proxy (counts IR instructions vs -Oz).
#     Good for training because each step is nearly free.
#
#   • RuntimePointEstimateReward – measures *actual* wall-clock runtime.
#     Accurate but very slow (runs the binary N times per step).
#     Best reserved for final evaluation, not inner-loop training.
# ---------------------------------------------------------------------------

BENCHMARK = "benchmark://cbench-v1/sha"
MAX_EPISODE_STEPS = 100


def make_train_env():
    env = compiler_gym.make(
        "llvm-v0",
        observation_space="Autophase",
        # AJP: new reward space for training
        reward_space="EnergyRewardAJP",
        benchmark=BENCHMARK,
    )
    # env = RuntimePointEstimateReward(
    #     env, runtime_count=10, warmup_count=3  # fewer runs to speed up training
    # )
    return ActionCastWrapper(env)    # ← wrap here


def make_runtime_train_env():
    env = compiler_gym.make(
        "llvm-v0",
        observation_space="Autophase",
        reward_space="IrInstructionCountOz",
        benchmark=BENCHMARK,
    )
    env = RuntimePointEstimateReward(
        env, runtime_count=30, warmup_count=5  # more runs = less noise
    )
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
    return ActionCastWrapper(env)    # ← wrap here
# ---------------------------------------------------------------------------
# 2.  Optional: simple logging callback
# ---------------------------------------------------------------------------

class EpisodeLogCallback(BaseCallback):
    """Print cumulative reward at the end of each episode."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._episode_reward = 0.0
        self._episode_steps = 0

    def _on_step(self) -> bool:
        self._episode_reward += self.locals["rewards"][0]
        self._episode_steps += 1
        if self.locals["dones"][0]:
            print(
                f"[train]  episode finished  |  "
                f"steps={self._episode_steps}  "
                f"reward={self._episode_reward:.4f}"
            )
            self._episode_reward = 0.0
            self._episode_steps = 0
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
        # Smaller rollout buffer keeps memory modest and episodes short.
        # Tune these to taste.
    )

    if algo == "ppo":
        model = PPO(
            **common_kwargs,
            n_steps=256,
            batch_size=64,
            n_epochs=4,
            learning_rate=3e-4,
            gamma=1.0,           # no discounting — we care about total IR reduction
            ent_coef=0.01,       # encourage exploration of the ~120 passes
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
# 4.  Evaluate with the trained policy
# ---------------------------------------------------------------------------

def evaluate(model, use_runtime_reward: bool = False, n_eval_episodes: int = 3):
    """Roll out the learned policy and report quality."""

    factory = make_eval_env if use_runtime_reward else make_train_env
    env = factory()

    for ep in range(n_eval_episodes):
        obs = env.reset()
        episode_reward = 0.0

        for step in range(1, MAX_EPISODE_STEPS + 1):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(int(action))
            episode_reward += reward

            if done:
                break

        print(
            f"[eval]  episode {ep + 1}/{n_eval_episodes}  |  "
            f"steps={step}  reward={episode_reward:.4f}"
        )

    env.close()


# ---------------------------------------------------------------------------
# 5.  Entry point
# ---------------------------------------------------------------------------

def main():
    print(f"=== Training PPO on {BENCHMARK} ===\n")
    print("=== Phase 1: Proxy training ===\n")
    model = train(total_timesteps=20_000, algo="ppo")

    ''' TODO : Phase 2
    # Phase 2: fine-tune on runtime with the pretrained policy
    print("\n=== Phase 2: Runtime fine-tuning ===\n")
    runtime_env = DummyVecEnv([make_runtime_train_env])
    model.set_env(runtime_env)
    model.ent_coef = 0.001          # less exploration — refine, don't restart
    model.learning_rate = 1e-4      # smaller steps
    model.learn(total_timesteps=5_000, callback=EpisodeLogCallback())
    runtime_env.close()
    '''

    print("\n=== Evaluating with proxy reward (IR instruction count) ===\n")
    evaluate(model, use_runtime_reward=False)

    # Uncomment below to also measure real wall-clock improvement.
    # Warning: each step runs the compiled binary 30 times — expect minutes.
    # print("\n=== Evaluating with runtime reward ===\n")
    # evaluate(model, use_runtime_reward=True)


if __name__ == "__main__":
    main()
