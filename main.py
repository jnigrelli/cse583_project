import os
from pathlib import Path

from compiler_gym.wrappers import RuntimePointEstimateReward
import gym
import compiler_gym


POWERCAP_DIR = Path(os.getenv("POWERCAP_DIR", "/sys/class/powercap/"))
DOMAINS = {
    p.parent.name : p
    for p in POWERCAP_DIR.glob("*/energy_uj")
    if p.parent.name.count(":") == 1
}

def main():
    env = gym.make("llvm-v0")
    env = RuntimePointEstimateReward(env, runtime_count=30, warmup_count=0)
    env.reset(benchmark="benchmark://cbench-v1/dijkstra")
    env.render()

    episode_reward = 0
    for i in range(1, 101):
        observation, reward, done, info = env.step(
            env.action_space.sample()
            )
        if done:
            break
        episode_reward += reward
        print(f"Step {i}, quality={episode_reward:.3%}")
        print(len(env.observation["Runtime"]))
        exit(0)

    env.close()



if __name__ == "__main__":
    main()
