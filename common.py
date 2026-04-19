import gym
import compiler_gym
from energy_reward_ajprater import EnergyReward
from perf_reward import PerfReward

BENCHMARK         = "benchmark://cbench-v1/sha"
TEST_BENCHMARK    = "benchmark://cbench-v1/ghostscript"
MAX_EPISODE_STEPS = 100
MODEL_SAVE_PATH   = "ppo_energy_sha"

# caen RAPL path is different
# POWERCAP_DIR = Path("/sys/class/powercap/")

class ActionCastWrapper(gym.ActionWrapper):
    """Cast numpy.int64 actions to plain int for CompilerGym compatibility."""
    def action(self, action):
        return int(action)

def make_energy_env(benchmark=BENCHMARK):
    env = compiler_gym.make(
        "llvm-v0",
        observation_space="Autophase",
        benchmark=benchmark,
    )
    env.reward.add_space(EnergyReward())
    env.reward_space = "energy"
    return ActionCastWrapper(env)

def make_perf_env(benchmark=BENCHMARK):
    env = compiler_gym.make(
        "llvm-v0",
        observation_space="Autophase",
        benchmark=benchmark,
    )
    env.reward.add_space(PerfReward())
    env.reward_space = "perf"
    return ActionCastWrapper(env)

def make_eval_env(runtime_count: int = 30):
    env = compiler_gym.make(
        "llvm-v0",
        observation_space="Autophase",
        benchmark=BENCHMARK,
    )
    # TODO: Runtime used a custom function for Point Estimate 
    # We could either build a new eval env or just use one of the other ones
    return ActionCastWrapper(env)
