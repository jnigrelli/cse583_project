import gym
import compiler_gym
import time
import subprocess
import logging
from pathlib import Path
from energy_reward_ajprater import EnergyReward
from perf_reward import PerfReward

BENCHMARK         = "benchmark://cbench-v1/sha"
TEST_BENCHMARK    = "benchmark://cbench-v1/qsort"
MAX_EPISODE_STEPS = 100
MODEL_SAVE_PATH   = "ppo_energy_sha"
MEASURE_RAPL      = True

POWERCAP_DIR = Path("/sys/class/powercap/")

class ActionCastWrapper(gym.ActionWrapper):
    """Cast numpy.int64 actions to plain int for CompilerGym compatibility."""
    def action(self, action):
        return int(action)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)

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

class power_meas:
    def __init__(self):
        self.mode = "RAPL" if MEASURE_RAPL else "PERF"
        if self.mode == "RAPL":
            self.rapl_domains = self.get_rapl_domains()
            if not self.rapl_domains:
                logging.error("ERROR: NO RAPL DOMAINS")
    
    def measure(self, cmd, cwd=None):
        if self.mode == "RAPL":
            return self.measure_rapl(cmd,cwd)
        else:
            return self.measure_perf(cmd,cwd)

    def get_rapl_domains(self):
        domains = {}
        for p in POWERCAP_DIR.glob("*/energy_uj"):
            name = p.parent.name
            if name.count(":") == 1:
                domains[name] = p
        return domains

    def read_rapl(self):
        readings = {}
        for name, path in self.rapl_domains.items():
            try:
                readings[name] = int(path.read_text().strip())
            except (PermissionError, FileNotFoundError, ValueError) as e:
                logging.warning(f"[warn] could not read {path}: {e}")
        return readings

    def measure_rapl(self, cmd, cwd=None):
        before = self.read_rapl()
        t0 = time.monotonic()
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=120, cwd=cwd)
            returncode = result.returncode
        except subprocess.TimeoutExpired:
            returncode = -1
        except Exception:
            returncode = -1

        t1 = time.monotonic()
        after = self.read_rapl()

        total_uj = 0
        for name in before:
            if name in after:
                diff = after[name] - before[name]
                if diff < 0:
                    try:
                        max_path = self.rapl_domains[name].parent / "max_energy_range_uj"
                        diff += int(max_path.read_text().strip())
                    except Exception:
                        diff = 0
                total_uj += diff

        return total_uj, (t1-t0), returncode

    def measure_perf(self, cmd, cwd=None):
        t0 = time.monotonic()

        perf_cmd = ["perf", "stat", "-x", ",", "-e", "power/energy-pkg/"] + cmd

        try:
            result = subprocess.run(perf_cmd, capture_output=True, text=True, timeout=120, cwd=cwd)
            t1 = time.monotonic()

            total_joules = 0.0
            for line in result.stderr.splitlines():
                if "power/energy-pkg/" in line or "Joules" in line:
                    parts = line.split(',')
                    if parts[0] and parts[0] != '<not supported>':
                        total_joules = float(parts[0])
                        break
            total_uj = int(total_joules * 1_000_000)
            return total_uj, (t1 - t0), result.returncode
        except subprocess.TimeoutExpired:
            return 0, 120.0, -1
        except Exception:
            return 0, (time.monotonic()-t0), -1
