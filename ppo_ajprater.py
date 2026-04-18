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
import subprocess
import tempfile
import time
import statistics

# AJP : import custom energy reward 
from energy_reward_ajprater import EnergyReward
from perf_reward import PerfReward

# Add near the top, after imports
class ActionCastWrapper(gym.ActionWrapper):
    """Cast numpy.int64 actions to plain int for CompilerGym compatibility."""

    def action(self, action):
        return int(action)

POWERCAP_DIR = Path("/sys/class/powercap/")
 
def get_rapl_domains():
    """Discover RAPL domains (package-level, e.g. intel-rapl:0)."""
    domains = {}
    for p in POWERCAP_DIR.glob("*/energy_uj"):
        name = p.parent.name
        # Package-level domains have exactly one colon (e.g. intel-rapl:0)
        # Sub-domains have two (e.g. intel-rapl:0:1) — skip those to avoid
        # double-counting
        if name.count(":") == 1:
            domains[name] = p
    return domains
 
 
def read_rapl_energy_uj(domains=None):
    """Read current RAPL energy counters in microjoules.
 
    Returns dict: {domain_name: energy_uj}.
    """
    if domains is None:
        domains = get_rapl_domains()
    readings = {}
    for name, path in domains.items():
        try:
            readings[name] = int(path.read_text().strip())
        except (PermissionError, FileNotFoundError, ValueError) as e:
            print(f"  [warn] Could not read {path}: {e}")
    return readings
 
 
def measure_energy_uj(func, domains=None):
    """Run `func()` and return (result, energy_dict_uj, walltime_s).
 
    energy_dict_uj maps each RAPL domain to the energy consumed in
    microjoules during the call.
    """
    if domains is None:
        domains = get_rapl_domains()
    before = read_rapl_energy_uj(domains)
    t0 = time.monotonic()
    result = func()
    t1 = time.monotonic()
    after = read_rapl_energy_uj(domains)
 
    delta = {}
    for name in before:
        if name in after:
            diff = after[name] - before[name]
            # Handle counter wraparound
            if diff < 0:
                max_val_path = domains[name].parent / "max_energy_range_uj"
                try:
                    max_val = int(max_val_path.read_text().strip())
                    diff += max_val
                except Exception:
                    diff = 0  # can't recover; skip
            delta[name] = diff
 
    return result, delta, t1 - t0
 
 
# ---------------------------------------------------------------------------
# Energy benchmarking
# ---------------------------------------------------------------------------
 
def benchmark_energy(
    model,
    benchmark_uri: str = "benchmark://cbench-v1/sha",
    n_runs: int = 10,
    warmup_runs: int = 3,
):
    """Compile a benchmark at -O0 (baseline), -O3, -Oz, and with the
    RL-learned pass sequence, then measure RAPL energy for each.
 
    Reports median energy (µJ), median walltime (s), and percentage
    change relative to -O3.
    """
 
    domains = get_rapl_domains()
    if not domains:
        print("[energy bench] ERROR: No RAPL domains found. "
              "Check /sys/class/powercap/ permissions.")
        return
 
    print(f"[energy bench] RAPL domains found: {list(domains.keys())}")
    print(f"[energy bench] Benchmark: {benchmark_uri}")
    print(f"[energy bench] Runs: {n_runs}  (warmup: {warmup_runs})\n")
 
    # --- Linker flags vary by benchmark ---
    # ghostscript needs zlib; all need math
    extra_link_flags = ["-lm"]
    if "ghostscript" in benchmark_uri:
        extra_link_flags.append("-lz")
 
    # --- helper: get bitcode from CompilerGym env ---
    def get_bitcode(apply_policy: bool):
        """Return path to bitcode file.
 
        If apply_policy is True, roll out the learned policy first.
        Otherwise return the unoptimized IR.
        """
        env = compiler_gym.make(
            "llvm-v0",
            observation_space="Autophase",
            reward_space="IrInstructionCountOz",
            benchmark=benchmark_uri,
        )
        obs = env.reset()
 
        if apply_policy:
            for _ in range(MAX_EPISODE_STEPS):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(int(action))
                if done:
                    break
 
        tmpdir = tempfile.mkdtemp(prefix="energy_bench_")
        bc_path = os.path.join(tmpdir, "bench.bc")
        env.write_bitcode(bc_path)
        env.close()
        return bc_path
 
    # --- helper: compile bitcode to binary with given opt level ---
    def compile_bitcode(bc_path, opt_level=None, label=""):
        """Compile bitcode → executable, optionally with -O3/-Oz/etc.
 
        opt_level: None (no extra opt), "-O3", "-Oz", "-O0", etc.
        Returns binary path or None on failure.
        """
        tmpdir = tempfile.mkdtemp(prefix=f"energy_bench_{label}_")
        bin_path = os.path.join(tmpdir, "bench")
 
        cmd = ["clang", bc_path]
        if opt_level:
            cmd.append(opt_level)
        cmd.extend(["-o", bin_path] + extra_link_flags)
 
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  [warn] Compilation failed ({label}): {result.stderr}")
            return None
        return bin_path
 
    # --- helper: run binary and measure energy ---
    def run_binary_with_energy(bin_path, n, warmup):
        """Execute the binary n+warmup times, return energy and time lists."""
        energy_readings = []
        time_readings = []
 
        for i in range(warmup + n):
            def execute():
                return subprocess.run(
                    [bin_path],
                    capture_output=True,
                    timeout=120,
                )
 
            _, energy_uj, wall_s = measure_energy_uj(execute, domains)
            total_uj = sum(energy_uj.values())
 
            if i >= warmup:
                energy_readings.append(total_uj)
                time_readings.append(wall_s)
 
        return energy_readings, time_readings
 
    # --- IQR helper ---
    def iqr(data):
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        return q3 - q1
 
    # =====================================================================
    # Build all variants
    # =====================================================================
 
    # 1) Get unoptimized bitcode (no RL passes)
    print("Extracting unoptimized bitcode from CompilerGym...")
    base_bc = get_bitcode(apply_policy=False)
 
    # 2) Get RL-optimized bitcode
    print("Extracting RL-optimized bitcode from CompilerGym...")
    rl_bc = get_bitcode(apply_policy=True)
 
    variants = {
        "-O0":   compile_bitcode(base_bc, opt_level="-O0", label="O0"),
        "-O3":   compile_bitcode(base_bc, opt_level="-O3", label="O3"),
        "-Oz":   compile_bitcode(base_bc, opt_level="-Oz", label="Oz"),
        "RL":    compile_bitcode(rl_bc,   opt_level=None,  label="RL"),
        "RL+O3": compile_bitcode(rl_bc,   opt_level="-O3", label="RL_O3"),
    }
 
    # Check all compiled
    for label, path in variants.items():
        if path is None:
            print(f"[energy bench] Could not compile {label}. Skipping it.")
 
    # =====================================================================
    # Run each variant and collect measurements
    # =====================================================================
 
    results = {}
    for label, bin_path in variants.items():
        if bin_path is None:
            continue
        print(f"\nRunning {label}  ({warmup_runs} warmup + {n_runs} measured)...")
        energy, walltime = run_binary_with_energy(bin_path, n_runs, warmup_runs)
        results[label] = {
            "energy": energy,
            "walltime": walltime,
            "med_energy": statistics.median(energy),
            "med_walltime": statistics.median(walltime),
            "iqr_energy": iqr(energy),
            "iqr_walltime": iqr(walltime),
        }
 
    if not results:
        print("[energy bench] No variants compiled successfully. Aborting.")
        return
 
    # =====================================================================
    # Report
    # =====================================================================
 
    # Use -O3 as the reference for percentage change
    ref_label = "-O3"
    ref_e = results[ref_label]["med_energy"] if ref_label in results else None
    ref_t = results[ref_label]["med_walltime"] if ref_label in results else None
 
    print("\n" + "=" * 80)
    print("ENERGY BENCHMARK RESULTS")
    print(f"Reference for %change: {ref_label}")
    print("=" * 80)
 
    header = (f"{'Variant':<10s} {'Med Energy (µJ)':>16s} {'E vs O3':>9s} "
              f"{'Med Time (s)':>14s} {'T vs O3':>9s} "
              f"{'E IQR (µJ)':>12s} {'T IQR (s)':>11s}")
    print(header)
    print("-" * 80)
 
    for label in ["-O0", "-O3", "-Oz", "RL", "RL+O3"]:
        if label not in results:
            continue
        r = results[label]
 
        e_pct = ""
        if ref_e and ref_e > 0 and label != ref_label:
            e_pct = f"{(r['med_energy'] - ref_e) / ref_e * 100:>+8.2f}%"
 
        t_pct = ""
        if ref_t and ref_t > 0 and label != ref_label:
            t_pct = f"{(r['med_walltime'] - ref_t) / ref_t * 100:>+8.2f}%"
 
        print(f"{label:<10s} {r['med_energy']:>16.1f} {e_pct:>9s} "
              f"{r['med_walltime']:>14.4f} {t_pct:>9s} "
              f"{r['iqr_energy']:>12.1f} {r['iqr_walltime']:>11.4f}")
 
    print("=" * 80)
 
    # Raw samples for inspection
    print("\nRaw energy samples (µJ):")
    for label, r in results.items():
        print(f"  {label:<10s}: {r['energy']}")
    print()
 

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
TEST_BENCHMARK = "benchmark://cbench-v1/ghostscript"
MAX_EPISODE_STEPS = 100


def make_train_env():
    env = compiler_gym.make(
        "llvm-v0",
        observation_space="Autophase",
        benchmark=BENCHMARK,
    )
    env.reward.add_space(EnergyReward())
    env.reward_space = "energy"
    # env = RuntimePointEstimateReward(
    #     env, runtime_count=10, warmup_count=3  # fewer runs to speed up training
    # )
    return ActionCastWrapper(env)    # ← wrap here


def make_runtime_train_env():
    env = compiler_gym.make(
        "llvm-v0",
        observation_space="Autophase",
        benchmark=BENCHMARK,
    )
    env.reward.add_space(PerfReward())
    env.reward_space="perf"
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
    # TEMP TESTING CHANGE
    # vec_env = DummyVecEnv([make_runtime_train_env])
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

    factory = make_train_env if use_runtime_reward else make_train_env
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
    model = train(total_timesteps=50_000, algo="ppo")

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
    evaluate(model, use_runtime_reward=True)

    print("\n=== RAPL Energy Benchmark ===\n")
    benchmark_energy(
        model,
        benchmark_uri=BENCHMARK,
        n_runs=10,
        warmup_runs=3,
    )


    benchmark_energy(
        model,
        benchmark_uri=TEST_BENCHMARK,
        n_runs=10,
        warmup_runs=3,
    )



if __name__ == "__main__":
    main()
