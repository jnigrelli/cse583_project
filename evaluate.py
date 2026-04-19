"""
Evaluation script for the trained PPO energy optimisation model.

Runs three kinds of evaluation:
  1. Policy rollout  — cumulative energy reward over episodes
  2. RAPL benchmark  — real hardware energy measurements vs -O0/-O3/-Oz
  3. (optional) Runtime reward rollout via RuntimePointEstimateReward

Usage:
    python evaluate.py                          # uses default MODEL_SAVE_PATH
    python evaluate.py --model my_model         # load a specific .zip
    python evaluate.py --no-rapl                # skip RAPL (e.g. no powercap access)
"""

import os
os.environ["COMPILER_GYM_TRANSIENT_CACHE"] = "/tmp/ajprater_scratch_compiler"

import argparse
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

import gym
import numpy as np
import compiler_gym
import common
from stable_baselines3 import PPO

from energy_reward_ajprater import EnergyReward

# ---------------------------------------------------------------------------
# 1.  Policy rollout evaluation (energy reward)
# ---------------------------------------------------------------------------

def evaluate_policy(model, benchmark=BENCHMARK, n_eval_episodes: int = 3):
    """Roll out the learned policy and report cumulative energy reward."""
    print(f"\n=== Policy Rollout Evaluation  [{benchmark}] ===\n")
    env = make_energy_env(benchmark)
    rewards = []

    for ep in range(n_eval_episodes):
        obs = env.reset()
        episode_reward = 0.0

        for step in range(1, MAX_EPISODE_STEPS + 1):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(int(action))
            episode_reward += reward
            if done:
                break

        rewards.append(episode_reward)
        print(
            f"  episode {ep + 1}/{n_eval_episodes}  |  "
            f"steps={step}  reward={episode_reward:.4f}"
        )

    mean_r = np.mean(rewards)
    std_r  = np.std(rewards)
    print(f"\n  mean={mean_r:.4f}  std={std_r:.4f}")
    env.close()
    return mean_r


# ---------------------------------------------------------------------------
# 2.  RAPL energy benchmark
# ---------------------------------------------------------------------------

def get_rapl_domains():
    domains = {}
    for p in POWERCAP_DIR.glob("*/energy_uj"):
        name = p.parent.name
        if name.count(":") == 1:
            domains[name] = p
    return domains


def read_rapl_energy_uj(domains):
    readings = {}
    for name, path in domains.items():
        try:
            readings[name] = int(path.read_text().strip())
        except (PermissionError, FileNotFoundError, ValueError) as e:
            print(f"  [warn] Could not read {path}: {e}")
    return readings


def measure_energy_uj(func, domains):
    before = read_rapl_energy_uj(domains)
    t0     = time.monotonic()
    result = func()
    t1     = time.monotonic()
    after  = read_rapl_energy_uj(domains)

    delta = {}
    for name in before:
        if name in after:
            diff = after[name] - before[name]
            if diff < 0:
                max_path = domains[name].parent / "max_energy_range_uj"
                try:
                    diff += int(max_path.read_text().strip())
                except Exception:
                    diff = 0
            delta[name] = diff

    return result, delta, t1 - t0


def benchmark_energy(
    model,
    benchmark_uri: str = BENCHMARK,
    n_runs: int = 10,
    warmup_runs: int = 3,
):
    """Compile benchmark at -O0, -O3, -Oz, RL, and RL+O3, then measure RAPL energy."""

    domains = get_rapl_domains()
    if not domains:
        print("[energy bench] ERROR: No RAPL domains found — check /sys/class/powercap/ permissions.")
        return

    print(f"\n=== RAPL Energy Benchmark  [{benchmark_uri}] ===\n")
    print(f"  RAPL domains : {list(domains.keys())}")
    print(f"  Runs         : {n_runs}  (warmup: {warmup_runs})\n")

    extra_link_flags = ["-lm"]
    if "ghostscript" in benchmark_uri:
        extra_link_flags.append("-lz")

    # --- get bitcode from CompilerGym ---
    def get_bitcode(apply_policy: bool):
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

        tmpdir  = tempfile.mkdtemp(prefix="energy_bench_")
        bc_path = os.path.join(tmpdir, "bench.bc")
        env.write_bitcode(bc_path)
        env.close()
        return bc_path

    # --- compile bitcode to binary ---
    def compile_bitcode(bc_path, opt_level=None, label=""):
        tmpdir   = tempfile.mkdtemp(prefix=f"energy_bench_{label}_")
        bin_path = os.path.join(tmpdir, "bench")
        cmd      = ["clang", bc_path]
        if opt_level:
            cmd.append(opt_level)
        cmd.extend(["-o", bin_path] + extra_link_flags)
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  [warn] Compilation failed ({label}): {result.stderr}")
            return None
        return bin_path

    # --- run binary and measure ---
    def run_binary_with_energy(bin_path, n, warmup):
        energy_readings, time_readings = [], []
        for i in range(warmup + n):
            def execute():
                return subprocess.run([bin_path], capture_output=True, timeout=120)
            _, energy_uj, wall_s = measure_energy_uj(execute, domains)
            total_uj = sum(energy_uj.values())
            if i >= warmup:
                energy_readings.append(total_uj)
                time_readings.append(wall_s)
        return energy_readings, time_readings

    def iqr(data):
        return np.percentile(data, 75) - np.percentile(data, 25)

    # --- build variants ---
    print("Extracting unoptimized bitcode...")
    base_bc = get_bitcode(apply_policy=False)
    print("Extracting RL-optimized bitcode...")
    rl_bc   = get_bitcode(apply_policy=True)

    variants = {
        "-O0"  : compile_bitcode(base_bc, opt_level="-O0", label="O0"),
        "-O3"  : compile_bitcode(base_bc, opt_level="-O3", label="O3"),
        "-Oz"  : compile_bitcode(base_bc, opt_level="-Oz", label="Oz"),
        "RL"   : compile_bitcode(rl_bc,   opt_level=None,  label="RL"),
        "RL+O3": compile_bitcode(rl_bc,   opt_level="-O3", label="RL_O3"),
    }

    # --- run measurements ---
    results = {}
    for label, bin_path in variants.items():
        if bin_path is None:
            continue
        print(f"Running {label}  ({warmup_runs} warmup + {n_runs} measured)...")
        energy, walltime = run_binary_with_energy(bin_path, n_runs, warmup_runs)
        results[label] = {
            "energy"      : energy,
            "walltime"    : walltime,
            "med_energy"  : statistics.median(energy),
            "med_walltime": statistics.median(walltime),
            "iqr_energy"  : iqr(energy),
            "iqr_walltime": iqr(walltime),
        }

    if not results:
        print("[energy bench] No variants compiled successfully.")
        return

    # --- report ---
    ref_label = "-O3"
    ref_e = results[ref_label]["med_energy"]   if ref_label in results else None
    ref_t = results[ref_label]["med_walltime"] if ref_label in results else None

    print("\n" + "=" * 80)
    print("ENERGY BENCHMARK RESULTS")
    print(f"Reference for %change: {ref_label}")
    print("=" * 80)
    print(f"{'Variant':<10s} {'Med Energy (µJ)':>16s} {'E vs O3':>9s} "
          f"{'Med Time (s)':>14s} {'T vs O3':>9s} "
          f"{'E IQR (µJ)':>12s} {'T IQR (s)':>11s}")
    print("-" * 80)

    for label in ["-O0", "-O3", "-Oz", "RL", "RL+O3"]:
        if label not in results:
            continue
        r     = results[label]
        e_pct = f"{(r['med_energy']   - ref_e) / ref_e * 100:>+8.2f}%" if ref_e and label != ref_label else ""
        t_pct = f"{(r['med_walltime'] - ref_t) / ref_t * 100:>+8.2f}%" if ref_t and label != ref_label else ""
        print(f"{label:<10s} {r['med_energy']:>16.1f} {e_pct:>9s} "
              f"{r['med_walltime']:>14.4f} {t_pct:>9s} "
              f"{r['iqr_energy']:>12.1f} {r['iqr_walltime']:>11.4f}")

    print("=" * 80)
    print("\nRaw energy samples (µJ):")
    for label, r in results.items():
        print(f"  {label:<10s}: {r['energy']}")
    print()


# ---------------------------------------------------------------------------
# 3.  Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO compiler policy.")
    parser.add_argument(
        "--model", default=MODEL_SAVE_PATH,
        help="Path to saved model (.zip), without extension (default: %(default)s)"
    )
    parser.add_argument(
        "--no-rapl", action="store_true",
        help="Skip RAPL energy benchmarking (e.g. if powercap is unavailable)"
    )
    parser.add_argument(
        "--episodes", type=int, default=3,
        help="Number of episodes for policy rollout evaluation (default: 3)"
    )
    args = parser.parse_args()

    # --- load model ---
    print(f"Loading model from {args.model}.zip ...")
    model = PPO.load(args.model)
    print("Model loaded.\n")

    # --- policy rollout on training benchmark ---
    evaluate_policy(model, benchmark=BENCHMARK, n_eval_episodes=args.episodes)

    # --- RAPL benchmarks ---
    if not args.no_rapl:
        benchmark_energy(model, benchmark_uri=BENCHMARK,      n_runs=10, warmup_runs=3)
        benchmark_energy(model, benchmark_uri=TEST_BENCHMARK, n_runs=10, warmup_runs=3)
    else:
        print("\n[RAPL skipped via --no-rapl]")


if __name__ == "__main__":
    main()
