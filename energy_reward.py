"""
Hardware energy reward for CompilerGym LLVM environments (Linux RAPL / powercap).

Uses cumulative microjoule counters under ``/sys/class/powercap/*/energy_uj`` (same
layout as ``intel-rapl``). Reward is the *negative* of the energy consumed during
each step (so standard RL maximization minimizes package energy). The Runtime
observation is requested so the compiled benchmark actually runs during the step;
the counter delta includes all CPU work on that RAPL domain during that window
(compiler service, benchmark runs, OS noise).

Requires a readable powercap tree (often root, or adjusted permissions on HPC).
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence, Union

logger = logging.getLogger(__name__)

from compiler_gym.datasets import Benchmark
from compiler_gym.envs.llvm import LlvmEnv
from compiler_gym.errors import BenchmarkInitError, ServiceError
from compiler_gym.spaces.reward import Reward
from compiler_gym.util.gym_type_hints import ActionType, ObservationType, RewardType
from compiler_gym.views.observation import ObservationView
from compiler_gym.wrappers import CompilerEnvWrapper


class RaplNotAvailableError(RuntimeError):
    """Raised when no RAPL ``energy_uj`` files are found or they are unreadable."""


# Command line of the CompilerGym LLVM backend (see service cache / connection).
_COMPILER_GYM_SERVICE_TOKEN = "compiler_gym-llvm-service"


def cleanup_stale_compiler_gym_processes(
    *,
    dry_run: bool = False,
    grace_seconds: float = 0.75,
) -> List[int]:
    """
    Send SIGTERM (then SIGKILL if needed) to **stale** CompilerGym LLVM service
    processes owned by the current user.

    This does **not** stop “all other” programs on the machine (that would be
    unsafe on shared clusters). It only matches the dedicated backend binary so
    leftover services from crashed or interrupted runs do not accumulate and do
    not add noise next to RAPL measurements.

    On non-Linux platforms or if ``pgrep`` is missing, returns an empty list.

    :param dry_run: If True, log what would be killed but do not signal.
    :param grace_seconds: Time to wait after SIGTERM before SIGKILL.
    :return: PIDs that were sent a terminating signal.
    """
    if sys.platform != "linux":
        logger.info("cleanup_stale_compiler_gym_processes: skipped (not Linux)")
        return []

    try:
        proc = subprocess.run(
            ["pgrep", "-u", str(os.getuid()), "-f", _COMPILER_GYM_SERVICE_TOKEN],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        logger.warning("cleanup_stale_compiler_gym_processes: pgrep not found")
        return []

    pids: List[int] = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            pid = int(line)
        except ValueError:
            continue
        if pid == os.getpid():
            continue
        pids.append(pid)

    if not pids:
        return []

    terminated: List[int] = []
    for pid in pids:
        if dry_run:
            logger.info("would terminate stale CompilerGym LLVM service pid=%s", pid)
            terminated.append(pid)
            continue
        try:
            os.kill(pid, signal.SIGTERM)
            terminated.append(pid)
        except ProcessLookupError:
            pass
        except PermissionError as e:
            logger.warning("cannot signal pid %s: %s", pid, e)

    if dry_run or not terminated:
        return terminated

    time.sleep(grace_seconds)
    for pid in pids:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            continue
        except OSError:
            continue
        try:
            os.kill(pid, signal.SIGKILL)
            logger.warning("sent SIGKILL to stubborn pid %s", pid)
        except ProcessLookupError:
            pass
        except PermissionError as e:
            logger.warning("cannot SIGKILL pid %s: %s", pid, e)

    return terminated


def discover_energy_uj_paths(
    powercap_dir: Optional[Union[str, Path]] = None,
) -> List[Path]:
    """
    Discover ``energy_uj`` sysfs paths (one file per RAPL domain).

    Mirrors the heuristic in ``main.py``: parent name contains exactly one ``:``
    (e.g. ``intel-rapl:0``, ``intel-rapl:0:0`` is skipped — adjust if you need
    sub-packages only).
    """
    base = Path(
        powercap_dir if powercap_dir is not None else os.environ.get("POWERCAP_DIR", "/sys/class/powercap")
    )
    paths = sorted(
        p
        for p in base.glob("*/energy_uj")
        if p.parent.name.count(":") == 1
    )
    return paths


def read_total_energy_uj(paths: Sequence[Path]) -> int:
    """Sum cumulative energy in microjoules across all given counter files."""
    total = 0
    for p in paths:
        try:
            total += int(p.read_text().strip())
        except OSError as e:
            raise RaplNotAvailableError(
                f"Cannot read RAPL counter {p}: {e}. "
                "Try running on a Linux node with powercap enabled and sufficient permissions."
            ) from e
    return total


class EnergyReward(Reward):
    """
    Incremental reward from RAPL cumulative energy (microjoules).

    At each step, ``reward = -scale * (E_after - E_before)`` where *E* is the sum
    of ``energy_uj`` across discovered domains. ``E_before`` is the reading left
    after the previous step (or after :meth:`reset`).

    Requests the ``Runtime`` observation so the service runs the benchmark;
    ``runtime_count`` / ``warmup_count`` must match the env's runtime settings
    (set by :class:`EnergyPointEstimateReward`).
    """

    def __init__(
        self,
        runtime_count: int,
        warmup_count: int,
        energy_paths: Optional[Sequence[Union[str, Path]]] = None,
        powercap_dir: Optional[Union[str, Path]] = None,
        scale: float = 1e-6,
        name: str = "energy",
        default_value: RewardType = 0,
    ):
        """
        :param runtime_count: Must match ``env.runtime_observation_count``.
        :param warmup_count: Must match ``env.runtime_warmup_runs_count``.
        :param energy_paths: Explicit list of ``energy_uj`` paths; if None, use
            :func:`discover_energy_uj_paths`.
        :param powercap_dir: Used only when ``energy_paths`` is None.
        :param scale: Multiply ``-(delta microjoules)`` by this (default: 1e-6 → joules).
        """
        super().__init__(
            name=name,
            observation_spaces=["Runtime"],
            default_value=default_value,
            min=None,
            max=None,
            default_negates_returns=True,
            deterministic=False,
            platform_dependent=True,
        )
        self.runtime_count = runtime_count
        self.warmup_count = warmup_count  # must match env; used by LLVM service, not here
        self.scale = scale
        if energy_paths is not None:
            self.energy_paths: List[Path] = [Path(p) for p in energy_paths]
        else:
            self.energy_paths = discover_energy_uj_paths(powercap_dir)
        if not self.energy_paths:
            raise RaplNotAvailableError(
                f"No energy_uj counters found under {powercap_dir or '/sys/class/powercap'}. "
                "EnergyReward needs Linux RAPL (powercap) with readable sysfs files."
            )
        self._last_total_uj: Optional[int] = None
        self._current_benchmark: Optional[Union[str, Benchmark]] = None

    def reset(self, benchmark: Union[str, Benchmark], observation_view: ObservationView) -> None:
        if benchmark != self._current_benchmark:
            if not observation_view["IsRunnable"]:
                raise BenchmarkInitError(f"Benchmark is not runnable: {benchmark}")
            self._current_benchmark = benchmark
        self._last_total_uj = read_total_energy_uj(self.energy_paths)

    def update(
        self,
        actions: List[ActionType],
        observations: List[ObservationType],
        observation_view: ObservationView,
    ) -> RewardType:
        del actions
        del observation_view
        runtimes = observations[0]
        if len(runtimes) != self.runtime_count:
            raise ServiceError(
                f"Expected {self.runtime_count} runtimes but received {len(runtimes)}"
            )
        now = read_total_energy_uj(self.energy_paths)
        assert self._last_total_uj is not None
        delta_uj = now - self._last_total_uj
        self._last_total_uj = now
        # Maximize reward => minimize energy consumed this step
        return RewardType(-float(delta_uj) * self.scale)


class EnergyPointEstimateReward(CompilerEnvWrapper):
    """
    LLVM wrapper: register :class:`EnergyReward` and drive Runtime measurements.

    Same integration pattern as :class:`compiler_gym.wrappers.RuntimePointEstimateReward`.
    """

    def __init__(
        self,
        env: LlvmEnv,
        runtime_count: int = 30,
        warmup_count: int = 0,
        energy_paths: Optional[Sequence[Union[str, Path]]] = None,
        powercap_dir: Optional[Union[str, Path]] = None,
        scale: float = 1e-6,
    ):
        super().__init__(env)
        self.env.unwrapped.reward.add_space(
            EnergyReward(
                runtime_count=runtime_count,
                warmup_count=warmup_count,
                energy_paths=energy_paths,
                powercap_dir=powercap_dir,
                scale=scale,
            )
        )
        self.env.unwrapped.reward_space = "energy"
        self.env.unwrapped.runtime_observation_count = runtime_count
        self.env.unwrapped.runtime_warmup_runs_count = warmup_count

    def fork(self) -> "EnergyPointEstimateReward":
        fkd = self.env.fork()
        del fkd.unwrapped.reward.spaces["energy"]
        er: EnergyReward = self.reward.spaces["energy"]
        return EnergyPointEstimateReward(
            env=fkd,
            runtime_count=er.runtime_count,
            warmup_count=er.warmup_count,
            energy_paths=list(er.energy_paths),
            scale=er.scale,
        )


__all__ = [
    "RaplNotAvailableError",
    "cleanup_stale_compiler_gym_processes",
    "discover_energy_uj_paths",
    "read_total_energy_uj",
    "EnergyReward",
    "EnergyPointEstimateReward",
]
