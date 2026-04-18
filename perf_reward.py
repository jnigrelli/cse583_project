import subprocess
import tempfile
import os
from compiler_gym.spaces import Reward
from typing import List
import compiler_gym

# imma try to base this off compiler_gym/spaces/runtime_reward.py

class PerfReward(Reward):
    def __init__(self):
        super().__init__(
            name="perf",
            observation_spaces=["Ir"],
            default_value=0.0,
            min=None,
            max=None,
            default_negates_returns=False,
            deterministic=False,
            platform_dependent=True,
        )
        self.prev_energy = None
        self.start_energy = None

    def run_perf(self, ir_string):
        with tempfile.TemporaryDirectory() as tmpdir:
            c_path = os.path.join(tmpdir, "c.out")
            
            # cbench needs this info file thats just "num_runs"
            finfo_path = os.path.join(tmpdir, "_finfo_dataset")
            with open(finfo_path, "w") as f:
                f.write("1\n")

            # NOTE: Some benchmarks might need -lm
            c_cmd = ["clang", "-x", "ir", "-", "-o", c_path]
            # print(f"[1] Compiling with command: {' '.join(c_cmd)}")
            try:
                subprocess.run(c_cmd,input=ir_string,text=True,check=True,capture_output=True)
                # print("[2] Compiled successful")
            except subprocess.CalledProcessError as e:
                # print("COMPILATION FAILED")
                # print(f"{e.stderr}")
                return -50.0

            # print("[3] Runnning Program")
            run = subprocess.run([c_path, finfo_path], capture_output=True, text=True, cwd=tmpdir)
            if run.returncode == 0:
                # print("[4] RUN SUCCESS")
                return 1.0
            else:
                # print("ERROR IN RUNTIME")
                if run.stderr:
                    print(f"{run.stderr}")
                if run.stdout:
                    print(f"{run.stdout}")
                return -1.0

    def reset(self, benchmark, observation_view):
        if not observation_view["IsRunnable"]:
            raise BenchmarkInitError(f"Benchmark is not runnable: {benchmark}")

        if self.start_energy is None:
            self.start_energy = self.run_perf(observation_view["Ir"])

        self.prev_energy = self.start_energy

    def update(
        self,
        actions,
        observations,
        observation_view
    ):
        c_IR = observations[0]

        # FUTURE NOTE: run_perf gonna have to do a lot of heavy lifting here...
        energy = self.run_perf(c_IR)
        
        reward = 100*(self.prev_energy - energy) / (self.prev_energy + 1e-8)
        self.prev_energy = energy
        
        return reward
