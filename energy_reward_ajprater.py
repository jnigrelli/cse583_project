from compiler_gym.spaces import Reward
from typing import List
import compiler_gym

# imma try to base this off compiler_gym/spaces/runtime_reward.py

class EnergyReward(Reward):
    def __init__(self):
        super().__init__(
            id="energy",
            observation_spaces=["InstCountDict"],
            default_value=0.0
            min=None,
            max=None,
            # idk these opts they prob fine like this
            default_negates_returns=True,
            deterministic=True,
            platform_dependent=False,
        )
        # values from power testing on my personal laptop
        self.w_int = 36.0
        self.w_float = 38.6
        self.w_branch = 44.6
        self.w_mem = 440.3
        self.curr_energy = None

    # fading reset

    def update(
        self,
        # idk these ops as well in runtime_reward they are "hints" so seems unimportant
        actions,
        observations,
        observation_view
    ):
        # observation_space is "InstCountDict", ref compilergym.com/llvm/index.html it has a list of insts
        inst_counts = observations[0]

        num_ints = inst_counts.get("addCount", 0) + inst_counts.get("subCount", 0) + inst_counts.get("mulCount", 0)
        num_floats = inst_counts.get("faddCount", 0) + inst_counts.get("fsubCount", 0) + inst_counts.get("fmulCount", 0)
        num_mems = inst_counts.get("loadCount", 0) + inst_counts.get("storeCount", 0)
        num_branches = inst_counts.get("brCount", 0)

        energy_func = (num_ints * self.w_int) + (num_floats * self.w_float) + (num_mems * self.w_mem) + (num_branches + self.w_branch)

        if self.curr_energy is None:
            self.curr_energy = energy_func
            return 0.0
        
        reward = self.curr_energy - energy_func
        self.curr_energy = energy_func
        return reward

