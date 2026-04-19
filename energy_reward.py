from compiler_gym.spaces import Reward
from typing import List
import compiler_gym

# imma try to base this off compiler_gym/spaces/runtime_reward.py

class EnergyReward(Reward):
    def __init__(self):
        super().__init__(
            name="energy",
            observation_spaces=["InstCountDict"],
            default_value=0.0,
            min=None,
            max=None,
            default_negates_returns=False,
            deterministic=True,
            platform_dependent=False,
        )
        # values from power testing on my personal laptop
        self.w_int = 36.0
        self.w_float = 38.6
        self.w_branch = 44.6
        self.w_mem = 440.3

        self.prev_energy= None
    
    def compute_energy(self,inst_counts):
        num_ints = inst_counts.get("AddCount", 0) + inst_counts.get("SubCount", 0) + inst_counts.get("MulCount", 0)
        num_floats = inst_counts.get("FAddCount", 0) + inst_counts.get("FSubCount", 0) + inst_counts.get("FMulCount", 0)
        num_mems = inst_counts.get("LoadCount", 0) + inst_counts.get("StoreCount", 0)
        num_branches = inst_counts.get("BrCount", 0)

        return (
            num_ints * self.w_int
            + num_floats * self.w_float
            + num_mems * self.w_mem
            + num_branches * self.w_branch
        )

    def reset(self, benchmark, observation_view):
        inst_counts = observation_view["InstCountDict"]
        self.prev_energy = self.compute_energy(inst_counts)

    def update(
        self,
        actions,
        observations,
        observation_view
    ):
        # observation_space is "InstCountDict", ref compilergym.com/llvm/index.html it has a list of insts
        inst_counts = observations[0]

        energy = self.compute_energy(inst_counts)
        
        reward = 100*(self.prev_energy - energy) / (self.prev_energy + 1e-8)
        self.prev_energy = energy
        
        return reward
