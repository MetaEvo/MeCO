from typing import Any, Tuple, Dict
from problem_utils import Basic_Problem
import numpy as np
import torch
import time

class Basic_Optimizer:
    def seed(self, seed = None):
        rng_seed = int(time.time()) if seed is None else seed

        self.rng_seed = rng_seed

        self.rng = np.random.RandomState(rng_seed)

    def optimize(self, problem: Basic_Problem): # 针对bbo
        raise NotImplementedError

    def init_population(self,
                        problem: Basic_Problem) -> Any:
        raise NotImplementedError

    def update(self,
               action: Any,
               problem: Basic_Problem) -> Tuple[Any]:
        raise NotImplementedError
    
    def get_results(self) -> Dict:
        raise NotImplementedError
    
class Env:
    def __init__(self,
                 problem,
                 optimizer):
        self.problem = problem
        self.optimizer = optimizer
    
    def reset(self, seed=None):
        self.optimizer.seed(seed)
        return self.optimizer.init_population(self.problem)

    def step(self, action):
        return self.optimizer.update(action, self.problem)
    
    def sample(self, seed=None):
        self.optimizer.seed(seed)
        return self.optimizer.init_sample(self.problem)
    
    def observe(self):
        return self.optimizer.observe()
    
    def get_results(self) -> Dict:
        return self.optimizer.get_results()
    
