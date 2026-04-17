import numpy as np
import torch

class Basic_Problem:
    def eval(self, x):    
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        x = self.transform_solution(x)
        if x.ndim == 1:  # x is a single individual
            y=self.func(x.reshape(1, -1))
            return y
        elif x.ndim == 2:  # x is a whole population
            y=self.func(x)
            return y
        else:
            y=self.func(x.reshape(-1, x.shape[-1]))
            return y

    def func(self, x):
        raise NotImplementedError
    
    def transform_solution(self, x):
        n = x.ndim
        if n == 1:
            x = x[np.newaxis, :]
        ub = self.ub
        lb = self.lb
        x = np.clip(x, 0, 1)
        x = lb + x * (ub - lb)
        return x[0] if n == 1 else x
    
    def eval_moo(self, x):
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        x = self.transform_solution(x)
        if x.ndim == 1:  # x is a single individual
            return self.func_moo(x.reshape(1, -1))
            
        elif x.ndim == 2:  # x is a whole population
            return self.func_moo(x)
        else:
            return self.func_moo(x.reshape(-1, x.shape[-1]))
    
    def func_moo(self, x):
        raise NotImplementedError


class CECConstrainedProblem(Basic_Problem):
    def __init__(self, problem_cls, dim, changeable=True):
        self.cec_problem = problem_cls(dim, changeable)
        self.dim = dim
        self.lb = float(self.cec_problem.min_tensor)
        self.ub = float(self.cec_problem.max_tensor)
        self.cons_num = self.cec_problem.cons_num

    def func(self, x):
        x_torch = torch.tensor(x, dtype=torch.float64)
        obres, cons = self.cec_problem.calculate_result(x_torch)
        costs = obres.detach().numpy().astype(np.float64)
        penalties = cons.detach().numpy().astype(np.float64).T  # (cons_num, NP) -> (NP, cons_num)
        return costs, penalties
    