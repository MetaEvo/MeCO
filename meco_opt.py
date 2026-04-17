import numpy as np
from typing import Tuple, List
import scipy.stats as stats
from basic_optimizer import Basic_Optimizer

class MeCO_Opt(Basic_Optimizer):
    def __init__(self, 
        init_pop_size:int=50, 
        min_pop_size:int=50,    
        maxfes:int=10000,
        p:float=0.11,           
        h:int=6,                
        ):
        self.init_pop_size = init_pop_size
        self.min_pop_size = min_pop_size
        self.pop_size = init_pop_size
        self.p = p
        self.maxfes = maxfes
        
        self.ub = 1.0
        self.lb = 0.0
        
        self.archive_size = self.pop_size
        self.h = h

        
        self.delta = 1e-3
        self._cost_clip = 1e12
        self._penalty_clip = 1e12
        self._state_clip = 1e6
        self._reward_clip = 10.0
        
    def __str__(self):
        return "MeCO_Opt"
    
    def observe(self):
        state = np.zeros(10)
        state[0] = np.mean(np.var(self.pop, axis=0))
        state[1] = np.mean(np.var((self.costs - self.gbest) / (self.init_obj_max - self.init_obj_min + 1e-10)))
        state[2] = np.mean(self.pop.copy())
        state[3] = np.mean((self.costs - self.gbest) / (self.init_obj_max - self.init_obj_min + 1e-10))
        
        state[4] = self.gbest / (self.init_obj_min + 1e-10) # 加上 1e-10 防止除0
        state[5] = self.__get_top5_violations() / (self.init_top5_violation + 1e-10)
        t_top5_violations = self.__get_top5_violations()
        state[5] = t_top5_violations / (self.init_top5_violation + 1e-10)
        
        state[6] = self.__cal_feasible_ratio()
        
        state[7] = self.fes / self.maxfes
        state[8] = self.eps
        
        NP = self.pop_size
        sum_p = self.__get_sum_penalties(self.penalties)
        
        # 使用高级向量化操作替换原本耗时的双层 for 循环，大幅提升 RL 采样速度
        if NP >= 2:
            i, j = np.triu_indices(NP, k=1)
            dc = self.costs[i] - self.costs[j]
            dp = sum_p[i] - sum_p[j]
            
            # 防止除以0
            valid_mask = dp != 0
            ratio = dc[valid_mask] / dp[valid_mask]
            
            total = np.sum(ratio > 0)
            state[9] = total / (NP * (NP - 1) / 2)
        else:
            state[9] = 0.0
            
        state = np.nan_to_num(
            state,
            nan=0.0,
            posinf=self._state_clip,
            neginf=-self._state_clip
        )
        return np.clip(state, -self._state_clip, self._state_clip)
     
    def __cal_feasible_ratio(self):
        penalties = self.penalties.copy()
        # 计算每一行惩罚值都小于0.001的个体数量
        feasible_count = np.sum(np.all(penalties < 0.001, axis=1))
        return feasible_count / penalties.shape[0]
    
    def __get_top5_violations(self):
        penalties = self.penalties.copy()
        sum_p = self.__get_sum_penalties(penalties)
        # 最好的5个个体的总惩罚值平均
        top5_indices = np.argsort(sum_p)[:5]
        return np.mean(sum_p[top5_indices])
    
    # 新增辅助函数：将 (NP, m) 的惩罚矩阵按行求和，转化为 (NP,) 的总惩罚向量
    def __get_sum_penalties(self, penalties: np.ndarray) -> np.ndarray:
        return np.sum(penalties, axis=1) if penalties.ndim > 1 else penalties
    
    def __sort(self):
        # 核心：获取总惩罚值后，使用 Proxy Penalty 进行 epsilon-级排序
        sum_p = self.__get_sum_penalties(self.penalties)
        proxy_penalties = np.where(sum_p <= self.eps, 0.0, sum_p)
        
        # 优先按 proxy_penalties 排序，相同时按 costs 排序
        ind = np.lexsort((self.costs, proxy_penalties))
        
        self.pop = self.pop[ind]
        self.costs = self.costs[ind]
        self.penalties = self.penalties[ind]
        
    def init_population(self, problem):
        dim = problem.dim
        self.pop_size = self.init_pop_size
        pop = self.rng.rand(self.pop_size, dim)
        costs, penalties = problem.eval(pop)
        costs = np.nan_to_num(
            costs,
            nan=self._cost_clip,
            posinf=self._cost_clip,
            neginf=-self._cost_clip
        )
        penalties = np.nan_to_num(
            penalties,
            nan=self._penalty_clip,
            posinf=self._penalty_clip,
            neginf=self._penalty_clip
        )
        self.pop = pop
        self.costs = costs
        self.penalties = penalties # 保存原始矩阵 (NP, m)
        
        sum_p = self.__get_sum_penalties(self.penalties)
        self.eps_base = np.mean(sum_p)
        self.eps = self.eps_base
        # 初始化排序 (基于当前的 eps)
        self.__sort()
        
        
        self.init_obj_max = np.max(self.costs)
        self.init_obj_min = np.min(self.costs)
        
        self.init_top5_violation = self.__get_top5_violations()

        
        self.gbest = self.costs[0]
        self.gbest_list = [self.gbest]
        # 记录当前最优个体的总惩罚值
        self.penalties_list = [self.__get_sum_penalties(self.penalties)[0]]
        self.eps_list = [self.eps_base]
        
        self.fes = self.costs.shape[0]
        self.fes_list = [self.fes]
        self.gen = 1
        
        self.archive = np.empty((0, dim))
        self.archive_size = self.pop_size
        
        self.mean_F = np.array([0.5] * self.h)
        self.mean_CR = np.array([0.5] * self.h)
        
        self.INIT_VALUES = self.costs.copy() + self.__get_sum_penalties(self.penalties).copy()
        self.INIT_VALUES = np.min(self.INIT_VALUES)
        
        self.costs_gen_list = [self.costs.copy()]
        self.penalties_gen_list = [self.penalties.copy()]
        
        
        # 初始化历史记录索引
        self.k = 0
        return self.observe()

    def __choose_F_CR(self):
        pop_size = self.pop_size
        r = self.rng.randint(0, self.h, size=pop_size)
        
        CR = self.rng.normal(loc=self.mean_CR[r], scale=0.1, size=pop_size) # type: ignore
        CR = np.clip(CR, 0, 1)
        
        F = stats.cauchy.rvs(loc=self.mean_F[r], scale=0.1, size=pop_size, random_state=self.rng) # type: ignore
        invalid_mask = F <= 0
        while np.any(invalid_mask):
            F[invalid_mask] = stats.cauchy.rvs(loc=self.mean_F[r[invalid_mask]], scale=0.1, size=np.sum(invalid_mask), random_state=self.rng)
            invalid_mask = F <= 0
            
        return CR, np.minimum(F, 1)
  
    def __ctb_w_arc(self, group, best, archive, Fs, pnum):
        NP = group.shape[0]
        NA = archive.shape[0]
        idx = np.arange(NP)
        
        rb = self.rng.randint(0, pnum, size=NP)
        
        Rand = self.rng.rand(NP, NP)
        Rand[idx, idx] = np.inf
        r1 = np.argmin(Rand, 1)
        
        Rand = self.rng.rand(NP, NP + NA) 
        Rand[idx, idx] = np.inf
        Rand[idx, r1] = np.inf
        r2 = np.argmin(Rand, 1)
        
        xb = best[rb]
        x1 = group[r1]
        x2 = np.concatenate((group, archive), 0)[r2] if NA > 0 else group[r2]
        v = group + Fs * (xb - group) + Fs * (x1 - x2)
        return v
    
    def __update_archive(self, old_items):
        if old_items.shape[0] == 0:
            return
        self.archive = np.concatenate((self.archive, old_items), axis=0)
        if self.archive.shape[0] > self.archive_size:
            indices = self.rng.permutation(self.archive.shape[0])[:self.archive_size]
            self.archive = self.archive[indices]
  
    def __update_M_F_CR(self, SF:np.ndarray, SCR:np.ndarray, df:np.ndarray, k:int):        
        if SF.shape[0] > 0:
            w = df / (np.sum(df) + 1e-10)
            self.mean_CR[k] = np.sum(w * SCR)
            self.mean_F[k] = np.sum(w * np.power(SF, 2)) / (np.sum(w * SF) + 1e-10)
            k = (k + 1) % self.h
        return k       
    
    def update(self, action, problem):
        # 1. 根据 Agent 给出的 action 计算当前步的 epsilon
        # action 是 at [0, 0.1, 0.2, 0.3, ..., 1] 11 个离散动作
        self.eps = np.power(self.eps_base, action) * np.power(self.delta, 1 - action)
        self.eps_list.append(self.eps)
        
        # 非常关键：根据新的 eps 重新对当前种群排序，确保选出的 pbest 是基于新标准的
        self.__sort()
        
        old_top5_violation = self.__get_top5_violations()
        old_gbest = self.gbest
        
        
        dim = problem.dim
        ub = np.array([self.ub] * dim)
        lb = np.array([self.lb] * dim)
        pop_size = self.pop_size
        
        # 2. L-SHADE 参数自适应
        CR, F = self.__choose_F_CR()
        Fs = F.repeat(dim).reshape(pop_size, dim)
        CRs = CR.repeat(dim).reshape(pop_size, dim)
        
        pnum = max(2, int(round(self.p * pop_size)))
        pbest = self.pop[:pnum] 
        
        # 3. DE/current-to-pbest/1 变异
        v = self.__ctb_w_arc(self.pop, pbest, self.archive, Fs, pnum)
        
        # 边界处理
        v = np.where(v < lb, self.rng.uniform(lb, ub), v)
        v = np.where(v > ub, self.rng.uniform(lb, ub), v)
        
        # 4. 二项式交叉 (Binomial Crossover)
        jrand = self.rng.randint(dim, size=pop_size)
        u = np.where(
            (self.rng.rand(pop_size, dim) < CRs),
            v,
            self.pop
        )
        u[np.arange(pop_size), jrand] = v[np.arange(pop_size), jrand]
        
        # 5. 评估适应度
        new_costs, new_penalties = problem.eval(u)
        new_costs = np.nan_to_num(
            new_costs,
            nan=self._cost_clip,
            posinf=self._cost_clip,
            neginf=-self._cost_clip
        )
        new_penalties = np.nan_to_num(
            new_penalties,
            nan=self._penalty_clip,
            posinf=self._penalty_clip,
            neginf=self._penalty_clip
        )
        
        self.fes += pop_size
        self.fes_list.append(self.fes)
        self.gen += 1
        
        # 6. Epsilon 选择机制
        sum_old = self.__get_sum_penalties(self.penalties)
        sum_new = self.__get_sum_penalties(new_penalties)
        
        proxy_old = np.where(sum_old <= self.eps, 0.0, sum_old)
        proxy_new = np.where(sum_new <= self.eps, 0.0, sum_new)
        
        mask_1 = proxy_new < proxy_old
        mask_2 = (proxy_new == proxy_old) & (new_costs <= self.costs)
        mask = mask_1 | mask_2
        
        # 7. 更新外部存档 Archive
        self.__update_archive(self.pop[mask])
        
        SF = F[mask]
        SCR = CR[mask]
        
        # 8. 计算适应度改善值 (df) 并更新历史记忆
        df = np.where(
            proxy_old[mask] == proxy_new[mask], 
            np.maximum(0, self.costs[mask] - new_costs[mask]), 
            np.maximum(0, proxy_old[mask] - proxy_new[mask])   
        )
        
        # 调用时传入并覆写类属性 self.k
        self.k = self.__update_M_F_CR(SF, SCR, df, self.k)
        
        # 9. 种群替换
        self.pop[mask] = u[mask]
        self.costs[mask] = new_costs[mask]
        self.penalties[mask] = new_penalties[mask]
        
        # 再次排序并记录最佳
        self.__sort()
        
        self.gbest = self.costs[0]
        self.gbest_list.append(self.gbest)
        self.penalties_list.append(self.__get_sum_penalties(self.penalties)[0])
        self.costs_gen_list.append(self.costs.copy())
        self.penalties_gen_list.append(self.penalties.copy())
        # 10. L-SHADE 线性种群缩减 (LPSR)
        new_pop_size = round(
            (self.min_pop_size - self.init_pop_size) / self.maxfes * self.fes + self.init_pop_size
        )
        new_pop_size = max(self.min_pop_size, new_pop_size) 
        
        if new_pop_size < self.pop_size:
            self.pop = self.pop[:new_pop_size]
            self.costs = self.costs[:new_pop_size]
            self.penalties = self.penalties[:new_pop_size]
            
            self.archive_size = new_pop_size
            if self.archive.shape[0] > self.archive_size:
                indices = self.rng.permutation(self.archive.shape[0])[:self.archive_size]
                self.archive = self.archive[indices]
                
            self.pop_size = new_pop_size
        
        
        new_top5_violation = self.__get_top5_violations()
        new_gbest = self.gbest
        
        # 计算目标函数分母（加上绝对值保护）
        obj_denominator = abs(self.gbest_list[0] - 0) + 1e-10

        # 计算违反度分母
        vio_denominator = self.init_top5_violation + 1e-10

        # 优化后的 Reward 计算
        reward = 0.5 * (1 - (new_top5_violation / vio_denominator)) * (old_gbest - new_gbest) / obj_denominator \
            + 0.5 * (old_top5_violation - new_top5_violation) / vio_denominator
        reward = np.nan_to_num(
            reward,
            nan=0.0,
            posinf=self._reward_clip,
            neginf=-self._reward_clip
        )
        reward = float(np.clip(reward, -self._reward_clip, self._reward_clip))
        is_done = self.fes >= self.maxfes
        return self.observe(), reward, is_done
            
    def get_results(self):
        return {
            'pop': self.pop,
            'costs': self.costs,
            'penalties': self.penalties, # 输出最终保留下来的个体的全部约束违规情况矩阵
            'gbest_list': self.gbest_list,
            'penalties_list': self.penalties_list, # 历代最优个体的总惩罚违规量记录
            'fes_list': self.fes_list,
            'eps_list': self.eps_list,
            'costs_gen_list': self.costs_gen_list,
            'penalties_gen_list': self.penalties_gen_list,
            'gen': self.gen
        }