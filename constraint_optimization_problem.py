
from abc import ABC, abstractmethod
import random
import numpy as np
import torch
import math

from scipy.constants import troy_ounce


class Batchprocessor:
    def __init__(self,Dim ,dim=0):
        self.batchsize = 200
        #雷：batchsize数需整除个体数
        self.dim = dim
    def process(self, matrix, function,epsilon):
        # if torch.cuda.is_available():
        #     device = torch.device('cuda')
        #     matrix = matrix.to(device)
        #     epsilon = epsilon.to(device)
        total_size = matrix.size(self.dim)
        num_size = (total_size + self.batchsize - 1) // self.batchsize
        ob_results = []
        con_results = []
        for i in range(num_size):
            start_index = i * self.batchsize
            end_index = min(start_index + self.batchsize, total_size)
            if self.dim == 0:
                batchmatrix = matrix[start_index:end_index, :]
            else:
                batchmatrix = matrix[:, start_index:end_index]
            ob_result, con_result = function(batchmatrix,epsilon)
            ob_result=torch.nan_to_num(ob_result, nan=1e+9)
            ob_results.append(ob_result)
            con_results.append(con_result)
        # epsilon=epsilon.to('cpu')

        if self.dim == 0:
            # ob,cons= torch.cat(ob_results, dim=0).to('cpu'), torch.cat(con_results, dim=1).to('cpu')
             ob,cons= torch.cat(ob_results, dim=0), torch.cat(con_results, dim=1)
             return torch.nan_to_num(ob,1e+9),cons
        else:
            ob, cons = torch.cat(ob_results, dim=0), torch.cat(con_results, dim=1)
            return torch.nan_to_num(ob, 1e+9), cons
def crossover_binomial(x_i, v_i, CR_i):
    u_i = torch.zeros(x_i.shape[0])
    for i in range(x_i.shape[0]):
        if random.random() < CR_i:
            u_i[i] = v_i[i].item()
        else:
            u_i[i] = x_i[i].item()
    return u_i
class Constraint_Optimization_Problem(ABC):
    def __init__(self,dim,changeable,min_tensor,max_tensor):
        self.dim=dim
        self.changeable=changeable
        self.min_tensor=min_tensor
        self.max_tensor=max_tensor
        rng1 = np.random.default_rng(seed=42)
        rng2 = np.random.default_rng(seed=47)
        random_matrix = rng1.normal(loc=0.0, scale=1.0, size=(dim, dim))
        random_matrix1 = rng2.normal(loc=0.0, scale=1.0, size=(dim, dim))
        q, _ = np.linalg.qr(random_matrix)
        q1, _ = np.linalg.qr(random_matrix1)
        det_q = np.linalg.det(q)
        det_q1 = np.linalg.det(q1)
        if det_q < 0:
            q[:, 0] = -q[:, 0]
        if det_q1 < 0:
            q1[:, 0] = -q1[:, 0]
        self.rotated=torch.from_numpy(q).t()
        self.rotated1 = torch.from_numpy(q1).t()
        b = rng1.uniform(min_tensor, max_tensor, size=(self.dim, ))
        b1 = rng2.uniform(min_tensor, max_tensor, size=(self.dim,))
        self.bias=torch.from_numpy(b)
        self.bias1 = torch.from_numpy(b1)
        self.optimum = 0
        # if torch.cuda.is_available():
        #     self.device=torch.device('cuda')
        # else:
        #     self.device = torch.device('cpu')
        # self.bias=self.bias.to(self.device)
        # self.bias1 = self.bias1.to(self.device)
        # self.rotated1 = self.rotated1.to(self.device)
        # self.rotated=self.rotated.to(self.device)

    def objective_function(self,x):
        pass
    def calculate_result(self,x,epsilon=0.001):
        pass
    def cal_feasible_rate(self,x):
        total_num=x.shape[1]//2
        feasible_num=0
        for i in range(total_num):
            num = torch.count_nonzero(x[:, i])
            if num==0:
               feasible_num+=1
        return feasible_num/total_num
    def element_clamp(self,tensor, min_tensor, max_tensor):
        lower_tensor = torch.where(tensor < min_tensor, min_tensor, tensor)
        clamp_tensor = torch.where(tensor > max_tensor, max_tensor, lower_tensor)
        return clamp_tensor
    def epsilon_clamp(self,x,epsilon):
        mask = (x<epsilon.unsqueeze(0))
        x[mask] = 0
        return x
class C01(Constraint_Optimization_Problem):
    def __init__(self,dim,changeable,min_tensor=-100,max_tensor=100):
        super().__init__(dim,changeable,min_tensor,max_tensor)
        self.cons_num=1
    def C01obfunction(self,x):
        x=x-self.bias
        length = x.size(dim=1)
        for i in range(1, length):
            x[:, i] = x[:, i] + x[:, i - 1]
        return torch.sum(x ** 2, dim=1)

    def C01inconfunction1(self,x, epsilon=0.001):
        x=x-self.bias
        x1 = x.clone()
        x = x ** 2
        x1 = 5000 * torch.cos(0.1 * math.pi * x1)
        x = x - x1 - 4000
        x = torch.sum(x, dim=1)
        x[x < epsilon] = 0
        return x
    def calculate_result(self,x, epsilon=0.001):
        obres = self.C01obfunction(x)
        conres1 = self.C01inconfunction1(x)
        cons=conres1.unsqueeze(0)
        return obres, torch.nan_to_num(cons, nan=1e+9)
class C02(Constraint_Optimization_Problem):
    def __init__(self,dim,changeable,min_tensor=-100,max_tensor=100):
        super().__init__(dim,changeable,min_tensor,max_tensor)
        self.cons_num=1
    def C02obfunction(self,x):
        x=x-self.bias
        length = x.size(dim=1)
        for i in range(1, length):
            x[:, i] = x[:, i] + x[:, i - 1]
        return torch.sum(x ** 2, dim=1)

    def C02inconfunction1(self,x, epsilon=0.001):
        x=(x-self.bias)@self.rotated
        x1 = x.clone()
        x = x ** 2
        x1 = 5000 * torch.cos(0.1 * math.pi * x1)
        x = x - x1 - 4000
        x = torch.sum(x, dim=1)
        x[x < epsilon] = 0
        return x
    def calculate_result(self,x, epsilon=0.001):
        obres = self.C02obfunction(x)
        conres1 = self.C02inconfunction1(x)
        cons=conres1.unsqueeze(0)
        return obres, torch.nan_to_num(cons, nan=1e+9)
class C03(Constraint_Optimization_Problem):
    def __init__(self,dim,changeable,min_tensor=-100,max_tensor=100):
        super().__init__(dim,changeable,min_tensor,max_tensor)
        self.cons_num=2

    def C03obfunction(self,x):
        length = x.size(dim=1)
        for i in range(1, length):
            x[:, i] = x[:, i] + x[:, i - 1]
        return torch.sum(x ** 2, dim=1)

    def C03inconfunction1(self,x, epsilon=0.001):
        x1 = x.clone()
        x = x ** 2
        x1 = 5000 * torch.cos(0.1 * math.pi * x1)
        x = x - x1 - 4000
        x = torch.sum(x, dim=1)
        x[x < epsilon] = 0
        return x

    def C03econfunction1(self,x, epsilon=0.001):
        x = -1 * x * torch.sin(0.1 * math.pi * x)
        x = torch.sum(x, dim=1)
        x[torch.abs(x) < epsilon] = 0
        return torch.abs(x)

    def calculate_result(self,x, epsilon=0.001):
        obres = self.C03obfunction(x)
        conres1 = self.C03inconfunction1(x)
        conres2 = self.C03econfunction1(x)
        cons=torch.stack((conres1, conres2), dim=0)
        return obres, torch.nan_to_num(cons, nan=1e+9)
class C04(Constraint_Optimization_Problem):
    def __init__(self,dim,changeable,min_tensor=-10,max_tensor=10):
        super().__init__(dim,changeable,min_tensor,max_tensor)
        self.cons_num=2
    def C04obfunction(self,x):
        x=x-self.bias
        x1 = x.clone()
        x = x ** 2
        x1 = 10 * torch.cos(2 * math.pi * x1)
        x = x - x1 + 10
        x = torch.sum(x, dim=1)
        return x
    def C04inconfunction1(self,x, epsilon=0.001):
        x=x-self.bias
        x =torch.sum( -1 * x * torch.sin(2 * x),dim=1)
        x[x < epsilon] = 0
        return x
    def C04inconfunction2(self,x, epsilon=0.001):
        x=x-self.bias
        x =torch.sum( x * torch.sin(x),dim=1)
        x[x < epsilon] = 0
        return x
    def calculate_result(self,x, epsilon=0.001):
        obres = self.C04obfunction(x)
        conres1 = self.C04inconfunction1(x)
        conres2 = self.C04inconfunction2(x)
        cons=torch.stack((conres1, conres2), dim=0)
        return obres, torch.nan_to_num(cons, nan=1e+9)
class C05(Constraint_Optimization_Problem):
    def __init__(self,dim,changeable,min_tensor=-10,max_tensor=10):
        super().__init__(dim,changeable,min_tensor,max_tensor)
        self.cons_num=2
    def C05obfunction(self,x):
        x=x-self.bias
        x1=x.clone()
        x[:,x.shape[1]-1]=0
        x1=x1[:,1:]
        x1=torch.cat((x1,torch.zeros(size=(x1.shape[0],)).unsqueeze(1)),dim=1)
        x=100*(x**2-x1)**2+(x-1)**2
        return torch.sum(x,dim=1)
    def C05inconfunction1(self,x, epsilon=0.001):
        x=(x-self.bias)@self.rotated
        x=torch.sum(x**2-50*torch.cos(2*torch.pi*x)-40,dim=1)
        x[x < epsilon] = 0
        return x
    def C05inconfunction2(self,x, epsilon=0.001):
        x = (x - self.bias) @ self.rotated1
        x = torch.sum(x ** 2 - 50 * torch.cos(2 * torch.pi * x) - 40, dim=1)
        x[x < epsilon] = 0
        return x
    def calculate_result(self,x, epsilon=0.001):
        obres = self.C05obfunction(x)
        conres1 = self.C05inconfunction1(x)
        conres2 = self.C05inconfunction2(x)
        cons=torch.stack((conres1, conres2), dim=0)
        return obres, torch.nan_to_num(cons, nan=1e+9)
class C06(Constraint_Optimization_Problem):
    def __init__(self,dim,changeable,min_tensor=-20,max_tensor=20):
        super().__init__(dim,changeable,min_tensor,max_tensor)
        self.cons_num=6
    def C06obfunction(self,x):
        x=x-self.bias
        return torch.sum(x**2-10*torch.cos(2*torch.pi*x)+10,dim=1)

    def C06econfunction1(self, x, epsilon=0.001):
        x=x-self.bias
        x=torch.sum(-1*x*torch.sin(x),dim=1)
        x[torch.abs(x) < epsilon] = 0
        return torch.abs(x)
    def C06econfunction2(self, x, epsilon=0.001):
        x=x-self.bias
        x=torch.sum(x*torch.sin(torch.pi*x),dim=1)
        x[torch.abs(x) < epsilon] = 0
        return torch.abs(x)
    def C06econfunction3(self, x, epsilon=0.001):
        x=x-self.bias
        x=torch.sum(-1*x*torch.cos(x),dim=1)
        x[torch.abs(x) < epsilon] = 0
        return torch.abs(x)
    def C06econfunction4(self, x, epsilon=0.001):
        x=x-self.bias
        x=torch.sum(x*torch.cos(torch.pi*x),dim=1)
        x[torch.abs(x) < epsilon] = 0
        return torch.abs(x)
    def C06econfunction5(self, x, epsilon=0.001):
        x=x-self.bias
        x=torch.sum(x*torch.sin(2*torch.sqrt(torch.abs(x))),dim=1)
        x[torch.abs(x) < epsilon] = 0
        return torch.abs(x)
    def C06econfunction6(self, x, epsilon=0.001):
        x=x-self.bias
        x = torch.sum(-1*x * torch.sin(2 * torch.sqrt(torch.abs(x))), dim=1)
        x[torch.abs(x) < epsilon] = 0
        return torch.abs(x)
    def calculate_result(self,x,y=0):
        obres = self.C06obfunction(x)
        conres1 = self.C06econfunction1(x)
        conres2 = self.C06econfunction2(x)
        conres3 = self.C06econfunction3(x)
        conres4 = self.C06econfunction4(x)
        conres5 = self.C06econfunction5(x)
        conres6 = self.C06econfunction6(x)
        cons=torch.stack((conres1, conres2, conres3, conres4, conres5, conres6), dim=0)
        return obres, torch.nan_to_num(cons, nan=1e+9)
class C07(Constraint_Optimization_Problem):
    def __init__(self,dim,changeable,min_tensor=-50,max_tensor=50):
        super().__init__(dim,changeable,min_tensor,max_tensor)
        self.cons_num=2
    def C07obfunction(self,x):
        x=x-self.bias
        return torch.sum(x*torch.sin(x),dim=1)
    def C07econfunction1(self, x, epsilon=0.001):
        x=x-self.bias
        x=torch.sum(x-100*torch.cos(0.5*x)+100,dim=1)
        x[torch.abs(x) < epsilon] = 0
        return torch.abs(x)
    def C07econfunction2(self, x, epsilon=0.001):
        x=x-self.bias
        x=-1*torch.sum(x-100*torch.cos(0.5*x)+100,dim=1)
        x[torch.abs(x) < epsilon] = 0
        return torch.abs(x)
    def calculate_result(self,x, epsilon=0.001):
        obres = self.C07obfunction(x)
        conres1 = self.C07econfunction1(x)
        conres2 = self.C07econfunction2(x)
        cons=torch.stack((conres1, conres2), dim=0)
        return obres, torch.nan_to_num(cons, nan=1e+9)
class C08(Constraint_Optimization_Problem):
    def __init__(self,dim,changeable,min_tensor=-100,max_tensor=100):
        super().__init__(dim,changeable,min_tensor,max_tensor)
        self.cons_num=2
    def C08obfunction(self,x):
        x=x-self.bias
        x,_=torch.max(x,dim=1)
        return x
    def C08econfunction1(self, x, epsilon=0.001):
        x=x-self.bias
        indexs=torch.arange(0,x.shape[1],2)
        # if torch.cuda.is_available():
        #     device = torch.device('cuda')
        #     indexs=indexs.to(device)
        x=x[:,indexs]
        for i in range(1,x.shape[1]):
            x[:,i]+=x[:,i-1]
        x=torch.sum(x**2,dim=1)
        x[torch.abs(x) < epsilon] = 0
        return torch.abs(x)
    def C08econfunction2(self, x, epsilon=0.001):
        x=x-self.bias
        indexs=torch.arange(1,x.shape[1],2)
        # if torch.cuda.is_available():
        #     device = torch.device('cuda')
        #     indexs=indexs.to(device)
        x=x[:,indexs]
        for i in range(1,x.shape[1]):
            x[:,i]+=x[:,i-1]
        x=torch.sum(x**2,dim=1)
        x[torch.abs(x) < epsilon] = 0
        return torch.abs(x)
    def calculate_result(self,x, epsilon=0.001):
        obres = self.C08obfunction(x)
        conres1 = self.C08econfunction1(x)
        conres2 = self.C08econfunction2(x)
        cons=torch.stack((conres1, conres2), dim=0)
        return obres, torch.nan_to_num(cons, nan=1e+9)
class C09(Constraint_Optimization_Problem):
    def __init__(self,dim,changeable,min_tensor=-10,max_tensor=10):
        super().__init__(dim,changeable,min_tensor,max_tensor)
        self.cons_num=2
    def C09obfunction(self,x):
        x=x-self.bias
        x,_=torch.max(x,dim=1)
        return x
    def C09inconfunction1(self,x, epsilon=0.001):
        x = x - self.bias
        indexs = torch.arange(1, x.shape[1], 2)
        # if torch.cuda.is_available():
        #     device = torch.device('cuda')
        #     indexs=indexs.to(device)
        x = x[:, indexs]
        x=torch.cumprod(x,dim=1)[:,x.shape[1]-1]
        x[x < epsilon] = 0
        return x
    def C09econfunction1(self,x, epsilon=0.001):
        x = x - self.bias
        indexs = torch.arange(0, x.shape[1], 2)
        # if torch.cuda.is_available():
        #     device = torch.device('cuda')
        #     indexs=indexs.to(device)
        x = x[:, indexs]
        for i in range(1, x.shape[1]):
            x[:, i] += x[:, i - 1]
        x1=x[:,1:]
        x1=torch.cat((x1,torch.zeros(size=(x1.shape[0],)).unsqueeze(1)),dim=1)
        x=torch.sum((x**2-x1)**2,dim=1)
        x[torch.abs(x) < epsilon] = 0
        return torch.abs(x)

    def calculate_result(self, x, epsilon=0.001):
        obres = self.C09obfunction(x)
        conres1 = self.C09inconfunction1(x)
        conres2 = self.C09econfunction1(x)
        cons = torch.stack((conres1, conres2), dim=0)
        return obres, torch.nan_to_num(cons, nan=1e+9)
class C10(Constraint_Optimization_Problem):
    def __init__(self,dim,changeable,min_tensor=-100,max_tensor=100):
        super().__init__(dim,changeable,min_tensor,max_tensor)
        self.cons_num=2

    def C10obfunction(self, x):
        x = x - self.bias
        x, _ = torch.max(x, dim=1)
        return x
    def C10econfunction1(self,x, epsilon=0.001):
        x = x - self.bias
        for i in range(1, x.shape[1]):
            x[:, i] += x[:, i - 1]
        x=torch.sum(x**2,dim=1)
        x[torch.abs(x) < epsilon] = 0
        return torch.abs(x)
    def C10econfunction2(self,x, epsilon=0.001):
        x = x - self.bias
        x1=x[:,1:]
        x1=torch.cat((x1,torch.zeros(size=(x1.shape[0],)).unsqueeze(1)),dim=1)
        x[:,x.shape[1]-1]=0
        x=torch.sum((x-x1)**2,dim=1)
        x[torch.abs(x) < epsilon] = 0
        return torch.abs(x)
    def calculate_result(self, x, epsilon=0.001):
        obres = self.C10obfunction(x)
        conres1 = self.C10econfunction1(x)
        conres2 = self.C10econfunction2(x)
        cons = torch.stack((conres1, conres2), dim=0)
        return obres, torch.nan_to_num(cons, nan=1e+9)
class C11(Constraint_Optimization_Problem):
    def __init__(self,dim,changeable,min_tensor=-100,max_tensor=100):
        super().__init__(dim,changeable,min_tensor,max_tensor)
        self.cons_num=2

    def C11obfunction(self, x):
        x=x-self.bias
        return torch.sum(x,dim=1)
    def C11inconfunction1(self,x, epsilon=0.001):
        x = x - self.bias
        x=torch.cumprod(x,dim=1)[:,x.shape[1]-1]
        x[x < epsilon] = 0
        return x
    def C11econfunction1(self,x, epsilon=0.001):
        x = x - self.bias
        x1=x[:,1:]
        x1=torch.cat((x1,torch.zeros(size=(x1.shape[0],)).unsqueeze(1)),dim=1)
        x[:,x.shape[1]-1]=0
        x=torch.sum((x-x1)**2,dim=1)
        x[torch.abs(x) < epsilon] = 0
        return torch.abs(x)
    def calculate_result(self, x, epsilon=0.001):
        obres = self.C11obfunction(x)
        conres1 = self.C11inconfunction1(x)
        conres2 = self.C11econfunction1(x)
        cons = torch.stack((conres1, conres2), dim=0)
        return obres, torch.nan_to_num(cons, nan=1e+9)
class C12(Constraint_Optimization_Problem):
    def __init__(self,dim,changeable,min_tensor=-100,max_tensor=100):
        super().__init__(dim,changeable,min_tensor,max_tensor)
        self.cons_num=2
    def C12obfunction(self, x):
        x=x-self.bias
        return torch.sum(x**2-10*torch.cos(2*torch.pi*x)+10,dim=1)

    def C12inconfunction1(self, x, epsilon=0.001):
        x=x-self.bias
        x=4-torch.sum(torch.abs(x),dim=1)
        x[x < epsilon] = 0
        return x
    def C12econfunction1(self,x, epsilon=0.001):
        x = x - self.bias
        x=torch.sum(x**2,dim=1)-4
        x[torch.abs(x) < epsilon] = 0
        return torch.abs(x)
    def calculate_result(self, x, epsilon=0.001):
        obres = self.C12obfunction(x)
        conres1 = self.C12inconfunction1(x)
        conres2 = self.C12econfunction1(x)
        cons = torch.stack((conres1, conres2), dim=0)
        return obres, torch.nan_to_num(cons, nan=1e+9)
class C13(Constraint_Optimization_Problem):
    def __init__(self,dim,changeable,min_tensor=-100,max_tensor=100):
        super().__init__(dim,changeable,min_tensor,max_tensor)
        self.cons_num=3
    def C13obfunction(self, x):
        x = x - self.bias
        x1 = x.clone()
        x[:, x.shape[1] - 1] = 0
        x1 = x1[:, 1:]
        x1 = torch.cat((x1, torch.zeros(size=(x1.shape[0],)).unsqueeze(1)), dim=1)
        x = 100 * (x ** 2 - x1) ** 2 + (x - 1) ** 2
        return torch.sum(x, dim=1)
    def C13inconfunction1(self, x, epsilon=0.001):
        x=x-self.bias
        x=torch.sum(x**2-10*torch.cos(2*torch.pi*x)+10,dim=1)-100
        x[x < epsilon] = 0
        return x
    def C13inconfunction2(self, x, epsilon=0.001):
        x=x-self.bias
        x=torch.sum(x,dim=1)-2*self.dim
        x[x < epsilon] = 0
        return x
    def C13inconfunction3(self, x, epsilon=0.001):
        x=x-self.bias
        x=5-torch.sum(x,dim=1)
        x[x < epsilon] = 0
        return x
    def calculate_result(self, x, epsilon=0.001):
        obres = self.C13obfunction(x)
        conres1 = self.C13inconfunction1(x)
        conres2 = self.C13inconfunction2(x)
        conres3 = self.C13inconfunction3(x)
        cons = torch.stack((conres1, conres2,conres3), dim=0)
        return obres, torch.nan_to_num(cons, nan=1e+9)
class C14(Constraint_Optimization_Problem):
    def __init__(self,dim,changeable,min_tensor=-100,max_tensor=100):
        super().__init__(dim,changeable,min_tensor,max_tensor)
        self.cons_num=2
        self.optimum = 4
    def C14obfunction(self, x):
        x=x-self.bias
        return -20*torch.exp(-0.2*torch.sqrt(torch.sum(x**2,dim=1)/self.dim))+20-torch.exp(torch.sum(torch.cos(2*torch.pi*x),dim=1)/self.dim)+math.e
    def C14inconfunction1(self, x, epsilon=0.001):
        x=x-self.bias
        x=torch.sum(x**2,dim=1)+1-torch.abs(x[:,0])
        x[x < epsilon] = 0
        return x
    def C14econfunction1(self,x, epsilon=0.001):
        x = x - self.bias
        x=torch.sum(x**2,dim=1)-4
        x[torch.abs(x) < epsilon] = 0
        return torch.abs(x)
    def calculate_result(self, x, epsilon=0.001):
        obres = self.C14obfunction(x)
        conres1 = self.C14inconfunction1(x)
        conres2 = self.C14econfunction1(x)
        cons = torch.stack((conres1, conres2), dim=0)
        return obres, torch.nan_to_num(cons, nan=1e+9)
class C15(Constraint_Optimization_Problem):
    def __init__(self,dim,changeable,min_tensor=-100,max_tensor=100):
        super().__init__(dim,changeable,min_tensor,max_tensor)
        self.cons_num=2
    def C15obfunction(self, x):
        x=x-self.bias
        x,_=torch.max(torch.abs(x),dim=1)
        return x
    def C15inconfunction1(self, x, epsilon=0.001):
        x=x-self.bias
        x=torch.sum(x**2,dim=1)-100*self.dim
        x[x < epsilon] = 0
        return x
    def C15econfunction1(self,x, epsilon=0.001):
        x=torch.cos(self.C15obfunction(x))+torch.sin(self.C15obfunction(x))
        x[torch.abs(x) < epsilon] = 0
        return torch.abs(x)
    def calculate_result(self, x, epsilon=0.001):
        obres = self.C15obfunction(x)
        conres1 = self.C15inconfunction1(x)
        conres2 = self.C15econfunction1(x)
        cons = torch.stack((conres1, conres2), dim=0)
        return obres, torch.nan_to_num(cons, nan=1e+9)
class C16(Constraint_Optimization_Problem):
    def __init__(self,dim,changeable,min_tensor=-100,max_tensor=100):
        super().__init__(dim,changeable,min_tensor,max_tensor)
        self.cons_num=2
    def C16obfunction(self, x):
        x=x-self.bias
        return torch.sum(torch.abs(x),dim=1)
    def C16inconfunction1(self, x, epsilon=0.001):
        x=x-self.bias
        x=torch.sum(x**2,dim=1)-100*self.dim
        x[x < epsilon] = 0
        return x
    def C16econfunction1(self,x, epsilon=0.001):
        x=(torch.cos(self.C16obfunction(x))+torch.sin(self.C16obfunction(x)))**2-torch.exp(torch.cos(self.C16obfunction(x))+torch.sin(self.C16obfunction(x)))-1+math.e
        x[torch.abs(x) < epsilon] = 0
        return torch.abs(x)
    def calculate_result(self, x, epsilon=0.001):
        obres = self.C16obfunction(x)
        conres1 = self.C16inconfunction1(x)
        conres2 = self.C16econfunction1(x)
        cons = torch.stack((conres1, conres2), dim=0)
        return obres, torch.nan_to_num(cons, nan=1e+9)
class C17(Constraint_Optimization_Problem):
    def __init__(self,dim,changeable,min_tensor=-100,max_tensor=100):
        super().__init__(dim,changeable,min_tensor,max_tensor)
        self.cons_num=2
    def C17obfunction(self,x):
        y1=x-self.bias
        l=x.shape[1]
        a1=torch.sqrt(torch.arange(1,l+1,1))
        # if torch.cuda.is_available():
        #     device=torch.device('cuda')
        #     a1=a1.to(device)
        return torch.sum(y1**2,dim=1)/4000+1-torch.prod(torch.cos(y1/a1),dim=1)
    def C17inconfunction1(self,x,epsilon=0.001):
        z1=x-self.bias
        z3=1-torch.sum(torch.sgn(torch.abs(z1)-(torch.sum(z1**2,dim=1).reshape(-1,1)-z1**2)-1),dim=1)
        z3[z3<epsilon]=0
        return z3
    def C17enconfunction1(self,x,epsilon=0.001):
        x=x-self.bias
        x=torch.abs(torch.sum(x**2,dim=1)-4*x.shape[1])
        x[x<epsilon]=0
        return x
    def calculate_result(self,x, epsilon=0.001):
        obres = self.C17obfunction(x)
        conres1 = self.C17inconfunction1(x)
        conres2 = self.C17enconfunction1(x)
        cons=torch.stack((conres1, conres2), dim=0)
        return obres, torch.nan_to_num(cons, nan=1e+9)
class C18(Constraint_Optimization_Problem):
    def __init__(self,dim,changeable,min_tensor=-100,max_tensor=100):
        super().__init__(dim,changeable,min_tensor,max_tensor)
        self.cons_num=3
    def C18obfunction(self, x):
        z1=x-self.bias
        return torch.sum(z1**2-10*torch.cos(2*z1*torch.pi)+10,dim=1)
    def C18inconfunction1(self,x,epsilon=0.001):
        x=1 - torch.sum(torch.abs(x-self.bias),dim=1)
        x[x<epsilon]=0
        return x
    def C18inconfunction2(self,x,epsilon=0.001):
        x=torch.sum((x-self.bias)**2,dim=1)-100*x.shape[1]
        x[x < epsilon] = 0
        return x
    def C18enconfunction1(self,x,epsilon=0.001):
        y=x-self.bias
        last=y[:,-1:]
        other=y[:,:-1]
        y1=torch.cat((last,other),dim=1)
        y=torch.abs(torch.sum(100*(y**2-y1)**2,dim=1)+torch.prod(torch.sin((y-1)*torch.pi)**2))
        y[y < epsilon] = 0
        return y

    def calculate_result(self, x, epsilon=0.001):
        obres = self.C18obfunction(x)
        conres1 = self.C18inconfunction1(x)
        conres2 = self.C18inconfunction2(x)
        conres3 = self.C18enconfunction1(x)
        cons = torch.stack((conres1, conres2,conres3), dim=0)
        return obres, torch.nan_to_num(cons, nan=1e+9)
class C19(Constraint_Optimization_Problem):
    def __init__(self,dim,changeable,min_tensor=-50,max_tensor=50):
        super().__init__(dim,changeable,min_tensor,max_tensor)
        self.cons_num=2
    def C19obfunction(self, x):
        x=x-self.bias
        return torch.sum(torch.abs(x)**0.5+2*torch.sin(x**3),dim=1)
    def C19inconfunction1(self,x, epsilon=0.001):
        x=x-self.bias
        x1 = x[:, 1:]
        x1 = torch.cat((x1, torch.zeros(size=(x1.shape[0],)).unsqueeze(1)), dim=1)
        x[:, x.shape[1] - 1] = 0
        x=torch.sum(-100*torch.exp(-0.2*torch.sqrt(x**2+x1**2)),dim=1)+(self.dim-1)*10/math.exp(-5)
        x[x < epsilon] = 0
        return x
    def C19inconfunction2(self,x, epsilon=0.001):
        x = x - self.bias
        x = torch.sum(torch.sin(2*x)**2,dim=1)-0.5*self.dim
        x[x < epsilon] = 0
        return x
    def calculate_result(self,x, epsilon=0.001):
        obres = self.C19obfunction(x)
        conres1 = self.C19inconfunction1(x)
        conres2 = self.C19inconfunction2(x)
        cons=torch.stack((conres1, conres2), dim=0)
        return obres, torch.nan_to_num(cons, nan=1e+9)
class C20(Constraint_Optimization_Problem):
    def __init__(self,dim,changeable,min_tensor=-100,max_tensor=100):
        super().__init__(dim,changeable,min_tensor,max_tensor)
        self.cons_num=2
    def C20obfunction(self, x):
        x=x-self.bias
        x1=torch.cat((x[:,1:],x[:,:1]),dim=1)
        return torch.sum(0.5+(torch.sin(torch.sqrt(x**2+x1**2))**2-0.5)/(1+0.001*torch.sqrt(x**2+x1**2))**2,dim=1)
    def C20inconfunction1(self,x, epsilon=0.001):
        x=x-self.bias
        x=torch.cos(torch.sum(x,dim=1))**2-0.25*torch.cos(torch.sum(x,dim=1))-0.125
        x[x < epsilon] = 0
        return x
    def C20inconfunction2(self,x, epsilon=0.001):
        x = x - self.bias
        x = torch.exp(torch.cos(torch.sum(x,dim=1)))-math.exp(0.25)
        x[x < epsilon] = 0
        return x
    def calculate_result(self,x, epsilon=0.001):
        obres = self.C20obfunction(x)
        conres1 = self.C20inconfunction1(x)
        conres2 = self.C20inconfunction2(x)
        cons=torch.stack((conres1, conres2), dim=0)
        return obres, torch.nan_to_num(cons, nan=1e+9)
class C21(Constraint_Optimization_Problem):
    def __init__(self,dim,changeable,min_tensor=-100,max_tensor=100):
        super().__init__(dim,changeable,min_tensor,max_tensor)
        self.cons_num=2
    def C21obfunction(self, x):
        x=x-self.bias
        return torch.sum(x**2-10*torch.cos(2*math.pi*x)+10,dim=1)
    def C21inconfunction1(self,x, epsilon=0.001):
        x=(x-self.bias)@self.rotated
        x=4-torch.sum(torch.abs(x),dim=1)
        x[x < epsilon] = 0
        return x
    def C21inconfunction2(self,x, epsilon=0.001):
        x = (x - self.bias)@self.rotated
        x = torch.sum(x**2,dim=1)-4
        x[x < epsilon] = 0
        return x
    def calculate_result(self,x, epsilon=0.001):
        obres = self.C21obfunction(x)
        conres1 = self.C21inconfunction1(x)
        conres2 = self.C21inconfunction2(x)
        cons=torch.stack((conres1, conres2), dim=0)
        return obres, torch.nan_to_num(cons, nan=1e+9)
class C22(Constraint_Optimization_Problem):
    def __init__(self,dim,changeable,min_tensor=-100,max_tensor=100):
        super().__init__(dim,changeable,min_tensor,max_tensor)
        self.cons_num=3
    def C22obfunction(self, x):
        x = (x - self.bias)@self.rotated
        x1 = x.clone()
        x[:, x.shape[1] - 1] = 0
        x1 = x1[:, 1:]
        x1 = torch.cat((x1, torch.zeros(size=(x1.shape[0],)).unsqueeze(1)), dim=1)
        x = 100 * (x ** 2 - x1) ** 2 + (x - 1) ** 2
        return torch.sum(x, dim=1)
    def C22inconfunction1(self, x, epsilon=0.001):
        x=(x-self.bias)@self.rotated
        x=torch.sum(x**2-10*torch.cos(2*torch.pi*x)+10,dim=1)-100
        x[x < epsilon] = 0
        return x
    def C22inconfunction2(self, x, epsilon=0.001):
        x=(x-self.bias)@self.rotated
        x=torch.sum(x,dim=1)-2*self.dim
        x[x < epsilon] = 0
        return x
    def C22inconfunction3(self, x, epsilon=0.001):
        x=(x-self.bias)@self.rotated
        x=5-torch.sum(x,dim=1)
        x[x < epsilon] = 0
        return x
    def calculate_result(self, x, epsilon=0.001):
        obres = self.C22obfunction(x)
        conres1 = self.C22inconfunction1(x)
        conres2 = self.C22inconfunction2(x)
        conres3 = self.C22inconfunction3(x)
        cons = torch.stack((conres1, conres2,conres3), dim=0)
        return obres, torch.nan_to_num(cons, nan=1e+9)
class C23(Constraint_Optimization_Problem):
    def __init__(self,dim,changeable,min_tensor=-100,max_tensor=100):
        super().__init__(dim,changeable,min_tensor,max_tensor)
        self.cons_num=2
    def C23obfunction(self, x):
        x=(x-self.bias)@self.rotated
        return -20*torch.exp(-0.2*torch.sqrt(torch.sum(x**2,dim=1)/self.dim))+20-torch.exp(torch.sum(torch.cos(2*torch.pi*x),dim=1)/self.dim)+math.e
    def C23inconfunction1(self, x, epsilon=0.001):
        x=(x-self.bias)@self.rotated
        x=torch.sum(x**2,dim=1)+1-torch.abs(x[:,0])
        x[x < epsilon] = 0
        return x
    def C23econfunction1(self,x, epsilon=0.001):
        x = (x - self.bias)@self.rotated
        x=torch.sum(x**2,dim=1)-4
        x[torch.abs(x) < epsilon] = 0
        return torch.abs(x)
    def calculate_result(self, x, epsilon=0.001):
        obres = self.C23obfunction(x)
        conres1 = self.C23inconfunction1(x)
        conres2 = self.C23econfunction1(x)
        cons = torch.stack((conres1, conres2), dim=0)
        return obres, torch.nan_to_num(cons, nan=1e+9)
class C24(Constraint_Optimization_Problem):
    def __init__(self,dim,changeable,min_tensor=-100,max_tensor=100):
        super().__init__(dim,changeable,min_tensor,max_tensor)
        self.cons_num=2
    def C24obfunction(self, x):
        x=(x-self.bias)@self.rotated
        x,_=torch.max(torch.abs(x),dim=1)
        return x
    def C24inconfunction1(self, x, epsilon=0.001):
        x=(x-self.bias)@self.rotated
        x=torch.sum(x**2,dim=1)-100*self.dim
        x[x < epsilon] = 0
        return x
    def C24econfunction1(self,x, epsilon=0.001):
        x=torch.cos(self.C24obfunction(x))+torch.sin(self.C24obfunction(x))
        x[torch.abs(x) < epsilon] = 0
        return torch.abs(x)
    def calculate_result(self, x, epsilon=0.001):
        obres = self.C24obfunction(x)
        conres1 = self.C24inconfunction1(x)
        conres2 = self.C24econfunction1(x)
        cons = torch.stack((conres1, conres2), dim=0)
        return torch.nan_to_num(obres, nan=1e+9), torch.nan_to_num(cons, nan=1e+9)
class C25(Constraint_Optimization_Problem):
    def __init__(self,dim,changeable,min_tensor=-100,max_tensor=100):
        super().__init__(dim,changeable,min_tensor,max_tensor)
        self.cons_num=2
    def C25obfunction(self, x):
        x=(x-self.bias)@self.rotated
        return torch.sum(torch.abs(x),dim=1)
    def C25inconfunction1(self, x, epsilon=0.001):
        x=(x-self.bias)@self.rotated
        x=torch.sum(x**2,dim=1)-100*self.dim
        x[x < epsilon] = 0
        return x
    def C25econfunction1(self,x, epsilon=0.001):
        x=(torch.cos(self.C25obfunction(x))+torch.sin(self.C25obfunction(x)))**2-torch.exp(torch.cos(self.C25obfunction(x))+torch.sin(self.C25obfunction(x)))-1+math.e
        x[torch.abs(x) < epsilon] = 0
        return torch.abs(x)
    def calculate_result(self, x, epsilon=0.001):
        obres = self.C25obfunction(x)
        conres1 = self.C25inconfunction1(x)
        conres2 = self.C25econfunction1(x)
        cons = torch.stack((conres1, conres2), dim=0)
        return obres, torch.nan_to_num(cons, nan=1e+9)
class C26(Constraint_Optimization_Problem):
    def __init__(self,dim,changeable,min_tensor=-100,max_tensor=100):
        super().__init__(dim,changeable,min_tensor,max_tensor)
        self.cons_num=2
    def C26obfunction(self,x):
        y1=x-self.bias
        l=x.shape[1]
        a1=torch.sqrt(torch.arange(1,l+1,1))
        # if torch.cuda.is_available():
        #     device=torch.device('cuda')
        #     a1=a1.to(device)
        return torch.sum(y1**2,dim=1)/4000+1-torch.prod(torch.cos(y1/a1),dim=1)
    def C26inconfunction1(self,x,epsilon=0.001):
        z1=(x-self.bias)@self.rotated
        z3=1-torch.sum(torch.sgn(torch.abs(z1)-(torch.sum(z1**2,dim=1).reshape(-1,1)-z1**2)-1),dim=1)
        z3[z3<epsilon]=0
        return z3
    def C26enconfunction1(self,x,epsilon=0.001):
        x=(x-self.bias)@self.rotated
        x=torch.abs(torch.sum(x**2,dim=1)-4*x.shape[1])
        x[x<epsilon]=0
        return x
    def calculate_result(self,x, epsilon=0.001):
        obres = self.C26obfunction(x)
        conres1 = self.C26inconfunction1(x)
        conres2 = self.C26enconfunction1(x)
        cons=torch.stack((conres1, conres2), dim=0)
        return obres, torch.nan_to_num(cons, nan=1e+9)
class C27(Constraint_Optimization_Problem):
    def __init__(self,dim,changeable,min_tensor=-100,max_tensor=100):
        super().__init__(dim,changeable,min_tensor,max_tensor)
        self.cons_num=3
    def C27obfunction(self, x):
        z1=(x-self.bias)@self.rotated
        return torch.sum(z1**2-10*torch.cos(2*z1*torch.pi)+10,dim=1)
    def C27inconfunction1(self,x,epsilon=0.001):
        x=1 - torch.sum(torch.abs(x-self.bias),dim=1)
        x[x<epsilon]=0
        return x
    def C27inconfunction2(self,x,epsilon=0.001):
        x=torch.sum((x-self.bias)**2,dim=1)-100*x.shape[1]
        x[x < epsilon] = 0
        return x
    def C27enconfunction1(self,x,epsilon=0.001):
        y=x-self.bias
        last=y[:,-1:]
        other=y[:,:-1]
        y1=torch.cat((last,other),dim=1)
        y=torch.abs(torch.sum(100*(y**2-y1)**2,dim=1)+torch.prod(torch.sin((y-1)*torch.pi)**2))
        y[y < epsilon] = 0
        return y

    def calculate_result(self, x, epsilon=0.001):
        obres = self.C27obfunction(x)
        conres1 = self.C27inconfunction1(x)
        conres2 = self.C27inconfunction2(x)
        conres3 = self.C27enconfunction1(x)
        cons = torch.stack((conres1, conres2,conres3), dim=0)
        return obres, torch.nan_to_num(cons, nan=1e+9)
class C28(Constraint_Optimization_Problem):
    def __init__(self,dim,changeable,min_tensor=-50,max_tensor=50):
        super().__init__(dim,changeable,min_tensor,max_tensor)
        self.cons_num=2
    def C28obfunction(self, x):
        x=(x-self.bias)@self.rotated
        return torch.sum(torch.abs(x)**0.5+2*torch.sin(x**3),dim=1)
    def C28inconfunction1(self,x, epsilon=0.001):
        x=(x-self.bias)@self.rotated
        x1 = x[:, 1:]
        x1 = torch.cat((x1, torch.zeros(size=(x1.shape[0],)).unsqueeze(1)), dim=1)
        x[:, x.shape[1] - 1] = 0
        x=torch.sum(-100*torch.exp(-0.2*torch.sqrt(x**2+x1**2)),dim=1)+(self.dim-1)*10/math.exp(-5)
        x[x < epsilon] = 0
        return x
    def C28inconfunction2(self,x, epsilon=0.001):
        x = (x - self.bias)@self.rotated
        x = torch.sum(torch.sin(2*x)**2,dim=1)-0.5*self.dim
        x[x < epsilon] = 0
        return x
    def calculate_result(self,x, epsilon=0.001):
        obres = self.C28obfunction(x)
        conres1 = self.C28inconfunction1(x)
        conres2 = self.C28inconfunction2(x)
        cons=torch.stack((conres1, conres2), dim=0)
        return obres, torch.nan_to_num(cons, nan=1e+9)


