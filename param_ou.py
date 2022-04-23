import torch
import math

if torch.cuda.is_available():
    dev = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
   dev = torch.device("cpu")
   print("Running on the CPU")

#######################
### Size of DataSet ###
#######################

# Number of Training Examples
N_E = 300

# Number of Cross Validation Examples
N_CV = 5

N_T = 10


############
## 1 x 1 ###
############


m = 1
n = 1
kappa = 1.0
tau = 1e-5
F = torch.exp(torch.tensor([[-1*kappa*tau]]))
# sigma = 1.0
H = torch.eye(1)
m1_0 = torch.tensor([3.0]).to(dev)
# m1x_0_design = torch.tensor([[10.0], [-10.0]])
m2_0 = 0 * 0 * torch.eye(m).to(dev)

delta_t_gen =  tau
delta_t_train = 0.02
delta_t_test = 0.01

# Decimation ratio
ratio = delta_t_gen/delta_t_test

# Sequence Length for Linear Case
T_true = 100000
T = int(T_true*delta_t_gen/delta_t_train)
T_test = int(T_true*delta_t_gen/delta_t_test)

# training hyperparameters
epoches = 100
batchsize = 10

def h(x):
    return torch.matmul(H,x)

def f(x):
    return torch.matmul(F,x)