import torch
import os
torch.pi = torch.acos(torch.zeros(1)).item() * 2 # which is 3.1415927410125732
from Linear_sysmdl import SystemModel
from Extended_data import DataGen,DataLoader,DataLoader_GPU, Decimate_and_perturbate_Data,Short_Traj_Split
import time


# import sys
# sys.path.insert(1, 'OU/')
from param_ou import kappa, N_E, N_CV, N_T, F, H, T_true, T, T_test, m1_0, m2_0, m, n #,F_rotated, H_rotated,

start = time.time()

if torch.cuda.is_available():
   dev = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
   torch.set_default_tensor_type('torch.cuda.FloatTensor')
   print("Running on the GPU")
else:
   dev = torch.device("cpu")
   print("Running on the CPU")



print("OU_True Trajectory Generation")


####################
### Design Model ###
####################
r2 = torch.tensor([10,1.,0.1,1e-2,1e-3])


dataFolderName = 'OU' + '/'
if not os.path.exists(dataFolderName):
   os.makedirs(dataFolderName)
dataFileName = 'true_continuous.pt'

# True model
r = torch.sqrt(r2[0])
q = 0
sys_model = SystemModel(F, q, H, r, T, T_test)
sys_model.InitSequence(m1_0, m2_0)

print("Start Data Gen")
sys_model.GenerateBatch(1, T_true, randomInit=False, True_data = 1)
# true_input = sys_model.Input
true_target = sys_model.Target


torch.save([true_target], dataFolderName + dataFileName)

print('true state size:', true_target.size())

end = time.time()

print('time spent = ', end-start)