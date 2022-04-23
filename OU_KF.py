import torch
import os
torch.pi = torch.acos(torch.zeros(1)).item() * 2 # which is 3.1415927410125732
import torch.nn as nn
from Linear_sysmdl import SystemModel
from KalmanNet_sysmdl import System_Model
from Extended_data import DataGen,DataLoader,DataLoader_GPU, Decimate_and_perturbate_Data,Short_Traj_Split
from Pipeline_EKF import Pipeline_EKF
from KalmanNet_nn import KalmanNetNN
from datetime import datetime

from KalmanFilter_test import KFTest
import matplotlib.pyplot as plt

# import sys
# sys.path.insert(1, 'OU/')
from param_ou import kappa, N_E, N_CV, N_T, F, f, H, h, T_true, T, T_test, m1_0, m2_0, m, n, delta_t_gen, delta_t_test, delta_t_train, batchsize, epoches #,F_rotated, H_rotated,


# from Plot import Plot_RTS as Plot

if torch.cuda.is_available():
   dev = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
   torch.set_default_tensor_type('torch.cuda.FloatTensor')
   print("Running on the GPU")
else:
   dev = torch.device("cpu")
   print("Running on the CPU")



print("OU_Experiment Start")

################
### Get Time ###
################
today = datetime.today()
now = datetime.now()
strToday = today.strftime("%m.%d.%y")
strNow = now.strftime("%H:%M:%S")
strTime = strToday + "_" + strNow
print("Current Time =", strTime)
path_results = 'OU/'

####################
### Design Model ###
####################
# q2 = sigma**2*(1-F**2)/(2*kappa)
r2 = torch.tensor([10,1.,0.1,1e-2,1e-3])
vdB = 0 # ratio v=q2/r2
v = 10**(vdB/10)
q2 = r2*v

dataFolderName = 'OU' + '/'
if not os.path.exists(dataFolderName):
   os.makedirs(dataFolderName)
dataFileName = ['OU_data'+str(i)+'.pt' for i in range(len(r2))]
ResultFileName = ['OU_result'+str(i)+'.pt' for i in range(len(r2))]

[true_sequence] = torch.load(dataFolderName+'true_continuous.pt', map_location=dev)


for index in range(0,len(r2)):

   print("1/r2 [dB]: ", 10 * torch.log10(1/r2[index]))
   print("1/q2 [dB]: ", 10 * torch.log10(1/q2[index]))

   # True model
   r = torch.sqrt(r2[index])
   q = torch.sqrt(q2[index])
   sys_model = SystemModel(F, q, H, r, T, T_test)
   sys_model.InitSequence(m1_0, m2_0)

   sys_model_KNet = System_Model(f, q.unsqueeze(0), h, r.unsqueeze(0), T, T_test)
   sys_model_KNet.InitSequence(m1_0, m2_0)

   # # Mismatched model
   # sys_model_partialh = SystemModel(F, q, H_rotated, r, T, T_test)
   # sys_model_partialh.InitSequence(m1_0, m2_0)

   ###################################
   ### Data Loader (Generate Data) ###
   ###################################

   # print("Start Data Gen")
   # dataFileName = ['rq'+str(index)+'.pt' for in   'OU.pt','1x1_rq020_T100.pt','1x1_rq1030_T100.pt','1x1_rq2040_T100.pt','1x1_rq3050_T100.pt']
   [test_target, test_input] = Decimate_and_perturbate_Data(true_sequence, delta_t_gen, delta_t_test, N_T, h, r)
   [train_target, train_input] = Decimate_and_perturbate_Data(true_sequence, delta_t_gen, delta_t_train, N_E, h, r)
   [cv_target, cv_input] = Decimate_and_perturbate_Data(true_sequence, delta_t_gen, delta_t_train, N_CV, h, r)

   # print("testset size:",test_target.size())
   # print("trainset size:",train_target.size())
   # print("cvset size:",cv_target.size())


   # DataGen(sys_model, dataFolderName + dataFileName[index], T, T_test, randomInit=False)
   # print("Data Load")
   # [train_input, train_target, cv_input, cv_target, test_input, test_target] = DataLoader_GPU(dataFolderName + dataFileName[index])
   # if n==1:
   #    train_input = train_input.unsqueeze(1)
   #    cv_input = cv_input.unsqueeze(1)
   #    test_input = test_input.unsqueeze(1)
   # if m==1:
   #    train_target = train_target.unsqueeze(1)
   #    cv_target = cv_target.unsqueeze(1)
   #    test_target = test_target.unsqueeze(1)
   # print("trainset size:",train_target.size())
   # print("cvset size:",cv_target.size())
   # print("testset size:",test_target.size())

   ##############################
   ### Evaluate Kalman Filter ###
   ##############################
   # print("Evaluate Kalman Filter True")

   # KF = KalmanFilter(sys_model)
   [KF, MSE_KF_linear_arr, MSE_KF_linear_avg, MSE_KF_dB_avg] = KFTest(sys_model, test_input, test_target)

   # plot
   # fig, ax = plt.subplots()
   # ax.plot(KF.x[0], label = 'KF predicts')
   # ax.plot(test_input, label = 'observation')
   # ax.plot(test_target, label = 'state')
   # ax.set_xlabel('discrete time t')
   #
   # ax.set_ylabel('interest rate')
   # ax.set_title('when q^2/r^2 = '+str(vdB)+' dB')
   # ax.legend()
   # plt.show()

   # datafolderName = 'Filters/Linear' + '/'
   DataResultName = ResultFileName[index]
   torch.save({
               'MSE_KF_linear_arr': MSE_KF_linear_arr,
               'MSE_KF_dB_avg': MSE_KF_dB_avg,
               # 'MSE_KF_linear_arr_partialh': MSE_KF_linear_arr_partialh,
               # 'MSE_KF_dB_avg_partialh': MSE_KF_dB_avg_partialh,
               }, dataFolderName+DataResultName)

   ################################
   ### Evaluate KalmanNet Arch2 ###
   ################################

   KNet_model = KalmanNetNN()
   KNet_model.Build(sys_model_KNet)
   print("Number of trainable parameters for KNet:",sum(p.numel() for p in KNet_model.parameters() if p.requires_grad))
   ## Train Neural Network
   KNet_Pipeline = Pipeline_EKF(strTime, "OU", "KalmanNet")
   KNet_Pipeline.setssModel(sys_model)
   KNet_Pipeline.setModel(KNet_model)
   KNet_Pipeline.setTrainingParams(n_Epochs=epoches, n_Batch=batchsize, learningRate=1e-3, weightDecay=1e-6)
#    print('Training')
   KNet_Pipeline.NNTrain(train_input, train_target,cv_input, cv_target)
   ## Test Neural Network
#    print('Testing')
   [KNet_MSE_test_linear_arr, KNet_MSE_test_linear_avg, KNet_MSE_test_dB_avg, KNet_test] = KNet_Pipeline.NNTest(test_input, test_target)
   KNet_Pipeline.save()

print("finish OU")
