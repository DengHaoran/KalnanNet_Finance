import torch
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

if torch.cuda.is_available():
   dev = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
   torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
   dev = torch.device("cpu")
   print("Running on the CPU")

r2 = torch.tensor([10,1.,0.1,1e-2,1e-3])
vdB = 0 # ratio v=q2/r2
v = 10**(vdB/10)
q2 = r2*v

dataFolderName = 'OU' + '/'
if not os.path.exists(dataFolderName):
   os.makedirs(dataFolderName)
dataFileName = ['OU_data'+str(i)+'.pt' for i in range(5)]
ResultFileName = ['OU_result'+str(i)+'.pt' for i in range(5)]

for i in range(5):
   [a, b] = torch.load(dataFolderName+ResultFileName[i],map_location=dev)