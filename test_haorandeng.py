import numpy as np
import torch

if torch.cuda.is_available():
   cuda0 = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
   torch.set_default_tensor_type('torch.cuda.FloatTensor')
   print("Running on the GPU")
else:
   cuda0 = torch.device("cpu")
   print("Running on the CPU")

test = torch.arange(24).reshape(2,3,4)

print(test)

sp = list(test.split(1,2))

print(torch.cat(sp,dim=0))
