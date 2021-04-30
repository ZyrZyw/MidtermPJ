#!/usr/bin/env python
# coding: utf-8

# In[4]:


import os
import matplotlib.pyplot as plt
import numpy as np
 
MainDir = '/Users/zhangyiwen/Documents/学校/复旦大学/2021春/计算机视觉/pj/darknet'
TrainLogPath = os.path.join(MainDir, 'backup', 'train_loss.txt')
Loss, AgvLoss = [], []
with open(TrainLogPath, 'r') as FId:
    TxtLines = FId.readlines()
    for TxtLine in TxtLines:
        SplitStr = TxtLine.strip().split(',')
        Loss.append(float(SplitStr[0]))
        AgvLoss.append(float(SplitStr[1]))
 
IterNum = len(AgvLoss)
StartVal, EndVal, Stride = 100, IterNum, 10 #视情况修改
Xs = np.arange(StartVal, EndVal, Stride)
Ys = np.array(AgvLoss[StartVal:EndVal:Stride])
plt.plot(Xs, Ys,label='avg_loss')
plt.xlabel('iteration')
plt.ylabel('average loss')
plt.title("Loss-Iter curve(batch=64,sub=16,max_baches=5000)")
plt.legend()
plt.show()


# In[ ]:




