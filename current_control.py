#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import matplotlib.pyplot as plt


# In[5]:


time=np.linspace(0,0.01,10000)
result=[]
time_val=[]
for i in range(0,len(time)):
    t=time[i]
    res=(0.809 - 0.809*np.exp(-1847.0589*t)*(np.cosh(0.808*t) + 2286.635*np.sinh(0.808*t)))*1.2355
    result.append(res)
    time_val.append(t)


# In[6]:


plt.figure()
plt.plot(time_val,result)


# In[ ]:





# In[ ]:




