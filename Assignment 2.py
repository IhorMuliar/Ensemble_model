#!/usr/bin/env python
# coding: utf-8

# In[175]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.datasets import make_blobs


# In[38]:


#Loading raw data file without headings + naming each column + defining "?" as NaN value 


# In[179]:


df = pd.read_csv('crx.data', delimiter=",", header=None, na_values=["?"], names=['0', 'First', 'Second', '3','4', '5', '6', 'Seventh','8', '9', 'Tenth', '11','12', 'Thirteen', 'Fourteen', 'Approval'])


# In[180]:


df


# In[113]:


# Columns with missing walues (12+12+13= 37 total) 


# In[114]:


df.isnull().sum()


# In[122]:


#Removing missing values (drop all the rows with missing values). As a result we got total 653 rows of clean data.


# In[150]:


df.dropna(axis='rows')


# In[124]:


#Calculating credits approval rate (383 rejected / 307 approved)


# In[125]:


df.Approval.value_counts(dropna=False)


# In[126]:


#Visualization of percentage distribution of approved/rejected credits (55.51% approved/44.49% rejected credits)


# In[127]:


df.Approval.value_counts().plot(kind='pie', figsize =(4,4), title='Credits approval rate', autopct='%0.2f%%');


# In[154]:


#descriptive statistics of each continuous variable


# In[155]:


df.describe().T


# In[157]:


# Checking type of distribution for each continuous variables (all of the variables are not normally distributed)


# In[159]:


df['Second'].plot(kind="hist")


# In[160]:


df['Seventh'].plot(kind="hist")


# In[181]:


df['Tenth'].plot(kind="hist")


# In[183]:


df['Thirteen'].plot(kind="hist")


# In[163]:


df['Fourteen'].plot(kind="hist")


# In[171]:


#Corellation between continuous variables


# In[184]:


df.corr()


# In[199]:


#Linear learner


# In[ ]:




