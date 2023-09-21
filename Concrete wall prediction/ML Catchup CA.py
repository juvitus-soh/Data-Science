#!/usr/bin/env python
# coding: utf-8

# In[64]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error as mae


# In[65]:


data = pd.read_csv('/home/leong/Desktop/concrete_clean.csv')
data.head()


# In[66]:


data.isnull().sum()


# In[67]:


X = data.drop(columns=['StrengthMPa'])
Y = data['StrengthMPa']


# In[68]:


#random_state controls the shuffling of the set selected at random for the split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)


# In[69]:


lin_reg = LinearRegression()
lin_reg.fit(X_train, Y_train)


# In[70]:


#predicting
predictions1 = lin_reg.predict([[445.8, 34.8, 0.1, 123.0, 1.2, 456.9, 213.4, 98, 0.606769]])
predictions1


# In[71]:


predictions2 = lin_reg.predict([[115.8, 24.8, 0.0, 123.0, 1.9, 206.9, 111.4, 15, 0.903369]])
predictions2


# In[72]:


#Mean Square Error
'''measures the average of error squares i.e.
the average squared difference between the estimated values and true value'''

training_data = lin_reg.predict(X_train)
MSE = mean_squared_error(Y_train, training_data)
MSE


# In[76]:


#mean absolute error
'''Mean Absolute Error calculates the average
difference between the calculated values and actual values'''
MAE = mae(Y_train, training_data)
MAE


# In[ ]:




