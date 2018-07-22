
# coding: utf-8

# In[1]:

5+5


# In[3]:

import pandas as pd


# In[5]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.datasets import load_boston


# In[6]:

boston=load_boston()


# In[7]:

bos = pd.DataFrame(boston.data)


# In[8]:

print(bos.head(10))


# In[9]:

bos.columns =boston.feature_names


# In[10]:

boston.feature_names


# In[11]:

bos['price']=boston.target


# In[12]:

Y=bos['price']


# In[14]:

X=bos.drop('price',axis=1)


# In[16]:

print(X.head())


# In[17]:

print(Y.head())


# In[18]:

from sklearn.linear_model import LinearRegression
X_train,X_test,Y_train,Y_test=sklearn.model_selection.train_test_split(X,Y,test_size=0.33,random_state=5)


# In[19]:

print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


# In[20]:

ln=LinearRegression()


# In[21]:

a=ln.fit(X_train,Y_train)


# In[23]:

Y_train_pred=ln.predict(X_train)


# In[24]:

Y_test_pred=ln.predict(X_test)


# In[26]:

df=pd.DataFrame(Y_test_pred,Y_test)
print(df)


# In[27]:

from sklearn.metrics import mean_squared_error
mse=mean_squared_error(Y_test,Y_test_pred)
print(mse)


# In[ ]:



