#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Dependenices 


# In[7]:


import numpy as np
import pandas as pd #loading data into tables and tables is known as data frame
#sklearn is pythin library for machine learning algorithms 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score 


# In[8]:


#data collection and data processing


# In[9]:


#loading data set to pandas dataframe
sonar_data = pd.read_csv(r"D:\CS\MLD\sonar data.csv",header = None)


# In[10]:


sonar_data.head()


# In[11]:


sonar_data.columns


# In[12]:


### number of rows of coloumns


# In[13]:


sonar_data.shape


# In[14]:


sonar_data.describe() # stastistical data


# In[15]:


sonar_data[60].value_counts()# for measuring various types of data


# In[16]:


sonar_data.groupby(60).mean()


# In[17]:


# supervised learning 
# separating data and labels
x = sonar_data.drop(columns=60,axis=1)#axis
y = sonar_data[60] # why...???


# In[18]:


print(x)


# In[19]:


# training and test data


# In[20]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.1, stratify=y, random_state=1)


# In[21]:


print(x.shape,x_train.shape,x_test.shape)


# In[22]:


# model training....logistic regression


# In[23]:


print(x_train)
print(y_train)


# In[24]:


model = LogisticRegression()
# training LR with training data


# In[25]:


model.fit(x_train, y_train)


# In[26]:


# accuracy on training data
x_train_prediction = model.predict(x_train)
training_data_accuracy = accuracy_score(x_train_prediction, y_train)


# In[27]:


print('Accuracy is :',training_data_accuracy)


# In[28]:


# accuracy on test data
x_test_prediction = model.predict(x_test)
test_data_accuracy = accuracy_score(x_test_prediction, y_test)


# In[29]:


print('Accuracy is :',test_data_accuracy)


# In[30]:


# Predictive data


# In[31]:


input_data=(0.021,0.0121,0.0203,0.1036,0.1675,0.0418,0.0723,0.0828,0.0494,0.0686,0.1125,0.1741,0.271,0.3087,0.3575,0.4998,0.6011,0.647,0.8067,0.9008,0.8906,0.9338,1,0.9102,0.8496,0.7867,0.7688,0.7718,0.6268,0.4301,0.2077,0.1198,0.166,0.2618,0.3862,0.3958,0.3248,0.2302,0.325,0.4022,0.4344,0.4008,0.337,0.2518,0.2101,0.1181,0.115,0.055,0.0293,0.0183,0.0104,0.0117,0.0101,0.0061,0.0031,0.0099,0.008,0.0107,0.0161,0.0133
)
# using numpy array for input data
input_data_as_numpy_array = np.asarray(input_data)

# reshaping the np array for predicting one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)
if prediction=='R':
    print("The object is 'Rock'")
else:
    print("The object is 'Mine'")


# In[32]:


### trials ###


# In[33]:


# sonar_data.groupby([5,6]).count()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




