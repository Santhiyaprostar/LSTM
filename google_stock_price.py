#!/usr/bin/env python
# coding: utf-8

#    # Google Stock Price Prediction 

# Import Libraries

# In[92]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras


# Import Dataset

# In[93]:


google_training_complete=pd.read_csv(r"C:\Users\SRKT\Desktop\Google_Stock_Price_Train.csv")
google_training_processed=google_training_complete.iloc[:,1:2].values
print(len(google_training_processed))


# Data Normalization

# In[94]:


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
google_training_scaled=scaler.fit_transform(google_training_processed)
print(keras.__version__)


# Convert Training Data to Right Shape

# In[95]:


feature_set=[]
labels=[]
for i in range(120,2299):
    feature_set.append(google_training_scaled[i-120:i,0])
    labels.append(google_training_scaled[i,0])
feature_set,labels=np.array(feature_set),np.array(labels)
feature_set=np.reshape(feature_set,(feature_set.shape[0],feature_set.shape[1],1))


# Training The LSTM

# In[96]:


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


# In[97]:


model=Sequential()


# Creating LSTM , Dropout & Dense Layers

# In[98]:


model.add(LSTM(units=100,return_sequences=True,input_shape=(feature_set.shape[1],1)))
model.add(Dropout(0.2))
model.add(LSTM(units=100,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=100,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=100))
model.add(Dropout(0.2))


# In[99]:


model.add(Dense(units=1))


# Model Compilation

# In[100]:


model.compile(optimizer='adam',loss='mean_squared_error')


# Training

# In[101]:


model.fit(feature_set,labels,epochs=100,batch_size=32)


# Testing our LSTM

# In[103]:


google_testing_complete=pd.read_csv(r"C:\Users\SRKT\Desktop\Google_Stock_Price_Test.csv")
google_testing_processed=google_testing_complete.iloc[:,1:2].values


# Convert Test Data to Right Shape

# In[104]:


google_total=pd.concat((google_training_complete['open'],google_testing_complete['open']),axis=0)
print(len(google_total))
test_inputs=google_total[len(google_total)-len(google_testing_complete)-120:].values
print(len(test_inputs))


# In[105]:


test_inputs=test_inputs.reshape(-1,1)
test_inputs=scaler.transform(test_inputs)


# In[106]:


test_features=[]
for i in range(120,340):
    test_features.append(test_inputs[i-120:i,0])


# In[107]:


test_features=np.array(test_features)
test_features=np.reshape(test_features,(test_features.shape[0],test_features.shape[1],1))


# Prediction

# In[108]:


predictions=model.predict(test_features)
predictions=scaler.inverse_transform(predictions)


# Graphs 

# In[109]:


plt.figure(figsize=(10,6))
plt.plot(google_testing_processed,color='blue',label='actual stock price')
plt.plot(predictions,color='red',label="predicted stock price")
plt.title("google stock price prediction")
plt.xlabel('data')
plt.ylabel('google stock price')
plt.legend()
plt.show()


# In[ ]:




