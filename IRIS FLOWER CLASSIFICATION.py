#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[5]:


data=pd.read_csv("iris.csv")
data


# In[6]:


data.head()


# In[7]:


data.tail()


# In[8]:


data.describe()


# In[18]:


print(data.shape)


# In[20]:


#Checking for null values
print(data.isna().sum())
print(data.describe())


# In[9]:


# Visualizing that data
sns.pairplot(data,hue="Species")


# In[22]:


#Checking for outliars
import matplotlib.pyplot as plt
plt.figure(1)
plt.boxplot([data['SepalLengthCm']])
plt.figure(2)
plt.boxplot([data['SepalWidthCm']])
plt.show()


# In[10]:


# Now let's sepparate the data
df=data.values #we are importing the flower data values on one variable df
x=df[:,0:5]
y=df[:,5]

df # so the actual values of the data in this we having 6 columns and 150 rows


# In[24]:


data.hist()
plt.show()


# In[27]:


X = data['SepalLengthCm'].values.reshape(-1,1)
print(X)


# In[28]:


Y = data['SepalWidthCm'].values.reshape(-1,1)
print(Y)


# In[31]:


plt.xlabel("SepalLengthCm")
plt.ylabel("SepalWidthCm")
plt.scatter(X,Y,color='b')
plt.show()


# In[33]:


#Correlation 
corr_mat = data.corr()
print(corr_mat)


# In[34]:


from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier


# In[36]:


train, test = train_test_split(data, test_size = 0.25)
print(train.shape)
print(test.shape)


# In[45]:


train_X = train[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm',
                 'PetalWidthCm']]
train_Y = train.Species

test_X = test[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm',
                 'PetalWidthCm']]
test_Y = test.Species


# In[46]:


train_X.head()


# In[47]:


test_Y.head()


# In[48]:


model = LogisticRegression()
model.fit(train_X, train_Y)
prediction = model.predict(test_X)
print('Accuracy:',metrics.accuracy_score(prediction,test_Y))


# In[49]:


#Confusion matrix
from sklearn.metrics import confusion_matrix
confusion_mat = confusion_matrix(test_Y,prediction)
print("Confusion matrix: \n",confusion_mat)


# In[ ]:





# In[ ]:




