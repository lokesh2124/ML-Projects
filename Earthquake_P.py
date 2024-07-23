#!/usr/bin/env python
# coding: utf-8

# # Program

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns


# In[2]:


pip install imbalanced-learn


# # Loading the Dataset

# In[3]:


dataset=pd.read_csv("earthquake_data[1].csv")
dataset


# # 1. Data Processing

# Analyzing the Dataset

# In[4]:


dataset.info()


# In[5]:


del dataset["title"]


# In[6]:


del dataset["location"]


# In[7]:


del dataset["country"]


# In[8]:


del dataset["continent"]


# In[9]:


dataset


# Checking for Null Vallues

# In[10]:


dataset.isnull().sum()


# Handling Missing Values

# In[11]:


dataset["alert"] = dataset["alert"].fillna("red")
dataset.isnull().sum()


# In[12]:


dataset


# # Changing datatype of datetime column

# In[13]:


dataset["date_time"]=pd.to_datetime(dataset["date_time"])
dataset.info()


# In[14]:


dataset["date_time"]=pd.DatetimeIndex(dataset["date_time"]).month


# In[15]:


dataset.describe(include=['object'])


# # Label Encoding

# In[16]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
alert_le = LabelEncoder()
magtype_le = LabelEncoder()
net_le = LabelEncoder()
dataset["alert"]=alert_le.fit_transform(dataset["alert"])
dataset["magType"]=magtype_le.fit_transform(dataset["magType"])
dataset["net"]=net_le.fit_transform(dataset["net"])
dataset


# In[17]:


dataset.corr()


# In[18]:


dataset.hist()


# Slicing the Dataset

# In[19]:


x=dataset.iloc[:,[1,2,3,4,6,7,8,9,10,11,12,13,14]]
y=dataset.iloc[:,[5]]


# Balance data using Imbalancing technique

# In[20]:


dataset["tsunami"].value_counts()


# In[21]:


from imblearn.over_sampling import SMOTE
s=SMOTE()
x_data,y_data=s.fit_resample(x,y)


# In[22]:


from collections import Counter
print(Counter(y_data))


# Feature Scaling

# In[23]:


from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
x_scaled=ss.fit_transform(x_data)


# In[24]:


x_scaled


# # 2. Developing the model

# In[25]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x_scaled,y_data,random_state=11,test_size=0.2)


# In[26]:


from sklearn.linear_model import LogisticRegression 
l1=LogisticRegression()
l1.fit(x_train,y_train)


# Logistic Regression

# In[27]:


y_pred=l1.predict(x_test)
y_pred


# In[28]:


from sklearn.metrics import accuracy_score
ac=accuracy_score(y_test,y_pred)*100
ac
print("Accuracy: ",ac)


# SVM MODEL

# In[29]:


from sklearn.svm import SVC 
SVM=SVC(kernel="linear",random_state=2)
SVM.fit(x_train,y_train)


# In[30]:


y_pred1=SVM.predict(x_test)
y_pred1


# In[31]:


ac1=accuracy_score(y_test,y_pred1)*100
ac1
print("Accuracy: ",ac1)


# Gaussian Nave Bayes

# In[32]:


from sklearn.naive_bayes import GaussianNB
nb=GaussianNB()
nb.fit(x_train,y_train)


# In[33]:


y_pred2=nb.predict(x_test)
y_pred2


# In[34]:


ac2=accuracy_score(y_test,y_pred2)*100
ac2
print("Accuracy: ",ac2)


# Decision Tree Classifier

# In[35]:


from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(x_train, y_train) 


# In[36]:


y_pred3=dt.predict(x_test)
y_pred3


# In[37]:


ac3 = accuracy_score(y_test, y_pred3)*100
ac3
print("Accuracy: ",ac3)


# Ensemble Technique

# In[38]:


from sklearn.ensemble import VotingClassifier
bc=VotingClassifier(estimators=[("logisticRegression",l1),("svm",SVM),("naivebayes",nb),("Decision Tree Classification", dt)])
bc.fit(x_train,y_train)


# In[39]:


y_pred4=bc.predict(x_test)
y_pred4


# In[40]:


from sklearn.metrics import accuracy_score
ac4=accuracy_score(y_test,y_pred4)*100
ac4
print("Accuracy: ",ac4)


# Cross validation using KFold technique

# In[41]:


from sklearn.model_selection import KFold
kf=KFold()
kf.split(x_train,y_train)
kf


# In[42]:


from sklearn.model_selection import cross_val_predict
cross_pred=cross_val_predict(bc,x_test,y_test,cv=kf)
cross_pred


# In[43]:


from sklearn.model_selection import cross_val_score
cross_score=cross_val_score(bc,x_train,y_train,cv=kf)
cross_score


# In[44]:


ac5=cross_score.mean()*100
ac5
print("Accuracy: ",ac5)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




