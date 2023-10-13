#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np


# In[3]:


import matplotlib.pyplot as plt
import seaborn as sms


# In[ ]:


##import data_set


# In[16]:


data_df = pd.read_csv("./Data.csv")


# In[9]:


data_df.head()


# In[ ]:


#Xử lí dữ liệu bị Nan


# In[10]:


data_df.info()


# In[29]:


for col in data_df.columns:
    Missing_data=data_df[col].isna().sum()
    Missing_percent=Missing_data/len(data_df)*100
    print(f"Cột {col} có {Missing_percent} % dữ liệu bị Nan")


# In[11]:


fix,ag=plt.subplots(figsize=(8,5))
sms.heatmap(data_df.isna(),);


# In[19]:


data_df = pd.read_csv("./Data.csv")


# In[28]:


x= data_df.iloc[:,:-1].values
x


# In[ ]:


#x= data_df.iloc[:,:-1] lấy các giá trị trong dataset, ngoại trừ giá trị cuối
#x chứa các giá trị hoành độ


# In[27]:


y= data_df.iloc[:,-1].values
y


# In[ ]:


#y chứa các giá trị mua hàng hay ko, tung độ


# In[41]:


from sklearn.impute import SimpleImputer
#Tạo class chứa các giá trị Nan
imputer= SimpleImputer(missing_values=np.nan,strategy="mean")
imputer.fit(x[:,-1:3])
x[:,-1:3]=imputer.transform(x[:, -1:3])
x


# In[ ]:


#Mã hóa dữ liệu danh mục
#Chuyển hóa dữ liệu dạng string sang dạng numberic
#encoding indenpendent varible(x) mã hóa biến độc lập


# In[42]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0])],remainder="passthrough")
X= ct.fit_transform(x)
X


# In[ ]:


#encoding denpendent varible(y) biến phụ thuộc mã hóa


# In[49]:


y= data_df.iloc[:,-1].values
y


# In[51]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
y=le.fit_transform(y) 
y


# In[ ]:


#Spliting the dataset(x=data input,y=output) into the training set and test set


# In[52]:


from sklearn.model_selection import train_test_split
np.random.seed(42)
X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.2,)


# In[53]:


X_train


# In[54]:


y_train


# In[55]:


X_test


# In[56]:


y_test


# In[ ]:


#Feature Scalling


# In[58]:


from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
X_train[:,3:]=sc.fit_transform(X_train[:,3:])
X_train


# In[59]:


X_test=sc.transform(X_train[:,3:])
X_test


# In[ ]:




