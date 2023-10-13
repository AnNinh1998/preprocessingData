import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sms
from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=np.nan,strategy="mean")
imputer.fit(X[:,1:3])

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0])],remainder="passthrough")
ct.fit_transform(x)

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
Y=le.fit_transform(y) 

from sklearn.model_selection import train_test_split
np.random.seed(42)
X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.2,)
# Import dataset
data_df=pd.read_csv("./Data.csv")
data_df.head()

from sklearn.preprocessing import StandardScaler
sc= StandardScaler()

import matplotlib.pyplot as plt
import seaborn as sms
