from unicodedata import category
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split 

# import data sets
data= pd.read_csv("Census_Income/data.csv")
print(data.head())
print("SHAPE:", data.shape)

#no identifiable column names, so adding column names
col_names = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation', 'relationship',
             'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income']
data.columns = col_names
print(data.columns)
#identify categorical variables
print(data.info())

#Segregate categorical variables 
categorical= [v for v in data.columns if data[v].dtype=='O']
print('The categorical variables :', categorical)
#only cloumns of categorical data
print(data[categorical].head())


# all unique values in categorical data
# print("\nall unique values in categorical data: \n",np.unique(data[categorical].values))
# unique values in each categorical data and frequecy distribution of values and 
# replacing '?' with 'NaN'
print("\nunique values in each categorical data and frequecy distribution of values: \n")
for col in categorical:
    print("\nUnique row names in Column: ",col,data[col].unique())
    print("Frequency of values: \n",data[col].value_counts())
    data[col].replace('?',np.NaN,inplace=True)
    data[col].replace(' ?',np.NaN,inplace=True)


# print("unique in occupation \n",data.occupation.value_counts())

print("\nNumber of missing values:",data[categorical].isnull().sum())
#workclass,occuption,native_country has missing values

for col in categorical:
    print(col,'contains ',len(data[col].unique()),' labels')

#Segregate numerical variables 
numerical= [v for v in data.columns if data[v].dtype!='O']
print('\nThe numberical variables :', numerical)
#only cloumns of numerical data
print('\n',data[numerical].head())

print("\nNumber of missing values:",data[numerical].isnull().sum())
#no missing values

#declaring features and labels
X=data.drop(['income'],axis=1)
y=data['income']

#splitting test and train data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)