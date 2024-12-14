## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:
        Read the given Data.
STEP 2:
        Clean the Data Set using Data Cleaning Process.
STEP 3:
        Apply Feature Encoding for the feature in the data set.
STEP 4:
        Apply Feature Transformation for the feature in the data set.
STEP 5: 
        Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
from google.colab import drive
drive.mount('/content/drive')


![Screenshot 2024-12-14 211021](https://github.com/user-attachments/assets/317058b4-9a2b-47ee-8ff4-9c82356178ad)

ls drive/MyDrive/bmi.csv


![Screenshot 2024-12-14 211036](https://github.com/user-attachments/assets/b94f110a-a0d0-4947-be0b-653ea4bdd8bd)

import pandas as pd
from scipy import stats
import numpy as np

df=pd.read_csv("drive/MyDrive/bmi.csv")

df.head()


![Screenshot 2024-12-14 211051](https://github.com/user-attachments/assets/f47e3e26-6bf1-4d23-aaf2-79e12a35f29e)

df.dropna()

max_vals=np.max(np.abs(df[['Height','Weight']]))
max_vals


![Screenshot 2024-12-14 211105](https://github.com/user-attachments/assets/375a59a2-4d04-4850-b0b9-1d99918697e5)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
df[['Height','Weight']]=sc.fit_transform(df[['Height','Weight']])
df.head(10)


![Screenshot 2024-12-14 211119](https://github.com/user-attachments/assets/43cc9ae1-7871-4de7-a56a-1f7d5193b988)

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)


![Screenshot 2024-12-14 211130](https://github.com/user-attachments/assets/5392b0ce-0c67-4333-b7f0-afed88184e0d)

from sklearn.preprocessing import Normalizer
scaler=Normalizer()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df


![Screenshot 2024-12-14 211142](https://github.com/user-attachments/assets/8848021e-5052-4605-9a96-e2e172172dea)

df=pd.read_csv("/content/drive/MyDrive/bmi.csv")

from sklearn.preprocessing import MaxAbsScaler
scaler=MaxAbsScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df


![Screenshot 2024-12-14 211254](https://github.com/user-attachments/assets/79499884-1e8d-43f9-892d-ade575e0de9b)

df=pd.read_csv("/content/drive/MyDrive/bmi.csv")

from sklearn.preprocessing import RobustScaler
scaler=RobustScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head()



![Screenshot 2024-12-14 211306](https://github.com/user-attachments/assets/4a86c0fa-657a-4982-b40f-f0082790e760)

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import chi2

df=pd.read_csv("/content/drive/MyDrive/titanic_dataset.csv")

df.columns


![Screenshot 2024-12-14 211319](https://github.com/user-attachments/assets/fa065f8a-efcc-4edc-aaad-fa0e50c5545c)

df.shape


![Screenshot 2024-12-14 211331](https://github.com/user-attachments/assets/6dad69ed-c6f4-4064-9917-fc6bb5e91d37)

x=df.drop('Survived',axis=1)
y=df['Survived']



df=df.drop(["Name","Sex","Ticket","Cabin","Embarked"],axis=1)
df.columns


![Screenshot 2024-12-14 211348](https://github.com/user-attachments/assets/b581dfa4-b064-4b2a-8098-3483297e3511)

df['Age'].isnull().sum()


![Screenshot 2024-12-14 211400](https://github.com/user-attachments/assets/1f0ab38c-5a21-480e-8f52-879d5216bec9)


df['Age'].fillna(method='ffill')


![Screenshot 2024-12-14 211423](https://github.com/user-attachments/assets/f8946361-2147-4ad0-83c6-96a99b753549)

df['Age']=df['Age'].fillna(method='ffill')
df['Age'].isnull().sum()

data=pd.read_csv("/content/drive/MyDrive/titanic_dataset.csv")

data=data.dropna()

x=data.drop(['Survived','Name','Ticket'],axis=1)
y=data['Survived']
x


![Screenshot 2024-12-14 211458](https://github.com/user-attachments/assets/c0d5f7ce-8653-44e9-9e9f-a04bd350862c)

data["Sex"]=data["Sex"].astype("category")
data["Cabin"]=data["Cabin"].astype("category")
data["Embarked"]=data["Embarked"].astype("category")

data["Sex"]=data["Sex"].cat.codes
data["Cabin"]=data["Cabin"].cat.codes
data["Embarked"]=data["Embarked"].cat.codes

data


![Screenshot 2024-12-14 211517](https://github.com/user-attachments/assets/f13ac3a9-6c49-4f86-8a64-0afb99c5778f)

for column in['Sex','Cabin','Embarked']:
   if x[column].dtype=='object':
             x[column]=x[column].astype('category').cat.codes
k=5
selector=SelectKBest(score_func=chi2,k=k)
X_new=selector.fit_transform(x,y)

selected_feature_indices=selector.get_support(indices=True)
selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)


![Screenshot 2024-12-14 211530](https://github.com/user-attachments/assets/a7f7f917-ef3c-48e4-8e8a-ccf0e8429eff)

x.info()


![Screenshot 2024-12-14 211543](https://github.com/user-attachments/assets/a5551287-a383-4306-8ee8-9f3c637ac563)

x=x.drop(["Sex","Cabin","Embarked"],axis=1)
x


![Screenshot 2024-12-14 211556](https://github.com/user-attachments/assets/3ab4f7cc-e752-4b19-bb3b-e4a2a0579e50)

from sklearn.feature_selection import SelectKBest, f_regression
selector=SelectKBest(score_func=f_regression,k=5)
X_new=selector.fit_transform(x,y)

selected_feature_indices=selector.get_support(indices=True)
selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)


![Screenshot 2024-12-14 211610](https://github.com/user-attachments/assets/53a242d1-77f3-47e4-9add-6b7d3bf5313e)

from sklearn.feature_selection import SelectKBest, mutual_info_classif
selector=SelectKBest(score_func=mutual_info_classif,k=5)
X_new=selector.fit_transform(x,y)

selected_features_indices=selector.get_support(indices=True)
selected_features=x.columns[selected_features_indices]
print("Selected Features:")
print(selected_features)


![Screenshot 2024-12-14 211622](https://github.com/user-attachments/assets/1bd8ac47-a795-4c97-93fe-3ea61fee04fd)

from sklearn.feature_selection import SelectPercentile,chi2
selector=SelectPercentile(score_func=chi2,percentile=10)
x_new=selector.fit_transform(x,y)

from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

model=RandomForestClassifier()
sfm=SelectFromModel(model,threshold='mean')
sfm.fit(x,y)
selected_features=x.columns[sfm.get_support()]
print("Selected Features:")
print(selected_features)


![Screenshot 2024-12-14 211634](https://github.com/user-attachments/assets/cc395c85-9659-40e9-b34c-ab71ae875f81)

model = RandomForestClassifier(n_estimators=100,random_state=42)
model.fit(x,y)
feature_importance = model.feature_importances_
threshold=0.15
selected_features=x.columns[feature_importance > threshold]
print("Selected Features:")
print(selected_features)


![Screenshot 2024-12-14 211646](https://github.com/user-attachments/assets/9aca0f42-cf70-4156-bafa-1028da885fef)

df=pd.read_csv("/content/drive/MyDrive/titanic_dataset.csv")
df.columns


![Screenshot 2024-12-14 211700](https://github.com/user-attachments/assets/8e90b311-f21f-4946-84cd-a41546bdb0da)

df


![Screenshot 2024-12-14 211719](https://github.com/user-attachments/assets/4f51b3c8-ac2f-42c9-b83e-fde7a3b67aef)

df.isnull().sum()


![Screenshot 2024-12-14 211733](https://github.com/user-attachments/assets/148e6ef3-0d98-49c7-aad1-e4943d8eb5d9)

import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer

df = pd.read_csv("/content/drive/MyDrive/titanic_dataset.csv")

from scipy.stats import chi2_contingency
import seaborn as sns
tips=sns.load_dataset("tips")
tips.head()


![Screenshot 2024-12-14 211747](https://github.com/user-attachments/assets/b5c555e4-f3da-4678-9829-e902def66e43)

contigency_table=pd.crosstab(tips["sex"],tips["time"])
contigency_table


![Screenshot 2024-12-14 211800](https://github.com/user-attachments/assets/09a66df2-077a-4d50-b1e3-460219f33268)

chi2,p,_,_=chi2_contingency(contigency_table)
print(f"chi-Squared Statistic: {chi2}")
print(f"p-value: {p}")



![Screenshot 2024-12-14 211814](https://github.com/user-attachments/assets/2d8935ce-3b2e-4c5c-96ae-47b40b4f2dd5)

import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif

      data={
      'Feature1':[1,2,3,4,5],
      'Feature2':['A','B','C','A','B'],
      'Feature3':[0,1,1,0,1],
      'Target':[0,1,1,0,1]
      }
      df=pd.DataFrame(data)
      x=df[['Feature1','Feature3']]
      y=df['Target']
      selector = SelectKBest(score_func=f_classif, k=2)
      selector.fit(x, y)
      selector_feature_indices=selector.get_support(indices=True)
      selected_features=x.columns[selector_feature_indices]
      print("Selected Features:")
      print(selected_features)
      print("selected_Features:")
      print(selected_features) # Assuming selected_features holds the desired value

      
  ![Screenshot 2024-12-14 211928](https://github.com/user-attachments/assets/ef83cea3-605b-4c5b-9306-b90d5ec1b248)

 OUTPUT SCREENSHOTS HERE
# RESULT:
       The above code is excuted successfully.

       
