# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/bmi.csv")
df.head()
df.dropna()
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
df[['Height','Weight']]=sc.fit_transform(df[['Height','Weight']])
df.head(10)
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)
from sklearn.preprocessing import Normalizer
scaler=Normalizer()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df
df1=pd.read_csv("/content/bmi.csv")
from sklearn.preprocessing import MaxAbsScaler
scaler=MaxAbsScaler()
df1[['Height','Weight']]=scaler.fit_transform(df1[['Height','Weight']])
df1
df2=pd.read_csv("/content/bmi.csv")
from sklearn.preprocessing import RobustScaler
scaler=RobustScaler()
df2[['Height','Weight']]=scaler.fit_transform(df2[['Height','Weight']])
df2.head()

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
data=pd.read_csv('/content/income(1) (1).csv',na_values=[" ?"])
data
data.isnull().sum()
missing=data[data.isnull().any(axis=1)]
missing
data2 = data.dropna(axis=0)
data2
sal=data['SalStat']
data2['SalStat']=data2['SalStat'].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])

sal2=data2['SalStat']
dfs=pd.concat([sal,sal2],axis=1)
dfs
data2
new_data=pd.get_dummies(data2, drop_first=True)
new_data
columns_list=list(new_data.columns)
print(columns_list)
features=list(set(columns_list)-set(['SalStat']))
print(features)
y=new_data['SalStat'].values
print(y)
x = new_data[features].values
print(x)
train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.3, random_state=0)
KNN_classifier=KNeighborsClassifier(n_neighbors = 5)
KNN_classifier.fit(train_x,train_y)
prediction = KNN_classifier.predict(test_x)
confusionMmatrix = confusion_matrix(test_y, prediction)
print(confusionMmatrix)
accuracy_score=accuracy_score(test_y, prediction)
print(accuracy_score)

print('Misclassified samples: %d' % (test_y != prediction).sum())
data.shape

(31978, 13)

FEATURE SELSECTION TECHNIQUES:
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)

chi2, p, _, _ = chi2_contingency(contingency_table)
print(f"Chi-Square Statistic: {chi2}")
print(f"P-value: {p}")
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
data={
    'Feature1':[1,2,3,4,5],
    'Feature2': ['A','B','C','A','B'],
    'Feature3':[0,1,1,0,1],
    'Target' :[0,1,1,0,1]
}
df=pd.DataFrame(data)
X=df[['Feature1','Feature3']]
y=df['Target']
selector=SelectKBest(score_func=mutual_info_classif, k=1)
X_new = selector.fit_transform (X,y)
selected_feature_indices = selector.get_support(indices=True)
selected_features = X.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```

<img width="273" height="63" alt="image" src="https://github.com/user-attachments/assets/1e5dcff5-2045-46fb-8698-4714f618fb03" />
<img width="266" height="193" alt="image" src="https://github.com/user-attachments/assets/b99db8b5-bc7f-49e0-9e28-d07b2fdd4fd0" />
<img width="423" height="340" alt="image" src="https://github.com/user-attachments/assets/8317ac95-2070-402e-95e1-685e2aac2745" />
<img width="285" height="314" alt="image" src="https://github.com/user-attachments/assets/4e99dfaf-f9ef-4df7-a9cb-2c9f10b4fe5f" />
<img width="383" height="303" alt="image" src="https://github.com/user-attachments/assets/25401fff-f5f0-472b-b0f2-eda7131c7027" />
<img width="361" height="363" alt="image" src="https://github.com/user-attachments/assets/e023e59d-102a-46f1-bfd4-d0e282c631d7" />
<img width="377" height="352" alt="image" src="https://github.com/user-attachments/assets/3400efbe-6d7a-4ce0-9896-2358546240e2" />
<img width="312" height="188" alt="image" src="https://github.com/user-attachments/assets/1a8c0f6c-22c7-4fe4-96b7-2781600cafbf" />
<img width="1068" height="334" alt="image" src="https://github.com/user-attachments/assets/eba7644f-67b9-4956-bf3b-2a9720cb24bc" />
<img width="193" height="203" alt="image" src="https://github.com/user-attachments/assets/23c0d274-5058-4511-9b96-483080bdf6f5" />
<img width="984" height="306" alt="image" src="https://github.com/user-attachments/assets/6cf671be-9125-4ae5-8299-d00150b46b03" />
<img width="1095" height="328" alt="image" src="https://github.com/user-attachments/assets/c2a066dc-e25e-4320-8aa3-a919140780a7" />
<img width="1082" height="271" alt="image" src="https://github.com/user-attachments/assets/dc18905b-13cf-40a7-859b-2a7af31c5233" />
<img width="433" height="334" alt="image" src="https://github.com/user-attachments/assets/53c4ee55-191e-441a-80ef-c8a1adb746e0" />
<img width="995" height="352" alt="image" src="https://github.com/user-attachments/assets/13c64bcc-f5e6-4abb-b2e8-b96c15cf5622" />
<img width="1047" height="354" alt="image" src="https://github.com/user-attachments/assets/7126ad5a-cbad-4495-b2fa-d56239c7461a" />
<img width="1075" height="38" alt="image" src="https://github.com/user-attachments/assets/060611ff-685f-43e6-90a8-7636d4e575fa" />
<img width="1070" height="34" alt="image" src="https://github.com/user-attachments/assets/73e43c4c-956d-45dc-9e4a-5bd84b36d91f" />
<img width="249" height="48" alt="image" src="https://github.com/user-attachments/assets/ff5c2f27-0139-44da-ac52-6fb9b817c4c6" />
<img width="322" height="137" alt="image" src="https://github.com/user-attachments/assets/7da88012-d14d-419e-b2fe-fe855e394726" />
<img width="245" height="81" alt="image" src="https://github.com/user-attachments/assets/4e801d41-1313-4bf9-89c3-c4631e137b10" />
<img width="145" height="43" alt="image" src="https://github.com/user-attachments/assets/d676fc23-180d-44bf-99ad-3b26a1d8e62a" />
<img width="245" height="81" alt="image" src="https://github.com/user-attachments/assets/bda2083f-dd0d-4bbd-a212-2d0d0441c740" />
<img width="322" height="137" alt="image" src="https://github.com/user-attachments/assets/ccce2df4-0757-459a-a37a-1aab5ada2459" />
<img width="145" height="43" alt="image" src="https://github.com/user-attachments/assets/03b6b809-e96a-4cc1-849b-1f3ad17b2cf8" />
<img width="257" height="39" alt="image" src="https://github.com/user-attachments/assets/b189ab85-50b2-47e1-9c43-e0aec8e5245b" />
<img width="335" height="50" alt="image" src="https://github.com/user-attachments/assets/4936b2e3-39ab-48b4-8f69-f47b668a9332" />
<img width="392" height="175" alt="image" src="https://github.com/user-attachments/assets/2c07b15d-b7aa-45ed-b9f9-11af9a6fb5dd" />
<img width="236" height="83" alt="image" src="https://github.com/user-attachments/assets/e2c36fca-4395-428b-8e80-3832731fc296" />
<img width="263" height="53" alt="image" src="https://github.com/user-attachments/assets/7b6be89c-34f7-4ef2-85f1-5b08d060018f" />

# RESULT:
To read the given data and perform Feature Scaling and Feature Selection process and save the data to a file has been executed successfully.

