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
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

data=pd.read_csv("/content/income(1) (1).csv",na_values=[ " ?"])
data
```
![322338017-b544c435-1cc1-4bc6-83c9-de2945348808](https://github.com/AdhithiyanK/EXNO-4-DS/assets/121029258/c37ba602-7090-41bc-8884-ceebd57170fb)

```

data.isnull().sum()

```
![322338037-40b1ab98-5a1a-41a1-b943-102b7c4cabed](https://github.com/AdhithiyanK/EXNO-4-DS/assets/121029258/9017fd3b-24e9-4c35-827f-51a3714bdc26)

```

missing=data[data.isnull().any(axis=1)]
missing
```
![322338066-a5fe88ab-c993-4c97-b249-cffea5a21a54](https://github.com/AdhithiyanK/EXNO-4-DS/assets/121029258/1ace2413-6173-49bc-94f9-730266e41cef)

```

data2=data.dropna(axis=0)
data2
```
![322338086-40a10680-63a6-4f18-87ae-517ceda76ca9](https://github.com/AdhithiyanK/EXNO-4-DS/assets/121029258/e20e1f29-1012-4c87-9133-d33ac72d8a73)
```
sal=data["SalStat"]

data2["SalStat"]=data["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
```
![322338114-e59ce957-1bdc-4455-97a5-15d66108b864](https://github.com/AdhithiyanK/EXNO-4-DS/assets/121029258/e91284cd-6b9a-42d4-b571-38bc092ab906)
```
sal2=data2['SalStat']

dfs=pd.concat([sal,sal2],axis=1)
dfs
```
![322338170-f8435063-835b-4eba-af2e-c46c67ea55e9](https://github.com/AdhithiyanK/EXNO-4-DS/assets/121029258/e9bf0ddd-c170-4871-af35-600f26f938a3)
```
data2
```
![322338187-c034e83a-8e21-400e-bc40-103e3da86d0e](https://github.com/AdhithiyanK/EXNO-4-DS/assets/121029258/fe030dad-804e-4ca4-8bc4-56d49c809f83)

```
new_data=pd.get_dummies(data2, drop_first=True)
new_data
```
![322338213-f21819e3-a5bd-47e6-b1b7-9bc08b64bed9](https://github.com/AdhithiyanK/EXNO-4-DS/assets/121029258/da2fed02-2db5-4232-a1f9-dd448891903e)
```
columns_list=list(new_data.columns)
print(columns_list)
```
![322338249-8af6f5ce-4d99-4ed6-9371-730aeaa5a56b](https://github.com/AdhithiyanK/EXNO-4-DS/assets/121029258/9eabe24e-93f6-43bc-b18f-6df558e6b3b7)

```

features=list(set(columns_list)-set(['SalStat']))
print(features)
```
![322338261-5f31a677-7d30-417a-8044-d5db741cafbf](https://github.com/AdhithiyanK/EXNO-4-DS/assets/121029258/5a200ff0-4144-4da8-8cfd-3382467fec1b)

```
y=new_data['SalStat'].values
print(y)
```
![322338286-f4c779af-4c87-449e-9daa-be5d8d275212](https://github.com/AdhithiyanK/EXNO-4-DS/assets/121029258/4e038f21-d43e-4fc7-90c5-1a5008271b95)

```
x=new_data[features].values
print(x)
```
![322338321-4154db03-4c87-4b98-a13b-964f19bee9b0](https://github.com/AdhithiyanK/EXNO-4-DS/assets/121029258/7387c76a-77e5-4aec-96da-3ea48e2420ae)

```

train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)

KNN_classifier=KNeighborsClassifier(n_neighbors = 5)

KNN_classifier.fit(train_x,train_y)
```
![322338343-e5e02520-eb39-436c-ac2e-e43048c1d672](https://github.com/AdhithiyanK/EXNO-4-DS/assets/121029258/c424e908-f0ad-419e-ad1c-f1324d4b6636)

```

prediction=KNN_classifier.predict(test_x)

confusionMatrix=confusion_matrix(test_y, prediction)
print(confusionMatrix)
```
![322338371-a6eedfe3-aedd-4500-958f-6faafd54f464](https://github.com/AdhithiyanK/EXNO-4-DS/assets/121029258/0152655c-039d-484a-b3fb-425bfec47bd3)
```

accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)

```
![322338387-0e56ff41-2f35-4d01-b479-53547391567b](https://github.com/AdhithiyanK/EXNO-4-DS/assets/121029258/ade4e726-f559-405d-88b9-31f076c34b3c)
```

print("Misclassified Samples : %d" % (test_y !=prediction).sum())
```
![322338405-4af5ed3f-362a-40c6-a438-c89f31584e51](https://github.com/AdhithiyanK/EXNO-4-DS/assets/121029258/705c8213-dde0-4ce9-9860-6bb5037110cb)

```

print("Misclassified Samples : %d" % (test_y !=prediction).sum())

```
![322338405-4af5ed3f-362a-40c6-a438-c89f31584e51](https://github.com/AdhithiyanK/EXNO-4-DS/assets/121029258/a68b913e-2560-43cb-a52a-c9462bc6c1ed)

```

data.shape
```
![322338420-1986f990-26e6-4b42-acfc-b2a6e52f8042](https://github.com/AdhithiyanK/EXNO-4-DS/assets/121029258/123092e1-2326-477f-a18a-59ab143aee83)

```

import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
data={
    'Feature1': [1,2,3,4,5],
    'Feature2': ['A','B','C','A','B'],
    'Feature3': [0,1,1,0,1],
    'Target'  : [0,1,1,0,1]
}

df=pd.DataFrame(data)
x=df[['Feature1','Feature3']]
y=df[['Target']]

selector=SelectKBest(score_func=mutual_info_classif,k=1)
x_new=selector.fit_transform(x,y)

selected_feature_indices=selector.get_support(indices=True)

selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```
![322338454-20777b0d-3cdb-4ae9-80e4-1f76ed093191](https://github.com/AdhithiyanK/EXNO-4-DS/assets/121029258/a0b63c88-0472-44e6-a9bb-92887b857aea)

```

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```
![322338479-6d6f7ff2-b1da-4568-9cd1-cb6fa9553cd6](https://github.com/AdhithiyanK/EXNO-4-DS/assets/121029258/daecf0a2-defc-436f-97b9-653a6bbcfe05)

```

tips.time.unique()

```
![322338497-f77bc757-8a31-4a5d-be15-5a447e6549c6](https://github.com/AdhithiyanK/EXNO-4-DS/assets/121029258/9bd463aa-3240-4e7c-8c3b-f3551e554c69)

```

contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
```
![322338518-06365e9f-f51b-4cf6-ab04-8a136726a025](https://github.com/AdhithiyanK/EXNO-4-DS/assets/121029258/f88a8b2b-b211-4aac-b150-e6acee130eb1)

```

chi2,p,_,_=chi2_contingency(contingency_table)
print(f"Chi-Square Statistics: {chi2}")
print(f"P-Value: {p}")
```
![322338543-6adc4da7-421c-458f-9ec6-f6158aa6f731](https://github.com/AdhithiyanK/EXNO-4-DS/assets/121029258/5d15a654-bbdb-45c6-b10b-b58f3b7362f5)

# RESULT:
       Thus, Feature selection and Feature scaling has been used on thegiven dataset.
