#backward elimination regression
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np

from sklearn.linear_model import LogisticRegression

dataset = pd.read_csv('Churn.csv')

dataset = dataset[dataset['TotalCharges']!=' ']
dataset['TotalCharges'] = pd.to_numeric(dataset['TotalCharges'])

for i in range(1,len(dataset.columns)):
    print(dataset.columns[i],"--",dataset[dataset.columns[i]].unique())

labels = [1,3,4,6,7,8,9,10,11,12,13,14,15,16,17,20]

for i in labels:
    dataset.iloc[:,i] = LabelEncoder().fit_transform(dataset.iloc[:,i])


dataset = dataset.drop(['customerID'], axis=1)


X=dataset.iloc[:,:-1]
y=dataset.iloc[:,-1].values
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.30, random_state=0)


logit  = LogisticRegression(random_state=0)
logit.fit(X_train, y_train)
#r_square = logit.score(df_val.iloc[:,:-1], df_val["Churn"])
pred = logit.predict(X_test)

import statsmodels.formula.api as sm
X = np.append(arr=np.ones((7032,1)).astype(int), values=X, axis=1)
X_opt = X[:,:]
logit_OLS = sm.OLS(endog=y, exog=X_opt).fit()
logit_OLS.summary()

X_opt = X[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,15,16,17,18,19]]
logit_OLS = sm.OLS(endog=y, exog=X_opt).fit()
logit_OLS.summary()

X_opt = X[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,15,16,17,18,19]]
logit_OLS = sm.OLS(endog=y, exog=X_opt).fit()
logit_OLS.summary()

X_opt = X[:,[0,1,2,4,5,6,7,8,9,10,11,12,15,16,17,18,19]]
logit_OLS = sm.OLS(endog=y, exog=X_opt).fit()
logit_OLS.summary()

X_opt = X[:,[0,2,4,5,6,7,8,9,10,11,12,15,16,17,18,19]]
logit_OLS = sm.OLS(endog=y, exog=X_opt).fit()
logit_OLS.summary()

X_opt = X[:,[0,2,4,5,6,7,8,9,10,11,12,15,16,18,19]]
logit_OLS = sm.OLS(endog=y, exog=X_opt).fit()
logit_OLS.summary()
#---------------------
X_opt = X[:,[0,2,5,6,7,8,9,10,11,12,15,16,18,19]]
logit_OLS = sm.OLS(endog=y, exog=X_opt).fit()
logit_OLS.summary()