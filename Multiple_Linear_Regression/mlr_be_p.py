import numpy as np
import pandas as pd

dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X), dtype=np.float)

X = X[:, 1:]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

y_pred = regressor.predict(X_test)

# Automatic Backward Elimination Using p-Values only

import statsmodels.api as sm

X = np.append(arr = np.ones([50,1],dtype=int), values = X, axis=1)
X_opt = X[:,[0 , 1, 2, 3, 4, 5]]


def backwardElimination(x,sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLSp = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLSp.pvalues).astype(float)
        if maxVar>sl :
            for j in range(0, numVars-i):
                if(regressor_OLSp.pvalues[j].astype(float)==maxVar):
                    x = np.delete(x,j,1)
    regressor_OLSp.summary()
    return x

SL = 0.05
X_Modeled = backwardElimination(X_opt, SL)