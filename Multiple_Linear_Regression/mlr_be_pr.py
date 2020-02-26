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

# Automatic Backward Elimination Using

import statsmodels.api as sm

X = np.append(arr = np.ones([50,1],dtype=int), values = X, axis=1)
X_opt = X[:,[0 , 1, 2, 3, 4, 5]]

def BackwardElimination(x,sl) :
    numVars = len(x[0])
    temp = np.zeros((50, 6)).astype(int)
    for i in range(0, numVars):
        regressor_OLSpr = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLSpr.pvalues).astype(float)
        adjR_before = regressor_OLSpr.rsquared_adj.astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLSpr.pvalues[j].astype(float) == maxVar):
                    temp[:, j] = x[:, j]
                    x = np.delete(x, j, 1)
                    tmp_regressor = sm.OLS(y, x).fit()
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)
                    if (adjR_before >= adjR_after):
                        x_rollback = np.hstack((x, temp[:, [0, j]]))
                        x_rollback = np.delete(x_rollback, j, 1)
                        print(regressor_OLSpr.summary())
                        return x_rollback
                    else:
                        continue
    regressor_OLSpr.summary()
    return x

SL = 0.05
X_Modeledpr = BackwardElimination(X_opt,SL)