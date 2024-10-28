# Importing the libraries
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding categorical data
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])

# Apply one-hot encoding to the first column
ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(), [3])], 
    remainder='passthrough'
)
X = ct.fit_transform(X)
X = X[:, 1:]  # Avoiding the dummy variable trap

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Ensure all X_train columns are numeric
X_train = pd.DataFrame(X_train).apply(pd.to_numeric, errors='coerce').values

# Ensure y_train is a 1D array and numeric
y_train = pd.Series(y_train).apply(pd.to_numeric, errors='coerce').values.flatten()

# Print shapes and types for debugging
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_train dtype:", X_train.dtype)
print("y_train dtype:", y_train.dtype)

# Fitting Multiple Linear Regression to the Training set
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Building the optimal model using Backward Elimination
X_train = np.append(arr=np.ones((X_train.shape[0], 1)).astype(int), values=X_train, axis=1)

# Selecting optimal features
X_opt = X_train[:, [0, 1, 2, 3, 4, 5]]

try:
    regressor_OLS = sm.OLS(endog=y_train, exog=X_opt).fit()
    print(regressor_OLS.summary())
except TypeError as e:
    print("Error:", e)

# 挑最高的P value 剃除
# P value:是當虛無假設為真時，觀察到的數據或更極端的數據出現的概率。它的值範圍從 0 到 1。
# 先考慮踢出 x2，因為它的 P 值最高（0.850），表明它與因變數 y 之間的關係不顯著。
# 對於線性回歸分析，通常設定的顯著性水平是 0.05，因此 P 值大於 0.05 的變數通常被認為不顯著

X_opt = X_train [:, [0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y_train, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X_train [:, [0, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y_train, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X_train [:, [0, 3, 5]]
regressor_OLS = sm.OLS(endog = y_train, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X_train [:, [0, 3]]
regressor_OLS = sm.OLS(endog = y_train, exog = X_opt).fit()
regressor_OLS.summary()