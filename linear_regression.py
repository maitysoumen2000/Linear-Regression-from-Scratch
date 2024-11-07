import numpy as np
import pandas as pd

df = pd.read_csv("student_data.csv")

df.head(5)

df.tail(5)

df.columns

Dependent_Columns = "Performance"
Independent_Columns = [col for col in df.columns if col != Dependent_Columns]
print(f"Dependent Variable is : {Dependent_Columns}")
print(f"Independent Variable is : {Independent_Columns}")

df.info()

df.describe()

df.isnull().sum()

[features for features in df.columns if df[features].isnull().sum()>0]

df.shape

# Commented out IPython magic to ensure Python compatibility.

import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline
sns.heatmap(df.isnull(),yticklabels=False, cbar=False,cmap = 'viridis')

df.columns

sns.heatmap(df.describe(), annot= True, fmt='.2f')

info = df.corr()
sns.heatmap(info, annot=True, fmt='.2f')

df = df.drop('Extracurricular Activities', axis=1)

df.head(5)

df['Sample Question Papers Practiced'].isnull().any()

df['Performance'].isnull().any()

df = df['Hours Studied'].shape

print(type(df))

if not isinstance(df, pd.DataFrame):
    df = pd.read_csv('student_data.csv')

print(type(df))

x = df.drop('Extracurricular Activities',axis=1)

x = df.drop('Performance',axis=1)
y = df['Performance']

x = df.drop('Performance',axis=1)
y = df['Performance']

x = df.drop('Extracurricular Activities',axis=1)

print(x.columns)

num_features = min(len(x.columns), 4)

plt.figure(figsize=(12, 12))

for i, col in enumerate(x.columns[:num_features]):
    plt.subplot(2, 2, i+1)
    sns.scatterplot(x=x[col], y=y, color='red')
    plt.title(f"Scatterplot of {col} vs. Performance")

plt.tight_layout()
plt.show()

num_features = len(x.columns)

num_rows = (num_features + 1) // 2
num_cols = 2

plt.figure(figsize=(12, 6 * num_rows))

for i, col in enumerate(x.columns):
    plt.subplot(num_rows, num_cols, i+1)

    if x[col].dtype in ['int64', 'float64']:
        sns.distplot(x[col])
        plt.title(f"Distribution of {col}")
    else:
        plt.title(f"Skipped: {col} (Non-numeric data)")

plt.tight_layout()
plt.show()

info = x.describe()
sns.heatmap(info,annot=True,fmt='.2f')

info = x.corr()
sns.heatmap(info,annot=True,fmt='.2f')

print(x.columns)

x.shape

y.shape

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

import numpy as np
import matplotlib.pyplot as plt

class LinearRegressionCustom:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None
        self.loss_history = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        X = (X - X.mean(axis=0)) / X.std(axis=0)
        y = (y - y.mean()) / y.std()

        for epoch in range(self.num_iterations):
            y_predicted = np.dot(X, self.weights) + self.bias

            mse = np.mean((y - y_predicted) ** 2)
            self.loss_history.append(mse)

            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

    def score(self, X, y):
        y_predicted = self.predict(X)
        mse = np.mean((y - y_predicted) ** 2)
        return 1 - (mse / np.var(y))

model = LinearRegressionCustom(learning_rate=0.01, num_iterations=1000)
model.fit(X_train, y_train)
score = model.score(X_test, y_test)
print(f"Custom Linear Regression Model Score: {score}")

plt.figure(figsize=(10, 6))
plt.plot(range(1, model.num_iterations + 1), model.loss_history, marker='o', linestyle='-')
plt.title('Loss vs. Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.show()

ypred = model.predict(X_test)
from sklearn.metrics import mean_absolute_error
print("MAE",mean_absolute_error(y_test,ypred))
from sklearn.metrics import mean_squared_error
print("MSE",mean_squared_error(y_test,ypred))
print("RMSE",(np.sqrt(mean_squared_error(y_test,ypred))))
ypred1 = ypred
from sklearn.metrics import r2_score
print("R2 score:",r2_score(y_test, ypred))

sns.scatterplot(x = y_test,y = ypred,color='orange',label='Predicted')
x = np.arange(0,np.max(y_test),0.1)
y = np.arange(0,np.max(y_test),0.1)
sns.lineplot(x=x,y=y,label='Actual')
plt.xlabel("Actual Collection")
plt.ylabel("Predicted Collection")
plt.title('LinearRegression')
plt.show()