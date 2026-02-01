import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv(r'C:\Users\user\Downloads\Salary_Data.csv')

x = dataset.iloc[:,:-1]
y = dataset.iloc[:,-1]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size= 0.2,random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)

plt.scatter(x_test,y_test,color = 'red')
plt.plot(x_train,regressor.predict(x_train), color = 'blue')
plt.title('Salary vs Experience(Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

m_slope = regressor.coef_
print(m_slope)

c_intercept = regressor.intercept_
print(c_intercept)

y_12 = m_slope*12+c_intercept
print(y_12)

y_20 = m_slope*20+c_intercept
print(y_20)

#bias_score or training_score
bias_score = regressor.score(x_train,y_train)
print(bias_score)

variance_score = regressor.score(x_test, y_test)
print(variance_score)

## Statstical 

dataset.mean()

dataset['Salary'].mean()

dataset.median()
dataset['Salary'].median()

dataset.mode()
dataset['Salary'].mode()

dataset.var()
dataset['YearsExperience'].var()
dataset['Salary'].var()

## Standard deviation

dataset.std()
dataset['Salary'].std()

## Coefficient of variation(cv)

from scipy.stats import variation
variation(dataset.values)

variation(dataset['Salary'])

## Correlation
dataset.corr()
## heat map in output we ger diagnal values as 1

dataset['Salary'].corr(dataset['YearsExperience'])

## Skewness

dataset.skew()
dataset['Salary'].skew()

## Standard Error
dataset.sem()

## Z-score
import scipy.stats as stats
dataset.apply(stats.zscore)
stats.zscore(dataset['Salary'])

## Anova
##SSR
y_mean = np.mean(y)
SSR = np.sum((y_pred-y_mean)**2)
print(SSR)
##SSE
y = y[0:6]
SSE = np.sum((y-y_pred)**2)
print(SSE)
## SST
mean_total = np.mean(dataset.values)
SST = np.sum((dataset.values-mean_total)**2)
print(SST)

## R square
r_square = 1-SSR/SST
print(r_square)

## Bias Score
bias = regressor.score(x_train,y_train)
print(bias)

## VARIANCE

variance = regressor.score(x_test,y_test)
print(variance)


## Save the model by pickle
import pickle
filename = 'linear_regression_model.pkl'

with open(filename,'wb') as file:
    pickle.dump(regressor,file)
    
print('Model has been pickled and saved as linear_regressor_model')
  