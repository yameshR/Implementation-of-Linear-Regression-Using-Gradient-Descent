# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import numpy, pandas modules and StandardScaler from sklearn module
2. Define the function linear_regression - with variables, learning rate and the number of times to be iterated
3. Read the csv file and create the dataframe
4. Assume the last column is your target variable 'y' and the preceding columns and define X1
5. Learn the model parameters
6. Predict the target value for a new data point

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: Yamesh R
RegisterNumber:  212222220059
*/
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate=0.1,num_iters=1000):
    X=np.c_[np.ones(len(X1)),X1]
    
    theta=np.zeros(X.shape[1]).reshape(-1,1)
    
    for _ in range(num_iters):
        #Calculate predictions
        predictions=(X).dot(theta).reshape(-1,1)
        #calculate errors
        errors=(predictions-y).reshape(-1,1)
        #Update theta using gradient descent
        theta-=learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta

data=pd.read_csv("50_Startups.csv",header=None)
data.head()

#Assuming the last column is your target variable 'y' and the preceding columns
X=(data.iloc[1:,:-2].values)
X1=X.astype(float)

scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print(X)
print(X1_Scaled)

#Learn model Parameters
theta=linear_regression(X1_Scaled,Y1_Scaled)
#Predict target value for a new data point
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(prediction)
print(f"Predicted value: {pre}")

```

## Output:
![linear regression using gradient descent](https://github.com/joeljohnjobinse/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/138955488/765eb6e3-204a-46d0-9ec0-3aa546eb93d3)

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
