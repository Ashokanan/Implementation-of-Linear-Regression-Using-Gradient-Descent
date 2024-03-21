# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: A.Ashokkumar 
RegisterNumber: 212223080006

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def linear_regression(X1,y,learning_rate=0.01,num_iters=1000):
    X=np.c_[np.ones(len(X1)),X1]
    theta = np.zeros(X.shape[1]).reshape(-1,1)
    for _ in range(num_iters):
        predictions = (X).dot(theta).reshape(-1,1)
        errors = (predictions-y).reshape(-1,1)
        theta-=learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta
    
data=pd.read_csv('50_Startups.csv',header=None)
data.head()
X = (data.iloc[1:,:-2].values)
print(X)

X1=X.astype(float)
scaler = StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
print(y)

X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print(X1_Scaled)
print(Y1_Scaled)

theta = linear_regression(X1_Scaled,Y1_Scaled)

new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled = scaler.fit_transform(new_data)
prediction = np.dot(np.append(1,new_Scaled),theta)
prediction = prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(f"Predicted Value:{pre}") 
```

## Output:
![WhatsApp Image 2024-03-14 at 22 01 32_4e53d12d](https://github.com/Ashokanan/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/160997973/729963f3-3ee4-440b-b355-fb27551367bc)

![WhatsApp Image 2024-03-14 at 22 01 32_7341a512](https://github.com/Ashokanan/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/160997973/53f638ed-7abb-4c0b-b510-9fab3e7691a3)
![WhatsApp Image 2024-03-14 at 22 01 32_005f46e1](https://github.com/Ashokanan/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/160997973/dfff6afc-5882-49dd-836b-e925bcd7fe4e)
![WhatsApp Image 2024-03-14 at 22 01 34_1fc4c579](https://github.com/Ashokanan/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/160997973/e42a80a5-d788-4a71-94cb-5bbb5c51524d)
![WhatsApp Image 2024-03-14 at 22 01 34_446d8f44](https://github.com/Ashokanan/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/160997973/193b474f-7e0d-4137-9d25-1c2cec4d9ded)
![WhatsApp Image 2024-03-14 at 22 01 34_052189f2](https://github.com/Ashokanan/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/160997973/e6230a25-7916-428b-82ea-6bf229dc3862)











## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
