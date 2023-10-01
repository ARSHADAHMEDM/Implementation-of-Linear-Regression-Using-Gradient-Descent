# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required library and read the dataframe.
2. Write a function computeCost to generate the cost function.
3. Perform iterations og gradient steps with learning rate.
4. Plot the Cost function using Gradient Descent and generate the required graph.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: ARSHAD AHMED M
RegisterNumber:  212222040014
*/
```
``` py
#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#reading the data
data = pd.read_csv("ex1.txt", header=None)
data
plt.scatter(data[0],data[1])
#representing the graph to make more clear
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
#label
plt.xlabel("Population of City (10,000s)")
plt.ylabel("Profit ($10,000)")

data.shape

def computeCost(X,y,theta):
    """
    Take in a numpy array X,y,theta and generate the cost function using
    gradient descent in a linear Regression Model
    
    """
    m=len(y)
    h=X.dot(theta)
    square_err=(h - y)**2
    return 1/(2*m)*np.sum(square_err)#returning 1

data_n=data.values
m=data_n[:,0].size
X=np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1)
y=data_n[:,1].reshape(m,1)
theta=np.zeros((2,1))

computeCost(X,y,theta)

def gradientDescent(x,y,theta,alpha,num_iters):
  m=len(y)
  J_history=[]

  for i in range(num_iters):
    predictions = x.dot(theta)
    error = np.dot(x.transpose(),(predictions -y))
    descent=alpha*1/m*error
    theta-=descent
    J_history.append(computeCost(x,y,theta))

  return theta,J_history

theta,J_history = gradientDescent(X,y,theta,0.01,1500)
print("h(x) ="+str(round(theta[0,0],2))+"+"+str(round(theta[1,0],2))+"x1")

plt.plot(J_history)
plt.xlabel("Iteration")
plt.ylabel("$J(\theta)$")
plt.title("Cost frunction using Gradient Descent")

def predict(x,theta):
  predictions=np.dot(theta.transpose(),x)
  return predictions[0]

predict1=predict(np.array([1,3.5]),theta)*10000
print("For population = 35,000, we predict a profit of $"+str(round(predict1,0)))

predict2=predict(np.array([1,7]),theta)*10000
print("For population = 70,000, we predict a profit of $"+str(round(predict2,0)))
```

## Output:
![image](https://github.com/ARSHADAHMEDM/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/128116503/538cf001-0e31-4d0a-b2d3-71819177263f)
![image](https://github.com/ARSHADAHMEDM/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/128116503/cd18994b-0103-430d-b193-b31dab8fb254)
![image](https://github.com/ARSHADAHMEDM/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/128116503/7455c31d-0d12-4bfc-866e-9f27c72e3baf)
![image](https://github.com/ARSHADAHMEDM/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/128116503/2b10836c-fbc0-4aae-98da-0a1c8facc6a3)
![image](https://github.com/ARSHADAHMEDM/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/128116503/23618762-08dc-464b-ae11-215f7ec05c6b)
![image](https://github.com/ARSHADAHMEDM/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/128116503/f0ac7fa7-0c56-46c2-b3c0-bfdc13c371d4)
![image](https://github.com/ARSHADAHMEDM/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/128116503/fc058013-d6a6-4827-a461-1c3eeb8b3689)
![image](https://github.com/ARSHADAHMEDM/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/128116503/a5823021-007c-43aa-a709-072e56934cec)

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
``
