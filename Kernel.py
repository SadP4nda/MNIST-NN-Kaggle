import numpy as np
import scipy as sp
import pandas
import numpy.random as npr
from ForwardProp import ForwardProp,CostFunc
from BackProp import BackProp
from NumericalGrad import NumericalGradient

def RandInit(cols,rows,eps):
	return npr.rand(cols,rows+1) * 2 * eps - eps
def logicalVectorize(y):
	newY = np.zeros([y.shape[0],10])
	for i in range(y.shape[0]):
		newY[i][y[i]] = 1
	return newY
Location = 'train.csv'
df = pandas.read_csv(Location)

y = np.asarray(df.as_matrix(['label']))
X = df.as_matrix()
X = np.asarray(np.delete(X, 0 ,axis = 1),dtype=np.float64)
Theta1 = RandInit(50,X.shape[1],.012)
Theta2 = RandInit(10,50,.012)
y = logicalVectorize(y)
print(CostFunc(X,Theta1,Theta2,y))
yeet = BackProp(X, y, Theta1, Theta2)
bpTheta1 = np.transpose(yeet[0].reshape(yeet[0].size,order='F'))
bpTheta2 = np.transpose(yeet[1].reshape(yeet[1].size,order='F'))
grad = np.transpose(np.append(bpTheta1,bpTheta2))
print(grad.shape)
NumGrad = NumericalGradient(X,y,Theta1,Theta2)

diff = np.linalg.norm(NumGrad-grad)/np.linalg.norm(NumGrad+grad)
print(diff)