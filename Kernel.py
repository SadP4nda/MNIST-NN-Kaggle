import numpy as np
import scipy as sp
from scipy.optimize import fmin_cg, fmin_l_bfgs_b,fmin_bfgs
import pandas
import numpy.random as npr
from ForwardProp import ForwardProp, CostFunc
import time
from BackProp import BackProp
from GradDescent import GradientDescent
from scipy.io import loadmat
def RandInit(cols,rows,eps):
	return npr.rand(cols,rows) * 2 * eps - eps
def logicalVectorize(y):
	newY = np.zeros([y.shape[0],10])
	for i in range(y.shape[0]):
		newY[i][y[i]] = 1
	return newY
Location = 'train.csv'
df = pandas.read_csv(Location)
hidden_layer_nodes = 50
hidden_layer_2_nodes = 50
classification_nodes = 10
y = np.asarray(df['label'].values)
X = df.values
X = np.asarray(np.delete(X, 0 ,axis = 1))
X = (X - np.mean(X))/np.std(X)
# arrs = loadmat('ex4data1.mat')
# y = arrs['y']
# for i in range(y.size):
# 	if y[i] == 10:
# 		y[i] = 0
# X = arrs['X']
eps1 = np.sqrt(6)/np.sqrt((hidden_layer_nodes+(X.shape[1]+1)))
eps2 = np.sqrt(6)/np.sqrt((classification_nodes + (hidden_layer_nodes+1)))
eps3 = np.sqrt(6)/np.sqrt(((hidden_layer_nodes+1) + hidden_layer_2_nodes))
Theta1 = RandInit(hidden_layer_nodes,X.shape[1]+1,eps1).reshape(hidden_layer_nodes*(X.shape[1]+1),order='F')
Theta2 = RandInit(hidden_layer_2_nodes,hidden_layer_nodes+1,eps1).reshape(hidden_layer_2_nodes*(hidden_layer_nodes+1),order='F')
Theta3 = RandInit(classification_nodes,hidden_layer_nodes+1,eps2).reshape(classification_nodes*(hidden_layer_nodes+1),order='F')
ThetaVect = np.append(Theta1,Theta2)
ThetaVect = np.append(ThetaVect,Theta3)
y = logicalVectorize(y)
print(ThetaVect)
print(y)
test = np.argmax(y,axis=1)
args=(X,y,X.shape[1],hidden_layer_nodes,hidden_layer_2_nodes,classification_nodes,1)
a = time.time()
grad = fmin_cg(CostFunc,ThetaVect,fprime=BackProp,args=args, maxiter = 100)
b = time.time()
print("Time for Conjugate Gradient Optimization %.6f" % (b-a))
print(y)
pred = ForwardProp(grad,X,y,X.shape[1],hidden_layer_nodes,hidden_layer_2_nodes,classification_nodes)
print("Error of Current Gradient %.8f" % (CostFunc(grad, X, y, X.shape[1], hidden_layer_nodes, hidden_layer_2_nodes, classification_nodes, 0)))
print(pred)
print(np.argmax(y,axis=1))
print(np.argmax(pred,axis=1))
print(pred[1])
pred_numbers = np.argmax(pred,axis=1)
print(np.sum(pred_numbers == test)/y.shape[0])