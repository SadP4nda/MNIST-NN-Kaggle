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
from Kernel import hidden_layer_2_nodes,hidden_layer_nodes, classification_nodes
Theta1 = pandas.read_csv('Theta1.csv')
Theta1 = Theta1.drop(Theta1.columns[0],axis=1)
Theta2 = pandas.read_csv('Theta2.csv')
Theta2 = Theta2.drop(Theta2.columns[0],axis=1)
Theta3 = pandas.read_csv('Theta3.csv')
Theta3 = Theta3.drop(Theta3.columns[0],axis=1)
print(Theta1)
Theta1 = np.asarray(Theta1.values)
Theta2 = np.asarray(Theta2.values)
Theta3 = np.asarray(Theta3.values)
print(Theta1.shape)
print(Theta2.shape)
print(Theta3.shape)
Theta1 = Theta1.reshape((Theta1.size,1), order='F')
Theta2 = Theta2.reshape((Theta2.size,1), order='F')
Theta3 = Theta3.reshape((Theta3.size,1),order = 'F')
ThetaVect = np.append(Theta1,Theta2)
ThetaVect = np.append(ThetaVect,Theta3)
X = np.asarray(pandas.read_csv('test.csv').values)
X = ((X) * ((1 + 1)/255)) - 1 
print(X)
pred = ForwardProp(ThetaVect,X,X.shape[1],hidden_layer_nodes,hidden_layer_2_nodes,classification_nodes)
print(pred)
pred_numbers = np.argmax(pred,axis=1)[np.newaxis].T
print(pred_numbers)
stuff = []
for i in range(X.shape[0]):
	stuff.append(i+1)
stuff = np.array(stuff)[np.newaxis].T

yeet = np.append(stuff,pred_numbers,axis=1)
print(yeet)
Ans = pandas.DataFrame(yeet,columns=['ImageId','Label'])


Ans.to_csv("D:\\Coding\\Kaggle\\MNIST\\Submission.csv")