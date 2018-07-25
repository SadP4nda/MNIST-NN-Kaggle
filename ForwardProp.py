import numpy as np
from sigmoid import sigmoid
from math import log
def ForwardProp(X,Theta1,Theta2):
	X = np.concatenate((np.ones([X.shape[0],1]),X),1)
	a2 = sigmoid(X.dot(np.transpose(Theta1)))
	a2 = np.concatenate((np.ones([a2.shape[0],1]),a2),1)
	return sigmoid(a2.dot(np.transpose(Theta2)))
def CostFunc(X,Theta1,Theta2,y,L = 0):
	m = X.shape[0]
	hyp = ForwardProp(X, Theta1, Theta2)
	return (-1 / m) * np.sum(np.sum(np.multiply(y ,np.log(hyp)) + np.multiply((1 - y) , np.log(1 - hyp)))) + (L / (2 * m)) * (np.sum(np.sum(Theta1)) + np.sum(np.sum(Theta2)))
