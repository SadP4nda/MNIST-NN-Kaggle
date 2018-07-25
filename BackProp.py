import numpy as np
from ForwardProp import ForwardProp
from sigmoid import sigmoid
def sigPrime(z):
	return sigmoid(z) * (1-sigmoid(z))
def BackProp(X,y,Theta1,Theta2,L = 0):
	X = np.concatenate((np.ones([X.shape[0],1]),X),1)
	a2 = sigmoid(X.dot(np.transpose(Theta1)))
	a2 = np.concatenate((np.ones([a2.shape[0],1]),a2),1)
	d3 = sigmoid(a2.dot(np.transpose(Theta2))) - y
	m = d3.shape[0]
	delta1 = 0
	delta2 = 0
	for i in range(m):
		d3Cur = np.transpose(d3[i])
		d2 = np.multiply(np.dot(np.transpose(Theta2),d3Cur)[1:], sigPrime(np.dot(Theta1 ,np.transpose(X[i]))))
		delta1 += np.array(np.dot(np.transpose(np.matrix(d2)),np.matrix(X[i])))
		delta2 += np.array(np.dot(np.transpose(np.matrix(d3Cur)),np.matrix(np.transpose(a2[i]))))
	Theta1_Grad = (1/m) * delta1 
	Theta2_Grad = (1/m) * delta2
	return (Theta1_Grad,Theta2_Grad)