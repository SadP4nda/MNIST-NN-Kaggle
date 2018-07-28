import numpy as np
from ForwardProp import ForwardProp, CostFunc
from scipy.special import expit
import time
def sigPrime(z):
	return expit(z) * (1-expit(z))
def BackProp(ThetaVect,X,y,input_layer_size,hidden_layer1_size,hidden_layer2_size,classification_nodes,L = 0):
	a = time.time()
	ThetaVect = ThetaVect.reshape(ThetaVect.size,1,order='F')
	Theta1 = ThetaVect[0:(1+input_layer_size)*hidden_layer1_size].reshape((hidden_layer1_size,input_layer_size+1),order='F')
	Theta2 = ThetaVect[(1+input_layer_size)*hidden_layer1_size:(1+input_layer_size)*hidden_layer1_size+(1+hidden_layer1_size)*hidden_layer2_size].reshape(\
		(hidden_layer2_size,hidden_layer1_size+1),order='F')
	Theta3 = ThetaVect[(1+input_layer_size)*hidden_layer1_size+(1+hidden_layer1_size)*hidden_layer2_size:].reshape((classification_nodes,hidden_layer2_size+1),order='F')
	Therta1 = np.copy(Theta1)
	Therta1[:,0] = np.zeros(Therta1.shape[0]).T
	Therta2 = np.copy(Theta2)
	Therta2[:,0] = np.zeros(Theta2.shape[0]).T
	Therta3 = np.copy(Theta3)
	Therta3[:,0] = np.zeros(Theta3.shape[0]).T
	X = np.concatenate((np.ones([X.shape[0],1]),X),1)
	a2 = expit(X.dot(np.transpose(Theta1)))
	a2 = np.concatenate((np.ones([a2.shape[0],1]),a2),1)
	a3 = expit(a2.dot(np.transpose(Theta2)))
	a3 = np.concatenate((np.ones([a3.shape[0],1]),a3),1)
	hyp = expit(a3.dot(np.transpose(Theta3)))
	d4 =  hyp - y
	m = d4.shape[0]
	delta1 = 0
	delta2 = 0
	delta3 = 0
	for i in range(m):
		d4Cur = d4[i].T
		d3 = np.multiply(np.dot(Theta3.T,d4Cur)[1:], sigPrime(np.dot(Theta2 ,a2[i].T)))
		d2 = np.multiply(np.dot(Theta2.T,d3)[1:],sigPrime(np.dot(Theta1,X[i].T)))
		delta1 += np.dot(d2[np.newaxis].T,X[i][np.newaxis])
		delta2 += np.dot(d3[np.newaxis].T,a2[i][np.newaxis])
		delta3 += np.dot(d4Cur[np.newaxis].T,a3[i][np.newaxis])
	Theta1_Grad = (1/m) * delta1 + ((L/m) * Therta1)
	Theta2_Grad = (1/m) * delta2 + ((L/m) * Therta2)
	Theta3_Grad = (1/m) * delta3 + ((L/m) * Therta3)
	GradVect = np.append(Theta1_Grad.reshape(Theta1_Grad.size,order='F'),Theta2_Grad.reshape(Theta2_Grad.size,order='F'))
	GradVect = np.append(GradVect,Theta3_Grad.reshape(Theta3_Grad.size,order='F'))
	b = time.time()
	print("Time for BackProp %.12f" % (b - a))
	return GradVect