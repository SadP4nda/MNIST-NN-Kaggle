import numpy as np
from scipy.special import expit
from math import log
def ForwardProp(ThetaVect,X,input_layer_size,hidden_layer1_size,hidden_layer2_size,classification_nodes):
	ThetaVect = ThetaVect.reshape(ThetaVect.size,1,order='F')
	Theta1 = ThetaVect[0:(1+input_layer_size)*hidden_layer1_size].reshape((hidden_layer1_size,input_layer_size+1),order='F')
	Theta2 = ThetaVect[(1+input_layer_size)*hidden_layer1_size:(1+input_layer_size)*hidden_layer1_size+(1+hidden_layer1_size)*hidden_layer2_size].reshape(\
		(hidden_layer2_size,hidden_layer1_size+1),order='F')
	Theta3 = ThetaVect[(1+input_layer_size)*hidden_layer1_size+(1+hidden_layer1_size)*hidden_layer2_size:].reshape((classification_nodes,hidden_layer2_size+1),order='F')
	X = np.concatenate((np.ones([X.shape[0],1]),X),1)
	a2 = expit(X.dot(np.transpose(Theta1)))
	a2 = np.concatenate((np.ones([a2.shape[0],1]),a2),1)
	a3 = expit(a2.dot(np.transpose(Theta2)))
	a3 = np.concatenate((np.ones([a3.shape[0],1]),a3),1) 
	return expit(a3.dot(np.transpose(Theta3)))
def CostFunc(ThetaVect,X,y,input_layer_size,hidden_layer1_size,hidden_layer2_size,classification_nodes,L = 0):
	m = X.shape[0]
	Theta1 = np.copy(ThetaVect[0:(input_layer_size+1)*hidden_layer1_size].reshape((hidden_layer1_size,input_layer_size+1),order='F'))
	Theta2 = ThetaVect[(1+input_layer_size)*hidden_layer1_size:(1+input_layer_size)*hidden_layer1_size+(1+hidden_layer1_size)*hidden_layer2_size].reshape(\
		(hidden_layer2_size,hidden_layer1_size+1),order='F')
	Theta3 = ThetaVect[(1+input_layer_size)*hidden_layer1_size+(1+hidden_layer1_size)*hidden_layer2_size:].reshape((classification_nodes,hidden_layer2_size+1),order='F')
	Therta1 = np.copy(Theta1)
	Therta1[:,0] = np.zeros(Therta1.shape[0]).T
	Therta2 = np.copy(Theta2)
	Therta2[:,0] = np.zeros(Theta2.shape[0]).T
	Therta3 = np.copy(Theta3)
	Therta3[:,0] = np.zeros(Theta3.shape[0]).T
	hyp = ForwardProp(ThetaVect,X, input_layer_size, hidden_layer1_size,hidden_layer2_size, classification_nodes)
	return (-1 / m) * np.sum(np.sum(np.multiply(y ,np.log(hyp)) + np.multiply((1 - y) , np.log(1 - hyp))))\
			+ (L / (2 * m)) * (np.sum(np.sum(np.power(Therta1,2))) + np.sum(np.sum(np.power(Therta2,2)))\
			+ np.sum(np.sum(np.power(Therta3,2))))