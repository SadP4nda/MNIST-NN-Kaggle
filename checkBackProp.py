from BackProp import BackProp
from NumericalGrad import NumericalGradient
import numpy as np
def DebugWeightGen(Sj_1,Sj):
	s = (Sj_1,Sj+1)
	Weights = np.zeros(s)
	Weights = np.reshape(np.sin(range(Weights.size)),[Sj_1,Sj+1])
	return Weights
def logicalVectorize(y,classification_nodes):
	newY = np.zeros([y.shape[0],classification_nodes])
	for i in range(y.shape[0]):
		newY[i][y[i]] = 1
	return newY
def checkBackwardsProp():
	input_layer = 3
	hidden_layer1_size = 5
	hidden_layer2_size = 5
	classification_nodes = 3
	m = 5
	Theta1 = DebugWeightGen(hidden_layer1_size,input_layer)
	Theta2 = DebugWeightGen(hidden_layer2_size,hidden_layer1_size)
	Theta3 = DebugWeightGen(classification_nodes,hidden_layer2_size)
	thetaVect = np.append(Theta1.reshape(Theta1.size,order='F'),Theta2.reshape(Theta2.size,order='F'))
	thetaVect = np.append(thetaVect,Theta3.reshape(Theta3.size,order='F'))
	X = DebugWeightGen(m,input_layer-1)
	y = np.remainder(np.arange(5),classification_nodes)-1
	y = logicalVectorize(y.T,classification_nodes)
	grad = BackProp(thetaVect,X, y,input_layer,hidden_layer1_size,hidden_layer2_size,classification_nodes,1)
	numgrad = NumericalGradient(X, y, Theta1, Theta2,Theta3,1)
	print(numgrad)
	print(grad)
	print(np.linalg.norm(numgrad-grad)/np.linalg.norm(numgrad+grad))
if __name__ == '__main__':
	checkBackwardsProp()