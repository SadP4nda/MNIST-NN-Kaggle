import numpy as np
from BackProp import BackProp

def GradientDescent(ThetaVect,X,y,input_layer_size,hidden_layer1_size,classification_nodes,L = 0,epoch = 50,alpha = .1):
	for i in range(epoch):
		print("Iteration %d" % (i))
		ThetaVect = ThetaVect -  alpha * BackProp(ThetaVect, X, y, input_layer_size, hidden_layer1_size, classification_nodes, L)
		print(ThetaVect)
	return ThetaVect