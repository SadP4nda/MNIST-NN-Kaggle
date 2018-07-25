from ForwardProp import CostFunc
import numpy as np

def NumericalGradient(X,y,Theta1,Theta2):
	theta1 = np.transpose(Theta1.reshape(Theta1.size,order='F'))
	theta2 = np.transpose(Theta2.reshape(Theta2.size,order='F'))
	thetaVect = np.transpose(np.append(theta1,theta2))
	testVect = np.transpose(np.zeros(np.size(thetaVect)))
	numgrad = testVect = np.transpose(np.zeros(np.size(thetaVect)))
	eps = .0001
	for i in range(thetaVect.shape[0]):
		print("Iteration %d" % (i))
		testVect[i] = eps
		negative = thetaVect - testVect
		positive = thetaVect - testVect
		Therta1neg = negative[0:np.size(Theta1)].reshape(Theta1.shape[0],Theta1.shape[1])
		Therta2neg = negative[np.size(Theta1):].reshape(Theta2.shape[0],Theta2.shape[1])
		Therta1pos = positive[0:np.size(Theta1)].reshape(Theta1.shape[0],Theta1.shape[1])
		Therta2pos = positive[np.size(Theta1):].reshape(Theta2.shape[0],Theta2.shape[1])
		loss1 = CostFunc(X, Therta1neg, Therta2neg, y)
		loss2 = CostFunc(X, Therta1pos, Therta2pos, y)
		numgrad[i] = (loss2 - loss1)/ (2*eps)
		testVect[i] = 0
	return numgrad
