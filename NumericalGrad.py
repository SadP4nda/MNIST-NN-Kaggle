from ForwardProp import CostFunc
import numpy as np
import time
def NumericalGradient(X,y,Theta1,Theta2,Theta3,L = 0):
	theta1 = np.copy(np.transpose(Theta1.reshape(Theta1.size,order='F')))
	theta2 = np.copy(np.transpose(Theta2.reshape(Theta2.size,order='F')))
	theta3 = np.copy(Theta3.reshape(Theta3.size,order='F').T)
	thetaVect = np.append(theta1,theta2)
	thetaVect = np.append(thetaVect,theta3).T
	testVect = np.transpose(np.zeros(np.size(thetaVect)))
	numgrad = np.transpose(np.zeros(np.size(thetaVect)))
	eps = .0001
	for i in range(thetaVect.shape[0]):
		print("Iteration %d" % (i))
		testVect[i] = eps
		negativeThetaVec = thetaVect - testVect
		positiveThetaVec = thetaVect + testVect
		loss1 = CostFunc(negativeThetaVec,X, y,Theta1.shape[1]-1,Theta1.shape[0],Theta2.shape[0],Theta3.shape[0],L)
		loss2 = CostFunc(positiveThetaVec,X, y,Theta1.shape[1]-1,Theta1.shape[0],Theta2.shape[0],Theta3.shape[0],L)
		numgrad[i] = (loss2 - loss1) / (2*eps)
		testVect[i] = 0
	return numgrad
