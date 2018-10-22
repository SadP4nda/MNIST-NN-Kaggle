'''
	A Tensorflow Keras module implementation of the Neural Network Kernal that I defined in Kernel.py
'''
#base python library
import time
#locally installed python libraries
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as npl
import numpy.random as npr
import pandas
import scipy as sp
from scipy.optimize import fmin_cg, fmin_l_bfgs_b,fmin_bfgs
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow import keras

def logicalVectorize(y):
	'''
		Creates a one hot matrix for each value in y
	'''
	newY = np.zeros([y.shape[0],10])
	for i in range(y.shape[0]):
		newY[i][y[i]] = 1
	return newY
hidden_layer_nodes = 700
hidden_layer_2_nodes = 700
classification_nodes = 10
if __name__ == '__main__':
	Location = 'train.csv'
	df = pandas.read_csv(Location)

	y = np.asarray(df['label'].values)
	X = df.values
	X = np.delete(X, 0 ,axis = 1)
	print(X)
	print(X.shape)
	X = ((X) * ((1 + 1)/255)) - 1
	y = logicalVectorize(y)
	X,X_test,y,y_test = train_test_split(X,y,test_size=0.2)

	X,y = shuffle(X,y)
	#X_test = np.copy(X)
	#y_test = np.copy(y)
	reg_param = 4
	reg_param /= X.shape[0]	
	print(reg_param)
	print(X)
	print(X_test.shape)
	model = keras.Sequential()
	model.add(keras.layers.Dense(hidden_layer_nodes,kernel_regularizer=keras.regularizers.l2(reg_param),activation="relu",use_bias=True,input_shape=(X.shape[1],)))
	model.add(keras.layers.Dropout(.5))
	model.add(keras.layers.Dense(hidden_layer_2_nodes//1.5,kernel_regularizer=keras.regularizers.l2(reg_param),use_bias=True,activation="relu"))
	model.add(keras.layers.Dropout(.5))
	model.add(keras.layers.Dense(hidden_layer_2_nodes//2,kernel_regularizer=keras.regularizers.l2(reg_param),use_bias=True,activation="relu"))
	model.add(keras.layers.Dropout(.5))
	model.add(keras.layers.Dense(hidden_layer_2_nodes//2,kernel_regularizer=keras.regularizers.l2(reg_param),use_bias=True,activation="relu"))
	model.add(keras.layers.Dropout(.5))
	model.add(keras.layers.Dense(hidden_layer_2_nodes//4,kernel_regularizer=keras.regularizers.l2(reg_param),use_bias=True,activation="relu"))
	model.add(keras.layers.Dense(10,kernel_regularizer=keras.regularizers.l2(reg_param),use_bias=True,activation="softmax"))
	model.compile(optimizer=tf.train.AdamOptimizer(.0001),loss="categorical_crossentropy",metrics=["accuracy"])
	model.fit(X,y,epochs=2000,batch_size=X.shape[0],validation_data = (X_test,y_test))
	model.evaluate(X,y,batch_size=X.shape[0])
	model.evaluate(X_test,y_test,batch_size=X_test.shape[0])
	predTest = model.predict(X_test,batch_size=X_test.shape[0])
	predTest_numbers = np.argmax(predTest,axis=1)
	yTestLabels = np.argmax(y_test,axis=1)
	X2 = np.asarray(pandas.read_csv('test.csv').values)
	X2 = ((X2) * ((1 + 1)/255)) - 1
	print("Accuracy of prediction on test set %.12f" % (np.sum(predTest_numbers == yTestLabels)/yTestLabels.shape[0]))
	pred_numbers = np.argmax(model.predict(X2),1)[np.newaxis].T
	print(pred_numbers)
	stuff = []
	for i in range(pred_numbers.shape[0]):
		stuff.append(i+1)
	stuff = np.array(stuff)[np.newaxis].T
	print(stuff.shape)
	print(pred_numbers.shape)
	yeet = np.hstack((stuff,pred_numbers))
	print(yeet)
	Ans = pandas.DataFrame(yeet,columns=['ImageId','Label'])
	Ans.to_csv("D:\\Coding\\Kaggle\\MNIST\\Submission.csv",index=False)
	X2ims = X2.reshape((X2.shape[0],28,28))
	for i in range(9):
		plt.subplot(330 + (i + 1))
		plt.imshow(X2ims[i])
	plt.show()
