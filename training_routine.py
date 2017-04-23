import random
import scipy
import numpy as np
import vgg_16_keras
from keras.layers.core import Dense, Flatten, Dropout, Merge
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
import pandas as pd



def get_data(filter=False):
	"""Load the data from passing_rate.csv and image directory

	:param filter: boolean -- Flag to filter data to include only Linac 1, 3, 4 and Energy 6X

	:returns: Numpy arrays containing the ids, passing rates, and fluence map images
	"""
	if filter:
		df = pd.read_csv('Image_description.csv')
		mask15 = df["'15 MV flag'"] == 1
		mask10 = df["'10 MV flag'"] == 1
		maskL1 = df["'Linac 1'"] == 1
		maskL3 = df["'Linac 3'"] == 1
		maskL4 = df["'Linac 4'"] == 1
		maskL = maskL1 | maskL3 | maskL4 
		maskE = mask15 | mask10
		filter_idsL = df[maskL]["'PlanSetupSer'"].tolist()
		filter_idsE = df[maskE]["'PlanSetupSer'"].tolist()
		ids_y_raw = np.genfromtxt("passing_rate.csv", delimiter=',')
		nobs = sum([1 if ids_y_raw[i][0] in filter_idsL and ids_y_raw[i][0] not in filter_idsE else 0 for i in range(len(ids_y_raw))])
		ids_y = np.empty((nobs, 2))
		j = 0
		for i in range(len(ids_y_raw)):
		    if ids_y_raw[i][0] in filter_idsL and ids_y_raw[i][0] not in filter_idsE:
		        ids_y[j][0], ids_y[j][1] = ids_y_raw[i][0], ids_y_raw[i][1]
		        j += 1
	else:
		ids_y = np.genfromtxt("passing_rate.csv", delimiter=',')
	imgs = np.empty((len(ids_y), 3, 224, 224))
	path = '/data/vwrideout/Images/'
	for i in range(len(ids_y)):
	    im = np.genfromtxt(path + str(int(ids_y[i][0])), delimiter=',')
	    imre = scipy.misc.imresize(im, (224,224))
	    for j in range(3):
	        imgs[i][j] = imre
	y = np.array([100.0 - ids_y[i][1] for i in range(len(ids_y))])
	ids = np.array([ids_y[i][0] for i in range(len(ids_y))])
	return ids, y, imgs


def get_metadata(ids):
	"""Get metadata for an array of observations

	:param ids: numpy array of ids

	:returns: numpy array of metadata values in the same order as the inputs
	"""
	d = {}
	with open("Image_description.csv", 'rU') as f:
		for line in f.readlines()[1:]:
			a = line.split(",")
			d[int(a[0])] = np.array(a[1:11], dtype='float64')
	return np.array([d[id] for id in ids])	


def get_masks(seeds=[47, 2352], num=50):
	"""Get masks for splitting data into train/test

	:param seeds: list of ints
	:num: desired size of test set

	:returns: a list of masks, one for each random seed provided
	"""
	masks = []
	for s in seeds:
		random.seed(s)
		masks.append(random.sample(range(532), num))
	return masks



"""VGG models:
A series of functions that return different modifications to the VGG architecture for experimentation with transfer learning.
"""
def get_model_VGG_final_layer():
	model = vgg_16_keras.VGG_16('vgg16_weights.h5')
	model.pop()
	for layer in model.layers: layer.trainable = False
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model


def get_model_VGG_pop_dense(add_dropout=False, add_batchnorm=False):
	model = vgg_16_keras.VGG_16('vgg16_weights.h5')
	for i in range(5): model.pop()
	for layer in model.layers: layer.trainable = False
	model.add(Dense(2048, activation='relu'))
	if(add_batchnorm):
		model.add(BatchNormalization())
	if(add_dropout):
		model.add(Dropout(0.5))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model


def get_model_merged():
	m1 = get_model_VGG_pop_dense(False, False)
	m1.pop()
	m2 = Sequential()
	m2.add(Dense(32, input_shape=(10,), activation='relu'))
	#m2.add(BatchNormalization())
	#m2.add(Dropout(0.5))
	model = Sequential()
	model.add(Merge([m1, m2], mode='concat'))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model


def get_model_big_dense(d=0.5):
	model = vgg_16_keras.VGG_16('vgg16_weights.h5')
	for i in range(5): model.pop()
	for layer in model.layers: layer.trainable = False
	model.add(Dense(4096, activation='relu'))
	model.add(BatchNormalization())
	model.add(Dropout(d))
	model.add(Dense(4096, activation='relu'))
	model.add(BatchNormalization())
	model.add(Dropout(d))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model

def get_model_VGG_pop_conv1(add_dropout=False):
	model = vgg_16_keras.VGG_16('vgg16_weights.h5')
	for i in range(13): model.pop()
	for layer in model.layers: layer.trainable = False
	model.add(Flatten())
	model.add(Dense(2048, activation='relu'))
	if(add_dropout):
		model.add(Dropout(0.5))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model
