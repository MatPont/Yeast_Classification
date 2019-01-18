import sys
import numpy as np 
import tensorflow as tf

checkpointDirectory = "./modelSaved/"
modelCheckpoint = checkpointDirectory+"modelWeights.ckpt"

def save_model(saver, session):
	# Save model
	saver.save(session, modelCheckpoint)
	print("=== Model saved as \"{}\" ===".format(modelCheckpoint))

def restore_model(saver, session):
	if(tf.train.checkpoint_exists(checkpointDirectory+"checkpoint")):
		# Restore model
		saver.restore(session, modelCheckpoint)
		print("\n=== Model restored from \"{}\" ===".format(modelCheckpoint))
	else:
		print("\n=== Model NOT restored ===".format(modelCheckpoint))

def min_max_norm(x, mini, maxi):
	return (x - mini) / (maxi - mini)
	
def min_max_denorm(x_norm, mini, maxi):
	return x_norm * (maxi - mini) + mini
	
def standard_score_norm(x):
	for j in range(0, x.shape[1]):
		mean = np.mean(x[:,j])
		std = np.std(x[:,j])
		for i in range(0, x.shape[0]):
			if std != 0:
				x[i][j] = (x[i][j] - mean) / std

def remove_outliers(x, y):
	out = set()
	for j in range(0, x.shape[1]):
		q1 = np.percentile(x[:,j], 25)
		q3 = np.percentile(x[:,j], 75)		
		eiq = q3 - q1
		mini = q1 - 1.5 * eiq
		maxi = q3 + 1.5 * eiq
		for i in range(0, x.shape[0]):	
			if(x[i][j] > maxi or x[i][j] < mini):
				out.add(i)
				#x[i][j] = np.percentile(x[:,j], 50)
	
	"""t = np.array(list(out))
	x = np.delete(x, t, axis=0)
	y = np.delete(y, t, axis=0)"""
	return x, y

def class_to_number(className):
	Y = np.zeros([10])
	if(className == "CYT"):
		ind = 0
	elif(className == "NUC"):
		ind = 1
	elif(className == "MIT"):
		ind = 2
	elif(className == "ME3"):
		ind = 3
	elif(className == "ME2"):
		ind = 4
	elif(className == "ME1"):
		ind = 5
	elif(className == "EXC"):
		ind = 6
	elif(className == "VAC"):
		ind = 7
	elif(className == "POX"):
		ind = 8
	elif(className == "ERL"):
		ind = 9
	Y[ind] = 1
	return Y

def make_Dataset(fileName):
	num_field = 10
	fichier = open(fileName, 'r')
	contenu = fichier.read()
	fichier.close()	
	lines = contenu.split()
	nbr_lines = (int)(len(lines)/num_field)
	X = np.zeros([nbr_lines, num_field-2])
	Y = np.zeros([nbr_lines, 10])	
	for i in range(0, len(X)):
		for j in range(1, num_field):
			dataTemp = lines[i*num_field + j]
			if(j < num_field-1):
				X[i][j-1] = dataTemp
			else:
				Y[i] = class_to_number(dataTemp)
	X, Y = remove_outliers(X, Y)
	standard_score_norm(X)
	return X, Y
