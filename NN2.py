from __future__ import print_function

import sys
import numpy as np 
import tensorflow as tf
import random
from random import randint, shuffle

from NN_data import make_Dataset, min_max_denorm, save_model, restore_model



# Get Dataset
train_X, train_Y = make_Dataset(sys.argv[1])
temp = list(zip(train_X, train_Y))
seed = 37 #sys.argv[3]
random.seed(seed)
shuffle(temp)
train_X, train_Y = zip(*temp)
train_X = np.array(train_X)
train_Y = np.array(train_Y)
len_dataset = train_X.shape[0]
test_ratio = 0.20
test_size = int(train_X.shape[0]*test_ratio)
#test_size = 0

# k-fold
k = int(sys.argv[2])
maxK = 1/test_ratio
if(k > maxK-1):
	k = maxK-1

# Data Splitting
if test_size != 0:
	"""test_size = int(test_size/2)
	validation_x, validation_y = train_X[-test_size:], train_Y[-test_size:]
	train_X, train_Y = train_X[:-test_size], train_Y[:-test_size]"""
	
	"""test_x, test_y = train_X[-test_size:], train_Y[-test_size:]
	train_X, train_Y = train_X[:-test_size], train_Y[:-test_size]"""
	
	"""test_x, test_y = train_X[:test_size], train_Y[:test_size]
	train_X, train_Y = train_X[test_size:], train_Y[test_size:]"""
	
	b_split = int(len_dataset * test_ratio * k)
	e_split = int((b_split + len_dataset * test_ratio) % len_dataset)
	test_x, test_y = train_X[b_split:e_split], train_Y[b_split:e_split]
	train_X = np.concatenate([train_X[0:b_split], train_X[e_split:]])
	train_Y = np.concatenate([train_Y[0:b_split], train_Y[e_split:]])
	
	validation_x, validation_y = test_x, test_y

# Training Parameters
learning_rate = 0.00025 #0.0002
training_steps = 5000
batch_size = train_X.shape[0]
#batch_size = 32 ; training_steps *= int(train_X.shape[0]/batch_size)
display_step = 200
save_step = 2000

# Network Parameters
num_input = 8
num_classes = 10 # num output 
num_last_hidden = 256 # hidden layer 
activation = tf.nn.leaky_relu
#activation = tf.nn.relu
#activation = tf.nn.elu

# Learning rate decay
use_learning_rate_decay = True
learning_rate_init = learning_rate
learning_rate_decay_steps = 1
learning_rate_decay = 0.99999
learning_rate_min = 0.0001

# Normalization
l2_beta = 0.005 #0.01 #0.009 #0.0009925
drop_out = 0.75
use_batch_norm = False
use_gradient_clipping = True
clip_by_norm = True # or by value
gradient_clipping_norm = 5.0



print("batch_size= "+str(batch_size))
print("shape= "+str(train_X.shape))



# =======================
# ======== MODEL ========
# =======================

# tf Graph input
X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None, num_classes])

is_train = tf.placeholder(tf.bool, name="is_train");
global_step = tf.placeholder(tf.int32, name="global_step")

# Define weights
weights = {
		
	'hidden': tf.get_variable('w_last_hidden', shape=[num_input, num_last_hidden],
			initializer=tf.contrib.layers.xavier_initializer()) ,						
	'out': tf.get_variable('w_out', shape=[num_last_hidden, num_classes],
			initializer=tf.contrib.layers.xavier_initializer())			
}
biases = {	
	'hidden': tf.get_variable('b_last_hidden', shape=[num_last_hidden],
			initializer=tf.constant_initializer(0.0)) ,
	'out': tf.get_variable('b_out', shape=[num_classes],
			initializer=tf.constant_initializer(0.0)) 					
}

def NN(x, weights, biases):
	hidden_layer = x
	
	# Last Hidden Layer		
	hidden_layer = tf.matmul(hidden_layer, weights['hidden']) + biases['hidden']
	hidden_layer = activation(hidden_layer)
	if use_batch_norm:
		hidden_layer = tf.layers.batch_normalization(hidden_layer, training=is_train)
	hidden_layer = tf.layers.dropout(hidden_layer, rate=drop_out, training=is_train)		
	
	y = hidden_layer
	x = y

	# Output layer
	return tf.matmul(x, weights['out']) + biases['out'], y
	#return outputs, outputs

logits, outputs = NN(X, weights, biases)
shape = tf.shape(outputs)
prediction = tf.nn.softmax(logits)
#prediction = logits
#prediction = activation(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y))
#loss_op = tf.reduce_mean(tf.square(tf.subtract(Y, prediction)))
if l2_beta != 0.0:
	"""l2 = l2_beta * sum(
		tf.nn.l2_loss(tf_var)
		    for tf_var in tf.trainable_variables()
		    if not ("noreg" in tf_var.name or "Bias" in tf_var.name)
	)"""
	l2 = l2_beta * tf.add_n([ tf.nn.l2_loss(v) for v in tf.trainable_variables()
                    if not ("noreg" in v.name or "Bias" in v.name) ])
	loss_op = tf.reduce_mean(loss_op + l2)
	#loss_op += l2

# Learning rate decay
if use_learning_rate_decay:
	learning_rate = tf.maximum(learning_rate_min, 
						tf.train.exponential_decay(learning_rate_init,
							global_step,
							learning_rate_decay_steps,
							learning_rate_decay))

# Optimizer
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
#optimizer = tf.train.RMSPropOptimizer(learning_rate) #, momentum=0.001)

# Gradient clipping
if use_gradient_clipping:
	gvs = optimizer.compute_gradients(loss_op)
	if clip_by_norm:
		grad, vs = zip(*gvs)
		grad, _ = tf.clip_by_global_norm(grad, gradient_clipping_norm)
		capped_gvs = zip(grad, vs)
	else:
		capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
	train_op = optimizer.apply_gradients(capped_gvs)
else:	
	train_op = optimizer.minimize(loss_op)

# Evaluate model
error = loss_op
#accuracy = tf.divide(1 ,tf.add(error, 1))
#accuracy = tf.subtract(1. , error)
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
#shape = tf.shape(outputs)

# =======================

def getCorrectClass(out, y):
	correct = np.zeros([10])
	correctConf = np.zeros([10, 10])
	for i in range(0, len(out)):
		maxOut = np.argmax(out[i])
		maxY = np.argmax(y[i])
		if(maxOut == maxY):
			correct[maxOut] += 1
			correctConf[maxOut, maxOut] += 1
		else:
			correctConf[maxY, maxOut] += 1
	return correct, correctConf

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

	saver = tf.train.Saver() 
	#restore_model(saver, sess)

	# Run the initializer
	sess.run(init)

	maxvAcc = 0
	maxAcc = 0
	maxStep = 0

	for step in range(1, training_steps+1):
		# Get batch
		if(batch_size == train_X.shape[0]):
			batch_x, batch_y = train_X, train_Y
		else:
			batch_x = np.zeros([batch_size, num_input])			
			batch_y = np.zeros([batch_size, num_classes])
			for i in range(0, batch_size):
				ind = randint(0, train_X.shape[0]-1)
				batch_x[i] = train_X[ind]
				batch_y[i] = train_Y[ind]
		
		# Run optimization op (backprop)
		sess.run(train_op, feed_dict={X: batch_x, Y: batch_y, is_train: True, global_step: step})
		if step % display_step == 0 or step == 1:
			# Calculate batch loss and accuracy
			loss, acc, out = sess.run([loss_op, accuracy, shape], 
								feed_dict={X: batch_x, Y: batch_y, is_train: False})
			#print(out)
			print("Step " + str(step) + ", Minibatch Loss= " + \
				  "{:.4f}".format(loss) + ", Training Accuracy= " + \
				  "{:.4f}".format(acc), end = '', flush=True)
			if test_size != 0:
				vAcc, out = sess.run([accuracy, prediction], 
									feed_dict={X: validation_x, Y: validation_y, is_train: False})
				#print(out)
				print(", Validation Accuracy= " + "{:.4f}".format(vAcc))
				if vAcc > maxvAcc:
					maxvAcc = vAcc
					maxAcc = acc
					maxStep = step
					#save_model(saver, sess)
					c1, c2 = getCorrectClass(out, validation_y)
					print(c1)
					print(c2)
			else:
				print()

	print("Optimization Finished!")
	print("___MAX - Validation: " + "{:.4f}".format(maxvAcc) + ", Training: " + "{:.4f}".format(maxAcc) + ", step = {}, seed = {}".format(maxStep, seed))
	fichier = open("result.txt", 'a')
	fichier.write("\n{} _ {:.4f}".format(seed, maxvAcc))
	fichier.close()

	# Calculate accuracy 	
	if test_size != 0:
		acc, out, err = sess.run([accuracy, prediction, error], 
							feed_dict={X: test_x, Y: test_y, is_train: False})
		np.set_printoptions(precision=4, threshold=sys.maxsize)		
		print("Testing Accuracy:", acc)
		print("Error:", err)
		"""input()		
		print(out)
		print(test_y)"""
