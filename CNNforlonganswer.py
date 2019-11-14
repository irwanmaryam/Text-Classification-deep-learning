import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np
import os
import time
import datetime
import sys
from cnn import cnn
# from utils import *
from sklearn.model_selection import train_test_split
from tensorflow.contrib import learn
import pandas as pd
import helpers
# import seaborn as sns


data = pd.read_csv("./data/spam.csv", encoding = 'latin-1')


data = data.drop(labels = ["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis = 1)




x_input ,y_raw = helpers.load_data_and_labels(data)

max_length = max([len(x.split(" ")) for x in x_input])

vocabprocess = learn.preprocessing.VocabularyProcessor(max_length)

x = np.array(list(vocabprocess.fit_transform(x_input)))
y = np.array(y_raw)


## shuffle data 

np.random.seed(10)
shuffleindice = np.random.permutation(np.arange(len(y)))
xshuf = x[shuffleindice]
yshuf = y[shuffleindice]

# x_train, xtest, y_train, ytest = train_test_split(x_input, y, test_size = 0.33, random_state = 42)
# print(y_train)
splittest = -1 * int(0.10 * float(len(y)))
x_train, xtest = xshuf[:splittest], xshuf[splittest:]
y_train, ytest = yshuf[:splittest], yshuf[splittest:]



print("vocab size : {:d}".format(len(vocabprocess.vocabulary_)))
print("Train/Testing split: {:d}/{:d}".format(len(y_train), len(ytest)))
print(y_train)
print(ytest)


lengthsequence = x_train.shape[1]
sizevocab = len(vocabprocess.vocabulary_)
epoch = 50
# # learning_rate = 0.001
batch_size = 32
num_class = y_train.shape[1]

# print(vocabulary)
print(num_class)


logdir = "./logs/nn_logs"




with tf.Session() as sess:
	

	model = cnn(sizevocab, lengthsequence, num_class, batch_size)



	init_op = tf.compat.v1.global_variables_initializer()
	sess.run(init_op)
	saver = tf.compat.v1.train.Saver(tf.global_variables())


	# tf.summary.text("text", b)

	# tf.summary.histogram("weights", weight)
	tf.summary.histogram("fc1", model.fc)
	# tf.summary.histogram("fc2", model.fc3)
	# tf.summary.histogram("fc3", model.fc3)

	tf.summary.scalar("accuracy", model.accuracy)
	tf.summary.scalar("loss", model.loss)
	merge = tf.summary.merge_all()

	train_write = tf.summary.FileWriter(logdir + "/train", sess.graph)
	test_write = tf.summary.FileWriter(logdir + "/test", sess.graph)

	tf.global_variables_initializer().run()



	trainBatch = helpers.batch_iter(list(zip(x_train, y_train)), batch_size, epoch)


	for batch in trainBatch:


		Xbatch,Ybatch = zip(*batch)

		 


		train_dict = {model.x: Xbatch, model.y:Ybatch, model.keep_drop:0.5}

		_,summary, step, loss, accuracy = sess.run([model.gradient, merge,  model.global_step, model.loss, model.accuracy], feed_dict = train_dict)

		train_write.add_summary(summary, step)

		if step % 1 == 0:


			print("step {0}: loss = {1:.5f} accuracy = {2:.5f}".format(step, loss, accuracy))

		# current_step = tf.train.global_step(sess, model.global_step)

		if step % 100 == 0:
			
			feed_dict = {model.x: xtest, model.y: ytest, model.keep_drop:1.0}

		
			summary, step ,accuracy, loss, matrix= sess.run([merge, model.global_step, model.accuracy, model.loss, model.confusion], feed_dict = feed_dict)
			# train_accuracy = sess.run(model.accuracy, feed_dict = feed_dict)
				
			print("Evaluate : ")
			test_write.add_summary(summary, step)
			print("step {0}: loss = {1:.5f} accuracy = {2:.5f}".format(step, loss, accuracy))
			print(matrix)

