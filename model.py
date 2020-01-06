import tensorflow as tf
import os
import time
import sys
import numpy as np
from tensorflow.python.platform import gfile



class cnn(object):

	def __init__(self, sizevocab, lengthsequence, num_class, batch_size):


		self.x = tf.placeholder(tf.int32, [None, lengthsequence], name = "x")
		
		self.y = tf.placeholder(tf.float32, [None, num_class], name = "y")
		# self.keep_drop = tf.placeholder(tf.float32)

		

		print("target")
		print(self.y)


		# ------for svm model------


		self.dimension = 50
		self.sizefilter = [5,6,7]
		self.numfilter = 16

		self.is_training = tf.placeholder(tf.bool,[], name= "is_training")
		self.global_step = tf.Variable(0, name = "global_step")
		learning_rate = 0.001
		# learning_rate_decay = tf.compat.v1.train.exponential_decay(learning_rate, batch_size , 100, 0.95, staircase=True)
		# learning_rate_decay = tf.compat.v1.train.exponential_decay(learning_rate, self.global_step, 100, 0.96, staircase=True)
		# self.keep_drop = tf.placeholder(tf.float32, name = "keep_drop")

		# l2_loss = tf.constant(0.0)
		regularizer = tf.constant(0.1)

		with tf.name_scope("embeddings"):

			# word_embedding = tf.get_variable("word_embedding", [vocabulary, dimension])

			# embedded_word_id = tf.nn.embedding_lookup(word_embedding, self.x)
			with tf.device('/cpu:0'):
				word_embedded = tf.Variable(tf.truncated_normal([sizevocab, self.dimension], -1, 1), name = "embedding")
				print("input")
				print(word_embedded)

				# with tf.Session() as sess:
				# 	embedding_value = sess.run(word_embedded)
				# 	with open('embedding.txt','w') as file:
				# 		for n in range(sizevocab):
				# 			embed = embedding_value[n,:]
				# 			word = word_to_idx[n]
				# 			file.write('%s %s\n' %(word, ' '.join(map(str,embed))))


				# with open('embedding.txt','w') as file:
				# 	for word in range(sizevocab):
				# 		file.write(word + '\n')


				self.embeddedlook = tf.nn.embedding_lookup(word_embedded, self.x)

				print("embed")
				print(self.embeddedlook.shape)
				self.embedinput = tf.expand_dims(self.embeddedlook, -1)

				print("embedded")
				print(self.embedinput)

		pool_output = []



		for i,filter_size in enumerate(self.sizefilter):





			self.convolution = tf.layers.conv2d( inputs = self.embedinput, filters = self.numfilter, kernel_size= [filter_size, self.dimension], padding = "VALID", strides = [1,1])

			# weight = tf.Variable(tf.truncated_normal([filter_size, self.dimension,1, self.numfilter]))

			# bias = tf.Variable(tf.contant(0.1, [self.numfilter]), name = "bias12")

			# self.convolution = tf.nn.conv2d(self.embedinput, weight, strides=[1,1,1,1], padding="VALID", name = "convolution1")
			convrelu = tf.nn.relu(self.convolution)

			# dropoutcn1 = tf.nn.dropout(self.convolution, rate = 0.5)

			bath1 = tf.layers.batch_normalization(self.convolution)

			pooling = tf.layers.max_pooling2d(inputs = bath1, pool_size = [lengthsequence - filter_size + 1, 1], strides = [2,2], padding = 'VALID')
	

			print("pooling")
			print(pooling)


			

			pool_output.append(pooling)

			print("shape pool")
			print(pool_output)

	
		tf.summary.histogram("pooling", pooling)

		self.totalfilter = self.numfilter * len(self.sizefilter)
		flat1 = tf.concat(pool_output, 3)


		flat = tf.reshape(flat1, [-1, self.totalfilter])
		
		print("totalfilter")
		print(self.totalfilter)

		print("flat output")
		print(flat1)

		print("flat_fc")
		print(flat)

		# h_drop = tf.nn.dropout(flat, self.keep_drop)

		self.fc = tf.layers.dense(flat, 64, name = "weight1")

		# h_drop1 = tf.nn.dropout(self.fc, self.keep_drop)

		self.fc2 = tf.layers.dense(self.fc, num_class, name="weight2")
		with tf.variable_scope("weight2", reuse = True):
			w = tf.get_variable('kernel')
	


		# model_output = tf.subtract(tf.matmul(x_data, A), b)

		####### SVM MODEL #######
		# x = tf.placeholder("float", shape=[None, lengthsequence])
		# self.x = tf.placeholder(tf.int32, [None, lengthsequence], name = "x")
		# self.xsvm = tf.placeholder("float", shape=[None, lengthsequence])
		# svmW = tf.Variable(tf.truncated_normal([lengthsequence,1]))
		# b = tf.Variable(tf.zeros([1]))
		# yraw = tf.matmul(self.xsvm,svmW) + b

		# regularization_loss = 0.5*tf.reduce_sum(tf.square(svmW)) 
		# hinge_loss = tf.reduce_sum(tf.maximum(tf.zeros([batch_size,1]), 1 - self.y*yraw))
		# self.svm_loss = regularization_loss + 1*hinge_loss
		# self.train_step = tf.train.GradientDescentOptimizer(0.001).minimize(self.svm_loss)



		# predicted_class = tf.sign(yraw);
		# correct_prediction = tf.equal(self.y,predicted_class)
		# self.accuracysvm = tf.reduce_mean(tf.cast(correct_prediction, "float"))

		# ----SVM Model-------

		with tf.name_scope('SVM_MODEL'):



			#pure

			self.regularization_loss = tf.reduce_mean(tf.square(w))
			self.loss = tf.reduce_mean(tf.losses.hinge_loss(logits = self.fc2, labels= self.y))

			loss = self.regularization_loss + 1 * self.loss
		
			
			#pure
			self.gradient = tf.compat.v1.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(loss, global_step=self.global_step)
	
			# #with dropout
	
			#pure
			cp = tf.equal(tf.argmax(self.fc2, 1), tf.argmax(self.y, 1))
			self.accuracy = tf.reduce_mean(tf.cast(cp, tf.float32))

			# #with dropout
			
			self.confusion = tf.confusion_matrix(tf.argmax(self.x, 1), tf.argmax(self.x, 1))

		


		
	# ------- COmplete CNN and SVM model -------




			
