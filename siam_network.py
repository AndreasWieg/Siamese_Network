import numpy as np
import tensorflow as tf
import os
from absl import app
from absl import flags
from util import load_data_art,shuffle_data,save_image
import time
import math
from random import shuffle
import random

class SIAM(object):

	def __init__(self,is_training,epoch,checkpoint_dir,learning_rate,z_dim,batch_size,beta1,beta2,image_size,margin):
		""""
		Args:
			beta1: beta1 for AdamOptimizer
			beta2: beta2 for AdamOptimizer
			learning_rate: learning_rate for the AdamOptimizer
			training: [bool] Training/NoTraining
			batch_size: size of the batch_
			epoch: number of epochs
			checkpoint_dir: directory in which the model will be saved
			name_art: name of the fake_art will be saved
			image_size: size of the image
		"""

		self.beta1 = beta1
		self.beta2 = beta2
		self.learning_rate = learning_rate
		self.training = is_training
		self.batch_size = batch_size
		self.epoch = epoch
		self.checkpoint_dir = checkpoint_dir
		self.name_art = "fake_art"
		self.margin = margin
		self.image_size = image_size
		self.lambda = 1e-7

		self.save_epoch = 0
		self.build_network()

		with tf.variable_scope("adam",reuse=tf.AUTO_REUSE) as scope:
			print("init_siamese_optimizer")
			self.siam_optim = tf.train.AdamOptimizer(self.learning_rate, beta1 = self.beta1,beta2 = self.beta2).minimize(self.sim_loss,var_list = self.vars_Network)


		self.init  = tf.global_variables_initializer()
		self.config = tf.ConfigProto()
		self.config.gpu_options.allow_growth = True
		self.sess = tf.Session(config = self.config)


	def network(self,x,reuse=False):
		with tf.variable_scope("siamese" ,reuse=reuse):

			x = tf.layers.conv2d(x,filters=128,kernel_size=5,kernel_initializer=tf.glorot_normal_initializer(),strides=(2,2),padding='same',activation = None,name="conv_1")
			x = tf.layers.batch_normalization(x,momentum=0.9,gamma_initializer = tf.random_normal_initializer(1., 0.02))
			x = tf.nn.leaky_relu(x)

			x = tf.layers.conv2d(x,filters=256,kernel_size=5,activation = None,kernel_initializer=tf.glorot_normal_initializer(),strides=2,padding='same',name="conv_3")
			x = tf.layers.batch_normalization(x,momentum=0.9,gamma_initializer = tf.random_normal_initializer(1., 0.02))
			x = tf.nn.leaky_relu(x)

			x = tf.layers.conv2d(x,filters=512,kernel_size=5,activation = None,kernel_initializer=tf.glorot_normal_initializer(),strides=2,padding='same',name="conv_4")
			x = tf.layers.batch_normalization(x,momentum=0.9,gamma_initializer = tf.random_normal_initializer(1., 0.02))
			x = tf.nn.leaky_relu(x)

			x = tf.layers.conv2d(x,filters=1024,kernel_size=5,activation = None,kernel_initializer=tf.glorot_normal_initializer(),strides=2,padding='same',name="conv_5")
			x = tf.layers.batch_normalization(x,momentum=0.9,gamma_initializer = tf.random_normal_initializer(1., 0.02))
			x = tf.nn.leaky_relu(x)

			x = tf.layers.flatten(x)
			x = tf.layers.dense(x,1,activation=None,kernel_initializer=tf.contrib.layers.xavier_initializer(),name="disc_output")

		return x

	def build_network(self):

		self.input_1 = tf.placeholder(tf.float32, [None,self.image_size,self.image_size,3], name="picture1")
		self.input_2 = tf.placeholder(tf.float32,[None,self.image_size,self.image_size,3], name ="picture2")


		#label = 1 if sim label = 0 if not sim
		self.label = tf.placeholder(tf.float32,[None,1],name="label data")

		self.output_1 = self.discriminator(self.input_1,reuse = False)
		self.output_2 = self.discriminator(self.input_2,reuse = True)

		self.euk_dis = tf.pow(tf.subtract(self.output_1, self.output_2), 2, name = 'euclidian distance')
        self.euk_dis = tf.reduce_sum(self.euk_dis, 1)
		self.euk_dis = tf.sqrt(self.euk_dis + self.lambda)

		self.L_sim = self.label * tf.square(self.euk_dis)
		self.L_unsim = ( 1 - self.label) * tf.square(tf.maximum((self.margin - self.euk_dis), 0))

		self.sim_loss = tf.reduce_mean(self.L_sim + self.L_unsim) / 2

		#Tensorboard variables
		self.sim_loss_his = tf.summary.histogram("d_real", self.sim_loss)

		'''
		self.G_sum = tf.summary.histogram("G",self.Gen)
		self.z_sum = tf.summary.histogram("z_input",self.z)

		tf.summary.scalar('self.g_loss', self.g_loss )
		tf.summary.scalar('self.d_loss', self.d_loss )
		'''

		#collect generator and encoder variables
		self.vars_Network = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='siamese')

		self.saver = tf.train.Saver()

		#Tensorboard variables
		'''
		self.summary_g_loss = tf.summary.scalar("g_loss",self.g_loss)
		self.summary_d_loss = tf.summary.scalar("d_loss",self.d_loss)
		'''

	def save_model(self, iter_time):
		model_name = 'model'
		self.saver.save(self.sess, os.path.join(self.checkpoint_dir, model_name), global_step=iter_time)
		print('=====================================')
		print('             Model saved!            ')
		print('=====================================\n')

	def train(self):
		with self.sess:
			if self.load_model():
				print(' [*] Load SUCCESS!\n')
			else:
				print(' [!] Load Failed...\n')
			#imported_meta = tf.train.import_meta_graph("C:/Users/Andreas/Desktop/punktwolkenplot/pointgan/checkpoint/model.ckpt-4.meta")
			#imported_meta.restore(sess, "C:/Users/Andreas/Desktop/punktwolkenplot/pointgan/checkpoint/model.ckpt-4")
			train_writer = tf.summary.FileWriter("./logs",self.sess.graph)
			merged = tf.summary.merge_all()
			#test_writer = tf.summary.FileWriter("C:/Users/Andreas/Desktop/punktwolkenplot/pointgan/")


			self.sess.run(self.init)


			self.training_data = load_data_art()
			print(self.training_data.shape)
			k = (len(self.training_data) // self.batch_size)

			loss_g_val,loss_d_val = 0, 0
			self.training_data = self.training_data[0:(self.batch_size*k)]

			self.start_time = time.time()
			print("Start Time:" self.start_time)
			self.counter = 0

			for e in range(0,self.epoch):
				epoch_loss_siam = 0.
				self.training_data = shuffle_data(self.training_data)

				for i in range(0,k):

					self.batch = self.training_data[i*self.batch_size:(i+1)*self.batch_size]
					self.image_1 = self.batch[0]
					self.image_2 = self.batch[1]
					self.label_training = self.batch[2]

					_, loss_d_val, loss_siam= self.sess.run([self.siam_optim,self.sim_loss,self.sim_loss_his],feed_dict={self.input_1: self.image_1, self.input_2: self.image_2, self.label: self.label_training})
					train_writer.add_summary(loss_siam,self.counter)

					self.counter += 1

					epoch_loss_siam += loss_siam

				epoch_loss_d /= k
				epoch_loss_g /= k
				print("Loss of D: %f" % epoch_loss_d)
				print("Loss of G: %f" % epoch_loss_g)
				print("Epoch%d" %(e))
				if e % 1 == 0:
					save_path = self.saver.save(self.sess,"C:/Users/Andreas/Desktop/siamese-network/checkpoint/model.ckpt",global_step=self.save_epoch)
					print("model saved: %s" %save_path)


			print("training finished")

	def load_model(self):
		print(' [*] Reading checkpoint...')
		ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
		if ckpt and ckpt.model_checkpoint_path:
			ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
			self.saver.restore(self.sess, os.path.join(self.checkpoint_dir, ckpt_name))
			meta_graph_path = ckpt.model_checkpoint_path + '.meta'
			self.save_epoch = int(meta_graph_path.split('-')[-1].split('.')[0])
			print('===========================')
			print('   iter_time: {}'.format(self.save_epoch))
			print('===========================')
			return True
		else:
			return False
