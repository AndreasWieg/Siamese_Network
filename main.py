import numpy as np
import tensorflow as tf
import os
from absl import app, flags
from siam_network import SIAM
import sys

flags = tf.app.flags

flags.DEFINE_integer("epochs",1000,"epochs per trainingstep")
flags.DEFINE_float("learning_rate",0.0001,"learning rate for the model")
flags.DEFINE_integer("image_size",128,"Image size of the input")
flags.DEFINE_bool("training",True,"running training of the poincloud gan")
flags.DEFINE_string("checkpoint_dir","C:/Users/Andreas/Desktop/siamese-network/checkpoint","where to save the model")
flags.DEFINE_string("sample_dir","samples","where the samples are stored")
flags.DEFINE_integer("iterations",100000,"number of patches")
flags.DEFINE_integer("batch_size",64,"size of the batch")
flags.DEFINE_float("beta1",0.5,"adam beta1")
flags.DEFINE_float("beta2",0.9,"adam beta2")
flags.DEFINE_integer("z_dim",128,"sample size from the normal distribution for the generator")
flags.DEFINE_float("margin",0.5,"margin of the siamese loss"
FLAGS = flags.FLAGS

def _main(argv):
	print("initializing Params")
	if not os.path.exists(FLAGS.checkpoint_dir):
		os.makedirs(FLAGS.checkpoint_dir)
	if not os.path.exists(FLAGS.sample_dir):
		os.makedirs(FLAGS.sample_dir)
	if FLAGS.training == True:
			cgan = SIAM(FLAGS.training,FLAGS.epochs,FLAGS.checkpoint_dir,FLAGS.learning_rate,FLAGS.z_dim,FLAGS.batch_size,FLAGS.beta1,FLAGS.beta2,FLAGS.image_size,FLAGS.margin)
			cgan.train()
	else:
		if not cgan.load(FLAGS.checkpoint_dir):
			print("first train your model")

if __name__ == '__main__':
	print('Starting the Programm....')
	app.run(_main)
