import sys
sys.path.append('/usr/local/lib/python2.7/dist-packages')

import tensorflow as tf
from Net.tensor import net6, inputdata
import cv2
import numpy as np
import os


def netify_image(net, model_path, layer, image, name):
	sess = net.load(var_path = model_path)
	print image
	img = cv2.resize(cv2.imread(image), (250,250))
	print img
	im = inputdata.im2tensor(img, channels = 3)

	imshape = im.shape
	im = np.reshape(im, (-1, imshape[0], imshape[1], imshape[2]))

	with sess.as_default():
		result = sess.run(layer, feed_dict={net.x:im})
		np.savetxt(name, result)
	sess.close()

def generate_rollout_values(net, model_path, layer, rollout_paths, destination):
	for folder in os.listdir(rollout_paths):
		for image in os.listdir(rollout_paths + "/" + folder):
			print image
			netify_image(net, model_path, layer, rollout_paths+ "/" + folder + "/" + image, destination + "/" + image + "_featurized.m")

def extract_rollout_data(rollout_paths="./../data/traindata", name = "/../data/featurized_images/"):
	states = []
	for folder in os.listdir(rollout_paths):
		for image in os.listdir(rollout_paths + "/" + folder):
			path = name + "/" + image + "_featurized.m"
			state += np.loadtxt(path)
	return np.array(state)




net =  net6.NetSix()
generate_rollout_values(net, "./../data/net6_04-08-2016_14h46m42s.ckpt", net.h_fc1, "./../data/traindata", "./../data/featurized_images")
