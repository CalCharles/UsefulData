import tensorflow as tf
from Net.tensor import net6
import cv2
import numpy as np
import os

def netify_image(net, model_path, layer, image, name):
	sess = net.load(var_path = model_path)
	im = inputdata.im2tensor(image, channels = 3)

	imshape = image.shape
	im = np.reshape(image, (-1, imshape[0], imshape[1], imshape[2]))

	with sess.as_default():
		result = sess.run(layer, feed_dict={net.x:im})
		np.savetxt(name, result)
	sess.close()

def generate_rollout_values(net, model_path, layer, rollout_paths, destination):
	for folder in rollout_paths:
		for rollout_folder in os.listdir(folder):
			for image in os.listdir(rollout_folder):
				print rollout_folder
				netify_image(net, model_path, layer, image, destination + "/" rollout_folder + "_" + image + ".m")


generate_rollout_values(net6, "./../data/net6_04-08-2016_14h46m42s.ckpt", net6.h_fc1, "./../data/traindata", "./../data/featurizedImages/")
