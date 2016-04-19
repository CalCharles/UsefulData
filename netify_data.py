import tensorflow as tf
from Net.tensor import net6
import cv2

def netify_image(net, model_path, layer, image):
	sess = net.load(var_path = model_path)
	im = inputdata.im2tensor(image, channels = 3)

	imshape = image.shape
	im = np.reshape(image, (-1, imshape[0], imshape[1], imshape[2]))

	with sess.as_default():
		result = sess.run(layer, feed_dict={net.x:im})