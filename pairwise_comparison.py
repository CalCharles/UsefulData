import sys
sys.path.append('/usr/local/lib/python2.7/dist-packages')

import tensorflow as tf
from Net.tensor import net6, inputdata
import cv2
import numpy as np
import os
import netify_data as netify



def compute_normalizing_value(name, ending, transform):
	mini = -1
	maxi = 0
	number = 0
	for img in os.listdir(name):
		if img.find(ending) != -1:
			print img
			im = transform(name+img)
			for oimg in os.listdir(name):
				if oimg.find(ending) != -1 and oimg != img:
					oim = transform(name +oimg)
					number += 1
					maxi = max(maxi, np.linalg.norm(im - oim))
					if mini >= 0:
						mini = min(mini, np.linalg.norm(im - oim))
					else:
						mini = np.linalg.norm(im - oim)
	return mini, maxi, number

def normalize(im, min_val, max_val):
	return (im-min_val)/float(max_val)

def compute_normalized_distances_raw_embed(name, min_val, max_val, othername, min_valother, max_valother, 
	transform, number, transformother, ending):
	total_dif = 0
	for img in os.listdir(name):
		if img.find(ending) != -1:
			im = transform(name + img)
			cmp_im = transformother(img, othername)
			for oimg in os.listdir(name):
				if oimg.find(ending) != -1 and oimg != img:
					oim = transform(name + oimg)
					ocmp_im = transformother(oimg, othername)
					dif = normalize(np.norm(im - oim), min_val, max_val)
					cmp_dif = normalize(np.norm(cmp_im - ocmp_im), min_valother, max_valother)
					total_dif += np.norm(dif-cmp_dif)
	return total_dif

def transform_rollout(image):
	return cv2.resize(cv2.imread(image), (250,250))

def transform_feature(image, name = ""):
	return np.loadtxt(image)

def transform_names_fromraw(image, othername):
	return transform_feature(othername + image[:image.find('.jpg')] + '_featurized.m')

minraw, maxraw, number = compute_normalizing_value("./../data/validationunorg/", ".jpg", transform_rollout)
minfeat, maxfeat, number = compute_normalizing_value("./../data/featurizedValidation/", ".m", transform_feature)

total = compute_normalized_distances_raw_embed("./../data/validationunorg/", minraw, maxraw, "./../data/validationFC1", minfeat, maxfeat, transform_rollout, number, transform_names_fromraw, ending)

print total
