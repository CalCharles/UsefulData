import learner
import netify_data as netify
import cv2
import os
from Net.tensor import net6, inputdata
import numpy as np

gamma = .003
nu = .001
validation = "./../data/validationdata"
net_validation = "./../data/featurizedValidation"
model_path = "./../data/net6_04-08-2016_14h46m42s.ckpt"

net =  net6.NetSix()
# netify.generate_rollout_values(net, "./../data/net6_04-08-2016_14h46m42s.ckpt", net.h_fc1, "./../data/traindata", "./../data/featurized_images")
Shiv_Bounds = learner.Learner()
Shiv_Bounds.Load(gamma, nu)
Shiv_Bounds.trainSupport()


rollout_frame = dict()
# netify.generate_rollout_values(net, "./../data/net6_04-08-2016_14h46m42s.ckpt", net.h_fc1, validation, net_validation)
for image in os.listdir(net_validation):
	frame_no = int(image[image.find("frame_") + 6:image.find("_featurized")])
	rollout_no = int(image[7:image.find("_frame")])
	img = np.loadtxt(net_validation + "/" + image)
	novel = Shiv_Bounds.askForHelp(img.reshape(1,-1))
	if novel == 1:
		if rollout_no not in rollout_frame or frame_no < rollout_frame[rollout_no]:
			rollout_frame[rollout_no] = frame_no
	else:
		print image
results = ""
for rollout_no in rollout_frame.keys():
	   results += str(rollout_no)+ "error at rollout: " + str(rollout_frame[rollout_no])  + "\n"
text_file = open("Output.txt", "w")
text_file.write(results)
text_file.close()
