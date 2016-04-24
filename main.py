from learner import learner
from netify_data import netify
import cv2
import os

gamma = .001
validation = "./../data/validationdata/"
rollo

net =  net6.NetSix()
netify.generate_rollout_values(net, "./../data/net6_04-08-2016_14h46m42s.ckpt", net.h_fc1, "./../data/traindata", "./../data/featurized_images")
Shiv_Bounds = learner.Learner()
Shiv_Bounds.Load(gamma)
Shiv_Bounds.trainSupport()

rollout_frame = dict()
for folder in os.listdir(validation):
    for value in os.listdir(validation + folder):
        sess = net.load(var_path = model_path)
        img = cv2.resize(cv2.imread(image), (250,250))
        novel = learner.askForHelp(img)
        if novel == 1:
            rollout_frame[folder] = value

    