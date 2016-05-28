import matplotlib.image as mpimg
import os
import numpy as np
import h5py
from numpy import loadtxt

from  assignment2.cs231n.classifiers.fc_net import * 
from assignment2.cs231n.solver import *
h5f = h5py.File('img_data.h5','r')
X = h5f['dataset_1'][:]
h5f.close()

best_model = None
################################################################################
# TODO: Train the best FullyConnectedNet that you can on CIFAR-10. You might   #
# batch normalization and dropout useful. Store your best model in the         #
# best_model variable.                                                         #
################################################################################
'''
Data must be of the format
  data = {
    'X_train': # training data
    'y_train': # training labels
    'X_val': # validation data
    'X_train': # validation labels
  }
'''
y = loadtxt("y_labels.txt", dtype=np.uint8, delimiter="\n", unpack=False)
data={
	'X_train':X[8000:35117,:],
	#load labels
	'y_train':y[8000:35117],
	'X_val':X[3000:8000,:],
	'y_val':y[3000:8000]
}

lr = 2.113669e-04
ws = 1.461858e-02

model = FullyConnectedNet([100, 100, 100, 100],weight_scale=ws, dtype=np.float64,use_batchnorm=True, reg= 1e-2)
solver = Solver(model, data,print_every=100, num_epochs=5, batch_size=10000,update_rule='adam',optim_config={'learning_rate': lr,},lr_decay = 0.9,verbose = True)   
solver.train()
