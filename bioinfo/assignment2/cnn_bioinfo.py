import numpy as np
from cs231n.classifiers.cnn import *
from cs231n.layers import *
from cs231n.solver import Solver
import h5py
from numpy import loadtxt
h5f = h5py.File('../img_data.h5','r')
X = h5f['dataset_1'][:]
h5f.close()
print X.shape
#load data
#'data=??
y = loadtxt("y_labels.txt", dtype=np.uint8, delimiter="\n", unpack=False)
data={
	'X_train':X[8000:35117,:],
	#load labels
	'y_train':y[8000:35117],
	'X_val':X[3000:8000,:],
	'y_val':y[3000:8000]
}


num_inputs = 35126
input_dim = (3, 256, 256)
reg = 0.1
num_classes = 5
model = ThreeLayerConvNet(num_filters=5, filter_size=5,input_dim=input_dim, hidden_dim=7,num_classes=5,dtype=np.float64,reg=reg)
solver = Solver(model, data,
                num_epochs=1, batch_size=5000,
                update_rule='adam',
                optim_config={
                  'learning_rate': 1e-3,
                },
                verbose=True, print_every=20)
solver.train()
