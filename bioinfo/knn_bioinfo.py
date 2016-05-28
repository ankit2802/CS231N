import numpy as np
import h5py
from numpy import loadtxt
from cs231n.classifiers import KNearestNeighbor

h5f = h5py.File('img_data.h5','r')
X = h5f['dataset_1'][:]
h5f.close()
y = loadtxt("y_labels.txt", dtype=np.uint8, delimiter="\n", unpack=False)
X_train = X[8000:35117,:]
y_train = y[8000:35117]
X_val=X[3000:8000,:]
y_val=y[3000:8000]
num_val = 5000

# Create a kNN classifier instance. 
# Remember that training a kNN classifier is a noop: 
# the Classifier simply remembers the data and does no further processing 
classifier = KNearestNeighbor()
classifier.train(X_train, y_train)
dists = classifier.compute_distances_no_loops(X_val)
y_val_pred = classifier.predict_labels(dists, k=5)
num_correct = np.sum(y_val_pred == y_val)
accuracy = float(num_correct) / num_val
print accuracy


