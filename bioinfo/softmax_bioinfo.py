from cs231n.classifiers.softmax import softmax_loss_naive
from cs231n.classifiers.softmax import softmax_loss_vectorized
import h5py
from cs231n.classifiers import Softmax
from numpy import loadtxt
import numpy as np
h5f = h5py.File('img_data.h5','r')
X = h5f['dataset_1'][:]
h5f.close()
y = loadtxt("y_labels.txt", dtype=np.uint8, delimiter="\n", unpack=False)

#X_train = np.zeros((27116,196608))
#y_train = np.zeros(27116)
#X_val = np.zeros((5000,196608))
#y_val = np.zeros(5000)

X_train = X[8000:35117,:]
y_train = y[8000:35117]
X_val=X[3000:8000,:]
y_val=y[3000:8000]
# Generate a random softmax weight matrix and use it to compute the loss.
W = np.random.randn(196608, 5) * 0.0001
loss_vectorized, grad_vectorized = softmax_loss_vectorized(W, X_train, y_train, 0.00001)
softmax=Softmax()
loss_hist = softmax.train(X_train, y_train, learning_rate=1e-7, reg=5e4,
                      num_iters=1500, verbose=False)
y_train_pred = softmax.predict(X_train)
training_accuracy = np.mean(y_train == y_train_pred)
y_val_pred = softmax.predict(X_val)
val_accuracy = np.mean(y_val == y_val_pred)
print 'training accuracy: %f' % (np.mean(y_train == y_train_pred), )
print 'validation accuracy: %f' % (np.mean(y_val == y_val_pred), )
