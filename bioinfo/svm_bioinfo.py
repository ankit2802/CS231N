from cs231n.classifiers.linear_svm import svm_loss_naive
from cs231n.classifiers.linear_svm import svm_loss_vectorized
from cs231n.classifiers import LinearSVM
import h5py
from numpy import loadtxt
import numpy as np

# generate a random SVM weight matrix of small numbers
W = np.random.randn(196608, 5) * 0.0001 

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
loss_vectorized, grad_vectorized = svm_loss_vectorized(W, X_train, y_train, 0.00001)
svm = LinearSVM()
loss_hist = svm.train(X_train, y_train, learning_rate=1e-7, reg=5e4,num_iters=1500, verbose=True)
y_train_pred = svm.predict(X_train)
print 'training accuracy: %f' % (np.mean(y_train == y_train_pred), )
y_val_pred = svm.predict(X_val)
print 'validation accuracy: %f' % (np.mean(y_val == y_val_pred), )
