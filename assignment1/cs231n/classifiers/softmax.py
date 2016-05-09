import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #


  #############################################################################
  num_train = X.shape[0]
  num_classes = W.shape[1]
  fs = (W.T).dot((X.T))
  for i in xrange(num_train):
    f = fs[:, i]
    # numerical stability trick
    f -= np.max(f)

    # probability for each class
    p = np.exp(f) / np.sum(np.exp(f), axis=0)
    loss += - f[y[i]] + np.log(np.sum(np.exp(f), axis=0))
    for j in xrange(num_classes):
      dW[:, j] += p[j] * X[i, :]
    # if i==j
    dW[:, y[i]] = dW[:, y[i]] - X[i, :]
  # mean gradient and loss
  dW /= num_train
  loss /= num_train
  # regularization
  loss += 0.5 * reg * np.sum(W ** 2)
  dW += reg * W
  # print dW.shape
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  f = (W.T).dot((X.T))
  num_train = X.shape[0]
  num_classes = W.shape[1]


  f -= np.amax(f, axis=0)

  prob = np.exp(f) / np.sum(np.exp(f), axis=0)
  prob[y, range(num_train)] -= 1

  # Data loss
  loss = np.sum(- f[y, range(num_train)] + np.log(np.sum(np.exp(f), axis=0))) / num_train

  # Weight sub-gradient
  dW = prob.dot(X) / num_train

  # Regularization term
  loss += 0.5 * reg * np.sum(W ** 2)
  print dW.shape,W.shape
  dW = dW.T
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

