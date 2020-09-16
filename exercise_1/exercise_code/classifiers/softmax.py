"""Linear Softmax Classifier."""
# pylint: disable=invalid-name
import numpy as np

from .linear_classifier import LinearClassifier


def cross_entropy_loss_naive(W, X, y, reg):
    """
    Cross-entropy loss function, naive implementation (with loops)

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
    # pylint: disable=too-many-locals
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    ############################################################################
    # TODO: Compute the cross-entropy loss and its gradient using explicit     #
    # loops. Store the loss in loss and the gradient in dW. If you are not     #
    # careful here, it is easy to run into numeric instability. Don't forget   #
    # the regularization!                                                      #
    ############################################################################

    N = X.shape[0] # number of images
    C = W.shape[1] # number of classes
    
    for i in range(N):
      
        yexp = np.dot(X[i],W) # yexp is expected value of y-vector, dim : 1xC
        prob = np.exp(yexp)/np.sum(np.exp(yexp)) # prob is probabilty vector calculated from y-vector
        loss += -np.log(prob[y[i]]) # calculation for the loss function : log of probabilty of the class which the image belongs
        prob[y[i]] -= 1 
        
        for j in range(C):
            dW[:,j] += X[i,:]*prob[j] # calculation for gradient dW
  
    loss /= N
    dW /= N
    loss += 0.5*reg*np.sum(W**2) # addition of reguarization in loss function
    dW += reg*W 

    ############################################################################
    #                          END OF YOUR CODE                                #
    ############################################################################

    return loss, dW


def cross_entropy_loss_vectorized(W, X, y, reg):
    """
    Cross-entropy loss function, vectorized version.

    Inputs and outputs are the same as in cross_entropy_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    ############################################################################
    # TODO: Compute the cross-entropy loss and its gradient without explicit   #
    # loops. Store the loss in loss and the gradient in dW. If you are not     #
    # careful here, it is easy to run into numeric instability. Don't forget   #
    # the regularization!                                                      #
    ############################################################################

    N=X.shape[0] 
    Yexp=np.dot(X,W) # expected value of y stored in Matrix Yexp with dim NxD
    Yexp=Yexp-np.reshape(Yexp.max(axis=1),(N,1)) # substraction of maximum value of each image from all dimentions of that image
    Prob=np.exp(Yexp)/np.sum(np.exp(Yexp),axis=1,keepdims=True) # calucaltion of probabilty matrix with dim NXD
    ProbTrue=np.zeros((N,10)) # generation of true probabilty matrix
    ProbTrue[range(N),y]=1.0 # Putting 1.0 at the place of actual class location in True probabilty matrix
    loss=-np.sum(ProbTrue*np.log(Prob))/N+0.5*reg*np.sum(W*W) # calculation of loss function 
    dW=(Prob-ProbTrue)/N
    dW=np.dot(X.T,dW)+reg*W # calculation of gradient matrix

    ############################################################################
    #                          END OF YOUR CODE                                #
    ############################################################################

    return loss, dW


class SoftmaxClassifier(LinearClassifier):
    """The softmax classifier which uses the cross-entropy loss."""

    def loss(self, X_batch, y_batch, reg):
        return cross_entropy_loss_vectorized(self.W, X_batch, y_batch, reg)


def softmax_hyperparameter_tuning(X_train, y_train, X_val, y_val):
    # results is dictionary mapping tuples of the form
    # (learning_rate, regularization_strength) to tuples of the form
    # (training_accuracy, validation_accuracy). The accuracy is simply the
    # fraction of data points that are correctly classified.
    results = {}
    best_val = -1
    best_softmax = None
    all_classifiers = []
    learning_rates = [1e-7, 5e-7]
    regularization_strengths = [2.5e4, 5e4]

    ############################################################################
    # TODO:                                                                    #
    # Write code that chooses the best hyperparameters by tuning on the        #
    # validation set. For each combination of hyperparameters, train a         #
    # classifier on the training set, compute its accuracy on the training and #
    # validation sets, and  store these numbers in the results dictionary.     #
    # In addition, store the best validation accuracy in best_val and the      #
    # Softmax object that achieves this accuracy in best_softmax.              #
    #                                                                          #
    # Hint: You should use a small value for num_iters as you develop your     #
    # validation code so that the classifiers don't take much time to train;   # 
    # once you are confident that your validation code works, you should rerun #
    # the validation code with a larger value for num_iters.                   #
    ############################################################################

    for lr in learning_rates:
        for r in regularization_strengths:
            classifier = SoftmaxClassifier() # generating softmax classifier
            loss = classifier.train(X_train, y_train, learning_rate=lr, reg=r ,num_iters=2000, verbose=False) # calculate loss
            y_train_pred = classifier.predict(X_train) # generate predicted values of y for training samples
            training_accuracy = np.mean(y_train == y_train_pred) # calculation of accuracy for training samples
            y_val_pred = classifier.predict(X_val) # generate predicted values of y for validation samples
            validation_accuracy = np.mean(y_val == y_val_pred) # calculation of accuracy for validation samples
            results[(lr, r)] = (training_accuracy, validation_accuracy) # storing data in result-dictionary
            all_classifiers.append(classifier) # storing classifiers in a list
            if validation_accuracy >= best_val: # checking for the best accuracy
                best_val = validation_accuracy 
                best_softmax = classifier # return best classifier

    ############################################################################
    #                              END OF YOUR CODE                            #
    ############################################################################
        
    # Print out results.
    for (lr, reg) in sorted(results):
        train_accuracy, val_accuracy = results[(lr, reg)]
        print('lr %e reg %e train accuracy: %f val accuracy: %f' % (
              lr, reg, train_accuracy, val_accuracy))
        
    print('best validation accuracy achieved during validation: %f' % best_val)

    return best_softmax, results, all_classifiers
