"""Linear Softmax Classifier."""
# pylint: disable=invalid-name
import numpy as np

from .linear_classifier import LinearClassifier


def cross_entropoy_loss_naive(W, X, y, reg):
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
    

    C = W.shape[1]
#    print("no. of classes {}".format(C))
    N,D = X.shape
#    print("no. of data {} and dimension {}".format(N,D))
    for i in range(N):
        xi = X[i,:]
#        print("one record shape: {}".format(xi.shape))
        scores = np.zeros(C)
        for c in range(C):
            w = W[:,c]
#            print("weight for one record {}".format(w.shape))
            scores[c] = xi.dot(w)
        scores -= np.max(scores)
        actual_y = y[i]
        total_score = np.sum(np.exp(scores))        
        loss_i = -scores[actual_y] + np.log(total_score)
#        print('naive score : {}'.format(scores[actual_y]))
        loss += loss_i
        
        #gradient
        probability = np.exp(scores)/total_score
        for j in range(C):
            dW[:,j] += probability[j]*xi
            
        dW[:,actual_y] -= xi
    loss = loss/N
    reg_loss = 0.5*reg*np.sum(W*W)
    loss = loss + reg_loss
    print("loss : {}".format(loss))
    dW = dW/N
    dW += reg*W
    
    
            
    

    ############################################################################
    # TODO: Compute the cross-entropy loss and its gradient using explicit     #
    # loops. Store the loss in loss and the gradient in dW. If you are not     #
    # careful here, it is easy to run into numeric instability. Don't forget   #
    # the regularization!                                                      #
    ############################################################################

    ############################################################################
    #                          END OF YOUR CODE                                #
    ############################################################################

    return loss, dW


def cross_entropoy_loss_vectorized(W, X, y, reg):
    """
    Cross-entropy loss function, vectorized version.

    Inputs and outputs are the same as in cross_entropoy_loss_naive.
    """    
    loss = 0 
    dW = np.zeros_like(W)
    N, D = X.shape 
    
    
    scores = X.dot(W) # [K, N]
    # for numerical stability)
    scores -= np.max(scores)
    scores_exp = np.exp(scores)
    correct_scores_exp = scores_exp[range(N),y] 
    scores_exp_sum = np.sum(scores_exp, axis=1) 
    loss = -np.sum(np.log(correct_scores_exp / scores_exp_sum))
    loss /= N
    loss += 0.5 * reg * np.sum(W * W)

    scores_exp_normalized = scores_exp.T / scores_exp_sum    
    scores_exp_normalized[y, range(N)] -= 1 
    dW = np.dot(scores_exp_normalized,X)
    dW = dW.T
    dW /= N    
    dW += reg * W
   
    
    ############################################################################
    # TODO: Compute the cross-entropy loss and its gradient without explicit   #
    # loops. Store the loss in loss and the gradient in dW. If you are not     #
    # careful here, it is easy to run into numeric instability. Don't forget   #
    # the regularization!                                                      #
    ############################################################################

    ############################################################################
    #                          END OF YOUR CODE                                #
    ############################################################################

    return loss, dW


class SoftmaxClassifier(LinearClassifier):
    """The softmax classifier which uses the cross-entropy loss."""

    def loss(self, X_batch, y_batch, reg):
        return cross_entropoy_loss_vectorized(self.W, X_batch, y_batch, reg)


def softmax_hyperparameter_tuning(X_train, y_train, X_val, y_val):
    # results is dictionary mapping tuples of the form
    # (learning_rate, regularization_strength) to tuples of the form
    # (training_accuracy, validation_accuracy). The accuracy is simply the
    # fraction of data points that are correctly classified.
    results = {}
    best_val = -1
    best_softmax = None     
    all_classifiers = []   
    learning_rates = [2e-3,2e-2]
    regularization_strengths= [0.45,0.25,0.1]
    

    for lr in learning_rates:
        for reg in regularization_strengths:
            softmax = SoftmaxClassifier()
            softmax.train(X_train, y_train, lr, reg,
                          num_iters=1500, verbose=False)
            y_val_pred = softmax.predict(X_val)            
            y_train_pred = softmax.predict(X_train)
            train_accuracy = np.mean(y_train == y_train_pred)
            val_accuracy = np.mean(y_val == y_val_pred)
            results[(lr,reg)] = (train_accuracy,val_accuracy)
            all_classifiers.append(softmax)
            if(val_accuracy > best_val):
                best_val = val_accuracy
                best_softmax = softmax
            
            
    ############################################################################
    # TODO:                                                                    #
    # Write code that chooses the best hyperparameters by tuning on the        #
    # validation set. For each combination of hyperparameters, train a         #
    # classifier on the training set, compute its accuracy on the training and #
    # validation sets, and  store these numbers in the results dictionary.     #
    # In addition, store the best validation accuracy in best_val and the      #
    # Softmax object that achieves this accuracy in best_softmax.              #                                      #
    #                                                                          #
    # Hint: You should use a small value for num_iters as you develop your     #
    # validation code so that the classifiers don't take much time to train;   # 
    # once you are confident that your validation code works, you should rerun #
    # the validation code with a larger value for num_iters.                   #
    ############################################################################

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
