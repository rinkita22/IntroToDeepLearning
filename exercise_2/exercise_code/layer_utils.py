from exercise_code.layers import *


def affine_relu_forward(x, w, b):
    """
    Convenience layer that perorms an affine transform followed by a ReLU
  
    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer
  
    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, fc_cache = affine_forward(x, w, b)
    out, relu_cache = relu_forward(a)
    cache = (fc_cache, relu_cache)
    return out, cache


def affine_relu_backward(dout, cache):
    """
    Backward pass for the affine-relu convenience layer
    """
    fc_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db

def affine_batchnorm_relu_forward(x, w, b, gamma, beta, bn_params):
    """
    combines affine transform, batch normalization and ReLU
    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer
    - gamma, beta: scale and shift parameter for batch_norm
    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    af_out, af_cache = affine_forward(x, w, b)
    bf_out, bf_cache = batchnorm_forward(af_out, gamma, beta, bn_params)
    out, relu_cache = relu_forward(bf_out)
    
    cache = (af_cache, bf_cache, relu_cache)
    return out, cache

def affine_batchnorm_relu_backward(dout, cache):
    """
    Backwardpass for combined ReLU,batchnorm and affine forward
    """
    af_cache, bf_cache, relu_cache = cache
    
    dbf_out = relu_backward(dout, relu_cache)
    daf_out, dgamma, dbeta = batchnorm_backward(dbf_out, bf_cache)
    dx, dw, db = affine_backward(daf_out, af_cache)
    return dx, dw, db, dgamma, dbeta

