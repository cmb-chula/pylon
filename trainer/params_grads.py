from torch import nn
import torch

def gradable_params(parameters):
    return [p for p in parameters if p.requires_grad and p.is_leaf]

def params_to_vec(parameters):
    # it could be mixed precision
    params = list(parameters)
    # convert to uniform types
    dtype = params[0].dtype
    pp = []
    for p in params:
        pp.append(p.type(dtype))
    return nn.utils.parameters_to_vector(gradable_params(pp))

def grads_from_params(parameters):
    """return a list of grads"""
    grads = [p.grad for p in gradable_params(parameters)]
    assert all(g is not None for g in grads), "grad must not be None"
    return grads

def grad_vec_from_params(parameters):
    """return a vec of grads"""
    grads = grads_from_params(parameters)
    return nn.utils.parameters_to_vector(grads)

def iter_opt_params(opt):
    """the save way to iterate the param under apex mix-precision training"""
    for g in opt.param_groups:
        for p in g['params']:
            yield p

def many_l2_norm(*tensors):
    """l2 norm of many tensors.
    It is the same as concatenate all the tensors and find its norm.
    """
    if len(tensors) == 0:
        return torch.zeros([])
    with torch.no_grad():
        norm = 0
        for t in tensors:
            _n = t.norm().float()  # it could float16 which could overflow
            norm = (norm**2 + _n**2).sqrt()
    return norm

def grads_norm(parameters):
    """get grad norms of the parameters, useful for model inspection"""
    t = [p.grad for p in parameters if p.grad is not None]
    return many_l2_norm(*t)