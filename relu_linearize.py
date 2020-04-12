import torch
from torch import nn

__all__ = ['relu_linearize']


def linear_from_parameters(weight, bias=None):
    """Create torch.nn.Linear given weight and, maybe, bias."""
    linear = nn.Linear.__new__(nn.Linear)
    nn.Module.__init__(linear)
    if not isinstance(weight, nn.Parameter):
        weight = nn.Parameter(weight)
    if bias is not None and not isinstance(bias, nn.Parameter):
        bias = nn.Parameter(bias)
    linear.register_parameter('weight', weight)
    linear.register_parameter('bias', bias)
    linear.out_features, linear.in_features = weight.shape
    return linear


@torch.no_grad()
def linearize(net, x):
    """Compute the affine approximation for a model around a point `x`.

    Args:
        net: torch.nn.Module.
        x: Point of approximation without batch dimension.

    Returns:
        (weights_matrix, bias_vector).
        If `net` was a linear layer, the data of the parameters
        will be returned instead of a computed clone (handle with care).

    """
    assert not net.training
    if isinstance(net, nn.Sequential) and len(net) == 1:
        return linearize(net[0], x)
    output = net(x.unsqueeze(0))[0]
    if isinstance(net, nn.Linear):
        weight = net.weight.data
        if net.bias is None:
            bias = weight.new_zeros(net.out_features)
        else:
            bias = net.bias.data
    else:
        x = x.expand(output.numel(), *x.shape)
        eye = x.new_zeros(output.numel(), output.numel())
        eye.diagonal().fill_(1)
        eye = eye.view(output.numel(), *output.shape)
        with torch.enable_grad():
            weight, = torch.autograd.grad(net(x.requires_grad_(True)), x, eye)
        weight = weight.view(output.numel(), -1)
        bias = output.view(-1) - (weight @ x[0].view(-1))
    return dict(output=output, weight=weight, bias=bias)


@torch.no_grad()
def relu_linearize(net, x, relu_index=-1):
    """Linearize a sequential model around the any ReLU layer."""
    assert isinstance(net, nn.Sequential) and not net.training
    i = [i for i, m in enumerate(net) if isinstance(m, nn.ReLU)][relu_index]
    a = linearize(net[:i].train(False), x)
    b = linearize(net[i + 1:].train(False), a['output'].clamp_(0))
    a = linear_from_parameters(a['weight'], a['bias'])
    b = linear_from_parameters(b['weight'], b['bias'])
    return nn.Sequential(a, nn.ReLU(), b)
