import torch
from .ops import flatten


__all__ = ['jac_at_x', 'lin_at_x', 'jacobian', 'linearize']


def jac_at_x(func, at, x, eps=1e-3):
    '''Computes `Ax` where `A` is the Jacobian of `func` at `at`.

    This implementation doesn't need to compute `A` to return `Ax`.
    We can numerically approximate `Ax` with `a*(func(at + x/a) - func(at))`.
    Where `a > 0` such that `x / a` is small. Here `a = x.abs().max() / eps`.

    Args:
        func: The function to linearize with the Jacobian `A`.
        at: The point of linearization (*Size).
        x: The points of evaluation (Batch, *Size).

    Returns:
        The estimated value of `Ax`.
    '''
    # alpha = x.abs().max() / eps
    alpha = x.data.view(x.size(0), -1).abs().max(-1)[0].clamp(min=1.0) / eps
    # delta = x / alpha
    delta = x / alpha.view(-1, *[1] * (x.dim() - 1))
    # return alpha * (func(at + delta) - func(at))
    diff = func(at + delta) - func(at)
    return alpha.view(-1, *[1] * (diff.dim() - 1)) * diff


def lin_at_x(func, at, x, eps=1e-3):
    '''Computes the linearization of `func approx Ax+b` at `at`.

    This implementation doesn't need to compute `A` to return `Ax+b`.
    We can approximate `Ax+b` with `(1-a)*f(at) + a*func(at + (x-at)/a)`.
    Where `a > 0` such that `(x-at) / a` is small.
    Here `a = (x-p).abs().max() / eps`.

    Args:
        func: The function to linearize with the Jacobian `A`.
        at: The point of linearization (*Size).
        x: The points of evaluation (Batch, *Size).

    Returns:
        The estimated value of `Ax+b`.
    '''
    diff = x - at
    # alpha = diff.abs().max() / eps
    alpha = diff.data.view(x.size(0), -1).abs().max(-1)[0].clamp(min=1.0) / eps
    # delta = diff / alpha
    delta = diff / alpha.view(-1, *[1] * (x.dim() - 1))
    # return (1-alpha)*func(at) + alpha*func(at + delta)
    out = (1 - alpha) * func(at)
    return out + alpha.view(-1, *[1] * (out.dim() - 1)) * func(at + delta)


def jacobian(model, at):
    '''Compute the Jacobian matrix of a model at a given input.

    Args:
        model: The model as a callable object.
        at: Batch of points at which to compute the Jacobian (Batch, *Size)
            Batch must be at least 1.

    Returns
        The Jacobian of the model at all the points in `at`.
    '''
    return linearize(model, at, jacobian_only=True)


def linearize(model, at, jacobian_only=False):
    '''Affine approximation for a model at a given point.

    The first order Taylor decomposition of `model` at the point `at` is
    an affine transformation `f(x) = A * x + b` such that `model(at) = f(at)`.
    The matrix A is actually the Jacobian matrix of model at the point `at`.
    Then, b is simply `b = model(at) - A * x`.

    Args:
        model: The model as a callable object.
        at: Batch of points (Batch, *Size) at which to compute the Jacobian A
            Batch must be at least 1.
        jacobian_only: Whether to return only A.

    Returns
        The matrix A and the bias vector b.
    '''
    with torch.enable_grad():
        inputs = torch.autograd.Variable(at, requires_grad=True)
        outputs = flatten(model(inputs))
    with torch.no_grad():
        grad_output = torch.empty_like(outputs)
        jac = torch.empty(outputs.size(1), *inputs.size(),
                          dtype=inputs.dtype, device=inputs.device)
        for i in range(outputs.size(1)):
            if inputs.grad is not None:
                inputs.grad.data.zero_()
            grad_output.zero_()
            grad_output[:, i] = 1
            jac[i, ...], = torch.autograd.grad(
                outputs, inputs,
                grad_outputs=grad_output,
                retain_graph=i + 1 < outputs.size(1),
                allow_unused=True)
        A = flatten(jac.transpose_(0, 1), 2)
        if jacobian_only:
            return A
        at = at.view(at.size(0), -1, 1)
        b = outputs.data - A.bmm(at).squeeze_(-1)
        return A, b
