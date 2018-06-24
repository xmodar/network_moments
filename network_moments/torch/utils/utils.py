import torch
from torch import nn
from torch.autograd import Variable
from torch.autograd.gradcheck import zero_gradients


def epsilon(dtype=None, device=None, _cache={}):
    '''Machine epsilon for a specific torch.dtype.'''
    if dtype in _cache:
        return _cache[dtype, device]
    value = one = torch.ones([], dtype=dtype, device=device)
    while one + value != one:
        machine_epsilon = value
        value = value >> 1
    _cache[dtype, device] = machine_epsilon
    return machine_epsilon


def diagonal(input, offset=0, dim1=0, dim2=1, view=True):
    '''Forward compatibility for torch.diagonal() in 0.4.0.'''
    if torch.__version__ > '0.4.0':
        return torch.diagonal(input, offset, dim1, dim2)
    else:
        out = torch.from_numpy(
            input.cpu().numpy().diagonal(offset, dim1, dim2))
        if input.is_cuda:
            out = out.cuda(input.device)
        return out if view else out.clone()


def mul_diag(A, vec):
    '''Batch support for matrix diag(vector) product.

    Args:
        A: General matrix of size (M, Size).
        vec: Vector of size (Batch, Size).

    Returns:
        The result of multiplying A with diag(vec) (Batch, M).
    '''
    return A * torch.unsqueeze(vec, - 2)


def outer(vec1, vec2=None):
    '''Batch support for vectors outer products.

    This function is broadcast-able,
    so you can provide batched vec1 or batched vec2 or both.

    Args:
        vec1: A vector of size (Batch, Size1).
        vec2: A vector of size (Batch, Size2)
            if vec2 is None, vec2 = vec1.

    Returns:
        The outer product of vec1 and vec2 (Batch, Size1, Size2).
    '''
    if vec2 is None:
        vec2 = vec1
    if len(vec1.size()) == 1 and len(vec2.size()) == 1:
        return torch.ger(vec1, vec2)
    else:  # batch outer product
        if len(vec1.size()) == 1:
            vec1 = torch.unsqueeze(vec1, 0)
        if len(vec2.size()) == 1:
            vec2 = torch.unsqueeze(vec2, 0)
        vec1 = torch.unsqueeze(vec1, -1)
        vec2 = torch.unsqueeze(vec2, -2)
        if vec1.size(0) == vec2.size(0):
            return torch.bmm(vec1, vec2)
        else:
            return vec1.matmul(vec2)


def normalize(vec, norm=1.0):
    '''Normalize a batch of vectors.

    Args:
        vec: A vector of size (Batch, Size).
        norm: The new norm (default is 1.0).

    Returns:
        The normalized vector.
    '''
    new_norm = norm / torch.norm(vec, dim=-1)
    new_vec = vec.view(vec.size(0), -1) * (new_norm.unsqueeze(-1))
    return new_vec.view(vec.size())


def normalize_(vec, norm=1):
    '''Normalize a batch of vectors in-place.

    Args:
        vec: A vector of size (Batch, Size).
        norm: The new norm (default is 1.0).

    Returns:
        The normalized vector (Batch, size).
    '''
    new_norm = norm / torch.norm(vec, dim=-1)
    vec.view(vec.size(0), -1).mul_(new_norm.unsqueeze(-1))
    return vec


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
    '''Approximate the output of a model at a given point with an affine function.

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
    inputs = Variable(at, requires_grad=True)
    outputs = flatten(model(inputs))
    grad_output = torch.zeros_like(outputs)
    jacobian = torch.empty(outputs.size(1), *inputs.size(),
                           dtype=inputs.dtype, device=inputs.device)
    for i in range(outputs.size(1)):
        zero_gradients(inputs)
        grad_output.zero_()
        grad_output[:, i] = 1.0
        outputs.backward(grad_output, retain_graph=True)
        jacobian[i] = inputs.grad

    A = flatten(jacobian.transpose_(0, 1), 2)
    if jacobian_only:
        return A
    batch_matmul = A.matmul(flatten(at).unsqueeze(-1)).squeeze(-1)
    b = outputs.detach() - batch_matmul
    return A, b


class Flatten(nn.Module):
    def forward(self, inputs, dim=1):
        '''Flatten a tensor.

        Keep the first `dim`-dimensions untouched and flatten the rest.

        Args:
            inputs: The input tensor.
            dim: Number of dimensions to skip.

        Returs:
            The flattened tensor.
        '''
        return inputs.view(*inputs.size()[:dim], -1)


flatten = Flatten()


def rand_from_eigen(eigen):
    '''Construct a random matrix with given the eigenvalues.

    To construct such a matrix form the eigenvalue decomposition,
    (i.e. U * Sigma * U.t()), we need to find a unitary matrix U
    and Sigma is the diagonal matrix of the eigenvalues `eigen`.
    The matrix U can be the unitary matrix Q from
    the QR-decomposition of a randomly generated matrix.

    Args:
        eigen: A vector of size (Batch, Size).

    Returns:
        A random matrix of size (Batch, Size, Size).
    '''
    size = eigen.size(-1)
    Q, _ = torch.qr(torch.randn(
        (size, size), dtype=eigen.dtype, device=eigen.device))
    return mul_diag(Q, eigen).matmul(Q.t())


def rand_definite(size, batch=None, norm=None,
                  positive=True, semi=False, dtype=None, device=None):
    '''Random definite matrix.

    A positive/negative definite matrix is a matrix
    with positive/negative eigenvalues, respectively.
    They are called semi-definite if the eigenvalues are allowed to be zeros.
    The eigenvalues are some random vector of unit norm.
    This vector is what control (positive vs. negative)
    and (semi-definite vs definite).
    We multiply this vector by the desired `norm`.

    Args:
        size: The output matrix is of size (`size`, `size`).
        batch: Number of matrices to generate.
        norm: The Frobenius norm of the output matrix.
        positive: Whether positive-definite or negative-definite.
        semi: Whether to construct semi-definite or definite matrix.
        dtype: The data type.
        device: In which device.

    Returns:
        Random definite matrices of size (`Batch`, `size`, `size`)
        and Frobenius norm `norm`.
    '''
    shape = size if batch is None else (batch, size)
    eigen = torch.rand(shape, dtype=dtype, device=device)
    if not semi:
        eigen = 1.0 - eigen
    if not positive:
        eigen = -eigen
    eigen = eigen if norm is None else normalize_(eigen, norm)
    return rand_from_eigen(eigen)
