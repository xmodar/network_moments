import torch
from .utils import even_zip, special_sylvester


__all__ = ['triu_numel', 'mul_diag', 'outer',
           'normalize', 'normalize_', 'Flatten', 'flatten',
           'MatrixSquareRoot', 'sqrtm', 'map_batch']


def triu_numel(m, n=None, diagonal=0):
    '''Number of elements in the upper triangle of a matrix.

    Args:
        m: The number of rows or the size of the matrix.
        n: The number of columns.
        diagonal: The paramenter of `triu()`.

    Returns:
        The number of elements which is equivalent to
        `torch.triu(torch.ones(m, n), diagonal=d).sum().item()`.
    '''
    if n is None:
        if hasattr(m, '__len__'):
            m, n = m
        else:
            n = m
    if m <= 0 or n <= 0:
        return 0
    diagonal = min(max(diagonal, 1 - m), n)
    if diagonal == 0:
        sum_to = lambda n: n * (n + 1) // 2  # torch.arange(n + 1).sum().item()
        cut = 0 if m >= n else sum_to(n - m)
        return sum_to(n) - cut
    cut = 0 if diagonal > 0 else triu_numel(abs(diagonal))
    return triu_numel(m, n - diagonal) - cut


def mul_diag(A, vec):
    '''Batch support for matrix-diag(vector) product.

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


def normalize(vec, norm=1):
    '''Normalize a batch of vectors.

    Args:
        vec: A vector of size (Batch, Size).
        norm: The new norm (default is 1).

    Returns:
        The normalized vector.
    '''
    if not torch.is_tensor(norm):
        norm = torch.tensor(norm, dtype=vec.dtype, device=vec.device)
    new_norm = norm / torch.norm(vec, dim=-1)
    new_vec = vec.view(vec.size(0), -1) * (new_norm.unsqueeze(-1))
    return new_vec.view(vec.size())


def normalize_(vec, norm=1):
    '''Normalize a batch of vectors in-place.

    Args:
        vec: A vector of size (Batch, Size).
        norm: The new norm (default is 1).

    Returns:
        The normalized vector (Batch, size).
    '''
    if not torch.is_tensor(norm):
        norm = torch.tensor(norm, dtype=vec.dtype, device=vec.device)
    new_norm = norm / torch.norm(vec, dim=-1)
    vec.view(vec.size(0), -1).mul_(new_norm.unsqueeze(-1))
    return vec


def map_batch(func, *inputs):
    '''Apply a function on a batch of data and stack the results.

    Args:
        func: The function to map.
        inputs: Batch arguments list (Batch, *size).

    Returns:
        Similar to `torch.stack([func(*x) for x in zip(*inputs)])` but faster.
        In case `func` returns a tuple, this function will also return a tuple.
    '''
    # compute the output of the first instance in the batch
    inputs = list(even_zip(*inputs))
    res = func(*inputs[0])
    single = not isinstance(res, tuple)
    as_tuple = (lambda x: (x,)) if single else (lambda x: x)
    res = as_tuple(res)

    # if more outputs are expected, keep computing them
    if len(inputs) == 1:
        out = tuple(r.unsqueeze(0) for r in res)
    else:
        out = tuple(torch.empty(len(inputs), *r.size(),
                                device=r.device, dtype=r.dtype)
                    for r in res)
        for i, args in enumerate(inputs):
            if i > 0:
                res = as_tuple(func(*args))
            for result, output in zip(res, out):
                output[i, ...] = result
    return out[0] if single else out


class Flatten(torch.nn.Module):
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


class MatrixSquareRoot(torch.autograd.Function):
    '''Square root of a positive definite matrix.
    NOTE: matrix square root is not differentiable for matrices with
          zero eigenvalues.
    '''
    @staticmethod
    def forward(ctx, matrix):
        '''Computes the square root of a matrix.

        This was adopted from: https://github.com/steveli/pytorch-sqrtm

        Multiple different implementations can be found here:
        https://github.com/msubhransu/matrix-sqrt

        Args:
            matrix: The matrix.

        Returns:
            The square root of the `matrix` matrix.
        '''
        # scipy.linalg.sqrtm(matrix.data.cpu().numpy()).real
        U, s, V = matrix.data.svd()
        ctx.save_for_backward(U, s.sqrt_())
        return U.mm(V.t_().mul_(s.unsqueeze(-1)))  # U*diag(sqrt(s))*V^T

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        if ctx.needs_input_grad[0]:
            Q = ctx.saved_tensors[0]
            d = None if len(ctx.saved_tensors) < 2 else ctx.saved_tensors[1]
            # Given a positive semi-definite matrix X,
            # since X = X^{1/2}X^{1/2}, we can compute the gradient of the
            # matrix square root dX^{1/2} by solving the Sylvester equation:
            # dX = (d(X^{1/2})X^{1/2} + X^{1/2}(dX^{1/2}).
            grad_input = special_sylvester(Q, grad_output, d)
        return grad_input


sqrtm = MatrixSquareRoot.apply
