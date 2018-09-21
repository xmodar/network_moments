import torch


__all__ = ['epsilon', 'even_zip', 'special_sylvester']


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


def even_zip(*lists):
    '''Similar to zip() but prioritizes the longest iterable.'''
    iterators = [iter(el) for el in lists]
    feed = [None] * len(lists)
    while True:
        done = True
        for i in range(len(iterators)):
            try:
                feed[i] = next(iterators[i])
                done = False
            except StopIteration:
                pass
        if done:
            break
        yield tuple(feed)


def special_sylvester(A, B, d=None):
    '''Solves the eqations `AX+XA=B` for positive definite `A`.

    This is a special case of Sylvester equation `AX+XB=C`.
    https://en.wikipedia.org/wiki/Sylvester_equation
    A unique solution exists when `A` and `-A` have no common eigenvalues.

    Sources:
        https://math.stackexchange.com/a/820313
        Explicit solution of the operator equation A*X+X*A=B:
        https://core.ac.uk/download/pdf/82315631.pdf

    Args:
        A: The matrix `A`.
        B: The matrix `B`.
        d: The eigenvalues or the singular values of `A` if available.
           If `d` is provided, `A` must be the eigenvectors, instead.

    Returns:
        The matrix `X`.
    '''
    if d is None:
        D, Q = torch.eig(A, eigenvectors=True)
        d = D[:, 0]
    else:
        Q = A
    C = Q.t().mm(B.mm(Q))
    Y = C / (d.view(-1, 1) + d.view(1, -1))
    return Q.mm(Y.mm(Q.t()))
