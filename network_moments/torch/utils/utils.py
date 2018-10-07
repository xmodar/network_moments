import torch
import importlib


__all__ = ['verbosify', 'epsilon', 'even_zip', 'special_sylvester']


def epsilon(dtype=None, device=None):
    '''Machine epsilon for a specific torch.dtype.'''
    if not hasattr(epsilon, 'cache'):
        epsilon.cache = {}
    if dtype in epsilon.cache:
        return epsilon.cache[dtype, device]
    value = one = torch.ones([], dtype=dtype, device=device)
    while one + value != one:
        machine_epsilon = value
        value = value >> 1
    epsilon.cache[dtype, device] = machine_epsilon
    return machine_epsilon


def even_zip(*lists):
    '''Similar to zip() but prioritizes the longest iterable.'''
    iterators = [iter(el) for el in lists]
    feed = [None] * len(lists)
    while True:
        done = True
        for i, itr in enumerate(iterators):
            try:
                feed[i] = next(itr)
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


def _verbosify(iterable):
    # shows only the iteration number and how many iterations are left
    len_iterable = len(iterable) if hasattr(iterable, '__len__') else None
    for i, element in enumerate(iterable, 1):
        if len_iterable is None:
            print('\rIteration #{}'.format(i), end='')
        else:
            print('\rIteration #{} out of {} iterations [Done {:.2f}%]'.format(
                i, len_iterable, 100 * i / len_iterable), end='')
        yield element
    print('\r', end='', flush=True)


def verbosify(iterable, leave=False, file=None, **kwargs):
    '''Utility function to print the progress of a for-loop.

    It is recommended that you install `tqdm` package.
    It will look for this package and try to use it.
    Otherwise, it will do a simplistic non-efficient implementation.

    Args:
        iterable: Of the for-loop (e.g., range(3)).
        leave: Whether to leave the progress print out after finishing.
        file: To which stream we will print the progress.
        kwargs: It will be passed to `tqdm.tqdm()` if available.

    Returns:
        Iterator over `iterable`.
    '''
    # try to use tqdm (shows the speed and the remaining time left)
    if importlib.util.find_spec('tqdm') is not None:
        tqdm = importlib.import_module('tqdm').tqdm
        if file is None:
            file = importlib.import_module('sys').stdout
        return tqdm(iterable, leave=leave, file=file, **kwargs)
    return iter(_verbosify(iterable))
