import torch
from ..ops import (mul_diag, normalize_)


__all__ = ['RNG', 'from_eigen', 'definite', 'ring']


class RNG():
    '''Preserve the state of the random number generators of torch.

    Inspired by torch.random.fork_rng().

    Seeding random number generators (RNGs):
    - (PyTorch) torch.manual_seed(seed)
    - (Numpy) numpy.random.seed(seed)
    - (Python) random.seed(seed)

    Example:
    seed = 0
    torch.manual_seed(seed)
    with gnm.utils.rand.RNG(seed, devices=['cpu', 'cuda:0']):
        print(torch.rand(1, device='cpu').item())
        print(torch.rand(1, device='cuda:0').item())
        print(torch.rand(1, device='cuda:1').item())

    print('........')
    print(torch.rand(1, device='cpu').item())
    print(torch.rand(1, device='cuda:0').item())
    print(torch.rand(1, device='cuda:1').item())

    # Outputs
    # 0.49625658988952637
    # 0.08403993397951126
    # 0.08403993397951126
    # ........
    # 0.49625658988952637
    # 0.08403993397951126
    # 0.41885504126548767
    '''
    def __init__(self, seed=None, devices=None):
        '''Seeding the random number generator of torch devices.

        Args:
            seed: The intial seed value or list of values. If None, don't seed.
            devices: List of devices to seed. If None, seed all devices.
        '''
        if devices is None:
            num_gpus = torch.cuda.device_count()
            devices = ['cpu'] + [f'cuda:{i}' for i in range(num_gpus)]
        self.devices = [torch.device(d) for d in devices]

        def is_iterable(value):
            try:
                iter(value)
                return True
            except:
                return False
        self.seed = seed if is_iterable(seed) else [seed] * len(self.devices)

    def __enter__(self):
        self.states = [(torch.random.get_rng_state() if d.type == 'cpu'
                        else torch.cuda.random.get_rng_state(d.index))
                       for d in self.devices]
        for i, d in enumerate(self.devices):
            if self.seed[i] is None:
                continue
            if d.type == 'cpu':
                torch.default_generator.manual_seed(self.seed[i])
            else:
                with torch.cuda.random.device_ctx_manager(d.index):
                    torch.cuda.random.manual_seed(self.seed[i])

    def __exit__(self, exce_type, exce_value, traceback):
        for device, state in zip(self.devices, self.states):
            if device.type == 'cpu':
                torch.random.set_rng_state(state)
            else:
                torch.cuda.random.set_rng_state(state)


def from_eigen(eigen):
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


def definite(size, batch=None, norm=None, trace=None,
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
        trace: Trace of the output matrix. Will be ignored if `norm` is set.
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
    if norm is not None:
        if not torch.is_tensor(norm):
            norm = torch.tensor(norm, dtype=eigen.dtype, device=eigen.device)
        eigen *= norm / eigen.norm(dim=-1, keepdim=True)
    elif trace is not None:
        if not torch.is_tensor(trace):
            trace = torch.tensor(trace, dtype=eigen.dtype, device=eigen.device)
        eigen *= trace / eigen.sum(dim=-1, keepdim=True)
    return from_eigen(eigen)


def ring(size, sigma=1, tolerance=0, batch=None, dtype=None, device=None):
    '''Gaussian noise with `abs(L2_norm / sqrt(size) - sigma) <= tolerance`.

    Mean of squared L2 norm of Gaussian random vector = trace(covariance).

    We generate a standard independent Gaussian vector x with the given `size`
    with an L2 norm that is in the range [n - `tolerance`, n + `tolerance`],
    where `n = sigma * sqrt(size)`.

    Args:
        size: The size of the generated noise.
        sigma: The scalar input standard deviation.
        tolerance: The tolerance term (as described above).
        batch: The number of noises to generate.
        dtype: The data type.
        device: In which device.

    Returns:
        The generated noise.
    '''
    if batch is not None:
        size = [batch] + list(size)
    noise = torch.randn(size, dtype=dtype, device=device)
    flat = noise.view(1 if batch is None else batch, -1)
    norms = flat.norm(dim=1, keepdim=True)
    if tolerance != 0:
        sigma = torch.rand_like(norms) * (2 * tolerance) + (sigma - tolerance)
    flat.mul_(sigma * flat.size(1) ** 0.5 / norms)
    return noise
