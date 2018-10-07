import math
import torch


__all__ = ['cov', 'skewness_score', 'kurtosis_score', 'percentile',
           'inter_quartile_range', 'num_hist_bins', 'integrate_area',
           'normalized_histogram', 'hist_as_func', 'trapz',
           'estimate_pdf', 'pdf_similarity']


def _indexer_into(x, dim=0, keepdim=False):
    '''indexes into x along dim.'''
    def indexer(i):
        # (e.g., x[:, 2, :] is indexer(2) if dim == 1)
        out = x[[slice(None, None)] * dim + [i, ...]]
        return out.unsqueeze(dim) if keepdim and x.dim() != out.dim() else out
    return indexer


def _shifted_slices(x, dim=-1):
    '''return x[..., 1:, ...], x[..., :-1, ...].'''
    slice1 = [slice(None)] * x.dim()
    slice2 = [slice(None)] * x.dim()
    slice1[dim] = slice(1, None)
    slice2[dim] = slice(None, -1)
    return x[slice1], x[slice2]


def trapz(y, x=None, dx=1, dim=-1):
    '''Integrate along the given axis using the composite trapezoidal rule.

    Equivalent to numpy.trapz().

    Args:
        y: Input tensor to integrate.
        x: The sample points corresponding to the `y` values.
            If it is None, the sample points are assumed to be evenly
            spaced `dx` apart.
        dx: The spacing between the sample points when `x` is None.
        dim: The dimension along which to integrate.

    Returns:
        Definite integral as apporximated by trapezodial rule.
    '''
    if x is None:
        d = dx
    elif x.dim() == 1:
        d = x[1:] - x[:-1]
        shape = [1] * y.dim()
        shape[dim] = d.size(0)
        d = d.view(shape)
    else:
        d1, d2 = _shifted_slices(x, dim)
        d = d1 - d2
    y1, y2 = _shifted_slices(y, dim)
    return (d * (y1 + y2) / 2).sum(dim)


def percentile(x, q, dim=0, keepdim=False, sort=True):
    '''The percentile of data samples.

    Args:
        x: Data tensor.
        q: The percentage in [0, 100].
        dim: The dimension along which we perform the operation.
        keepdim: Whether to keep the dimension of operation.
        sort: Set to False only if x is sorted to save computation.

    Returns:
        The `q`th percentile of the data.
    '''
    if sort:
        x = x.data.sort(dim)[0]
    if not 0 <= q <= 100:
        raise ValueError('`q` must be in between 0 and 100.')
    value = (q / 100) * x.size(dim) - 1
    if value < 0:
        return 0
    index = int(value)
    frac = value - index
    next_index = min(index + 1, x.size(dim) - 1)
    X = _indexer_into(x.data, dim, keepdim)
    return (1 - frac) * X(index) + frac * X(next_index)


def inter_quartile_range(x, dim=0, keepdim=False, sort=True):
    '''IQR: the inter-quartile range of samples of data.

    Args:
        x: Data tensor.
        dim: The dimension along which we perform the operation.
        keepdim: Whether to keep the dimension of operation.
        sort: Set to False only if x is sorted to save computation.

    Returns:
        The IQR of the data.
    '''
    if sort:
        x = x.data.sort(dim)[0]
    q1 = percentile(x, 25, dim=dim, keepdim=keepdim, sort=False)
    q3 = percentile(x, 75, dim=dim, keepdim=keepdim, sort=False)
    return q3 - q1


def num_hist_bins(x, min_bins=1, max_bins=1e6,
                  dim=0, keepdim=False, sort=True):
    '''Freedman-Diaconis rule for the number of bins in a histogram.

    Args:
        x: Data tensor.
        dim: The dimension along which we perform the operation.
        keepdim: Whether to keep the dimension of operation.
        min_bins: The minimum number of bins.
        max_bins: The maximum number of bins.
        sort: Set to False only if x is sorted to save computation.

    Returns:
        The appropriate number of bins to histogram the given data.
    '''
    if sort:
        x = x.data.sort(dim)[0]
    long = lambda v: torch.tensor(v, dtype=torch.long, device=x.device)
    iqr = inter_quartile_range(x, dim=dim, keepdim=keepdim, sort=False)
    bin_width = 2 * x.size(dim)**(-1 / 3) * iqr
    X = _indexer_into(x.data, dim, keepdim)
    num_bins = ((X(-1) - X(0)) / bin_width).round().long()
    return num_bins.clamp(min=long(min_bins), max=long(max_bins))


def integrate_area(values, bounds=None, dx=1, dim=0, keepdim=False, _sum=None):
    '''Integrate using the composite trapezoidal rule.

    Args:
        values: Data tensor of y-axis values.
        bounds: A tuple (lower, upper) of the range of the x-axis.
        dx: The distance between `values` if `bounds is None`.
        dim: The dimension along which we perform the operation.
        keepdim: Whether to keep the dimension of operation.
        _sum: The pre-computed sum of `values`.

    Returns:
        The area under the curve using the compoite trapezoidal rule.
    '''
    if _sum is None:
        _sum = values.sum(dim)
    v = _indexer_into(values, dim, keepdim)
    if bounds is None:
        return dx * (_sum - (v(0) + v(-1)) / 2)
    else:
        return _sum * (bounds[1] - bounds[0]) / (values.size(dim) - 1)


def normalized_histogram(x, bins=None, sort=True):
    '''Estimates the PDF of samples of data as a histogram.

    In case that x was constant

    Args:
        x: The 1D data tensor.
        bins: The desired number of bins. If None, use num_hist_bins(x).
        sort: Set to False only if x is sorted to save computation.

    Returns:
        The normalized histogram of the data as 1D tensor and
        the range bounds (lower, upper).
    '''
    if sort:
        x = x.sort()[0]
    if bins is None:
        bins = num_hist_bins(x, sort=False).item()
    hist = x.histc(bins)
    bounds = (x[0], x[-1])
    area = integrate_area(hist, bounds, _sum=x.numel())
    if x[0] == x[-1]:
        area = hist.clone()
        area[area == 0] = 1
    return hist / area, bounds


def hist_as_func(hist, bounds, filler=float('nan')):
    '''Converts a discrete histogram into a continuous function.

    One can discretize a function f in a domain `bounds = [lower, upper]`
    using a histogram hist of n bins, i.e. n functions values at uniformly
    distributed samples in the range [lower, upper].
    In other words, we can say that f(x) = hist[index]
    where index = (x - lower) * (n - 1) / (upper - lower)
    assuming that index is an integer in the range [lower, upper].
    If index is not in the range, the default value is `filler`.
    If index is not integer, we linearly interpolate the adjacent elements.

    Args:
        hist: A sorted list of values that represent a histogram.
        bounds: A tuple (lower, upper) of the range of the x-axis.
        filler: The default value for out of range samples.

    Returns:
        A function `func` that evaluates hist[x] at any given x.
        If x < lower or x > upper, the default hist value will be `filler`.
        This function will have all the parameters {hist, lower, upper, filler}
        as attributes, e.g. `func.hist is hist` and so on.
    '''
    lower, upper = bounds
    if upper == lower:
        hist_sum = hist.sum()
    else:
        factor = (hist.size(0) - 1) / (upper - lower)

    def func(x):
        out = torch.empty_like(x)
        in_bound = (lower <= x) * (x <= upper)
        out[~in_bound] = filler
        if upper == lower:
            out[in_bound] = hist_sum
        else:
            pos = (x[in_bound] - lower) * factor
            index, frac = pos.long(), pos % 1
            next_index = (index + 1).clamp(max=hist.size(0) - 1)
            out[in_bound] = (1 - frac) * hist[index] + frac * hist[next_index]
        return out
    func.hist = hist
    func.bounds = bounds
    func.filler = filler
    return func


def estimate_pdf(x, sort=True):
    '''Fit the PDF of samples of data.

    Args:
        x: A 1D data tensor.
        sort: Set to False only if x is sorted to save computation.

    Returns:
        The PDF function (output of `hist_as_func()`).
    '''
    hist, bounds = normalized_histogram(x.data, sort=sort)
    return hist_as_func(hist, bounds, filler=0)


def pdf_similarity(hist1, hist2, bounds):
    '''Similarity between to PDFs using histogram kernel intersection.

    Args:
        hist1: The first PDF as a normalized histogram (1D tensor).
        hist2: The second histogram.
        bounds: A tuple (lower, upper) range of the x-axis.

    Returns:
        The similarity measure [0, 1].
    '''
    if hist1.numel() != hist2.numel():
        raise ValueError('The two PDFs should have the same length.')
    return integrate_area(torch.min(hist1, hist2), bounds)


def cov(m, rowvar=False):
    '''Estimate a covariance matrix given data.

    Covariance indicates the level to which two variables vary together.
    If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
    then the covariance matrix element `C_{ij}` is the covariance of
    `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.

    Args:
        m: A 1-D or 2-D array containing multiple variables and observations.
            Each column of `m` represents a variable, and each row a single
            observation of all those variables.
        rowvar: If `rowvar` is True, then each row represents a
            variable, with observations in the columns. Otherwise, the
            relationship is transposed: each column represents a variable,
            while the rows contain observations.

    Returns:
        The covariance matrix of the variables.
    '''
    if m.dim() > 2:
        raise ValueError('m has more than 2 dimensions')
    if m.dim() < 2:
        m = m.view(1, -1)
    if not rowvar and m.size(0) != 1:
        m = m.t()
    m = m - torch.mean(m, dim=1, keepdim=True)
    mt = m.t()  # if complex: mt = m.t().conj()
    fact = 1.0 / (m.size(1) - 1)
    return fact * m.matmul(mt).squeeze()


def _x_n_dim(x, dim=None):
    '''Helper for skewness_score and kurtosis_score.'''
    x = x.data
    if dim is None:
        dim = 0
        n = x.numel()
        x = x.view(-1) - x.mean()
    else:
        n = x.shape[dim]
        x = x - x.mean(dim, keepdim=True)
    return x, n, dim


def skewness_score(x, dim=0):
    '''Test whether the skew is different from the normal distribution.

    This function tests the null hypothesis that the skewness of
    the population that the sample was drawn from is the same
    as that of a corresponding normal distribution.
    ripoff from: `scipy.stats.skewtest`.

    Args:
        a: Array of the sample data
        axis: Axis along which to compute test. Default is 0. If None,
           compute over the whole array `a`.
    Returns:
        statistic: The computed z-score for this test.
        p-value: A 2-sided chi squared probability for the hypothesis test.
    '''
    x, n, dim = _x_n_dim(x, dim)
    b2 = (x**3).mean(dim) / (x**2).mean(dim)**1.5
    y = b2 * math.sqrt(((n + 1) * (n + 3)) / (6.0 * (n - 2)))
    beta2 = 3.0 * (n**2 + 27 * n - 70) * (n + 1) * (n + 3) /\
        ((n - 2.0) * (n + 5) * (n + 7) * (n + 9))
    W2 = -1.0 + math.sqrt(2 * (beta2 - 1))
    delta = 1.0 / math.sqrt(0.5 * math.log(W2))
    alpha = math.sqrt(2.0 / (W2 - 1))
    y[y == 0] = 1
    yalpha = y / alpha
    Z = delta * torch.log(yalpha + torch.sqrt(yalpha**2 + 1))
    return Z, 1 + torch.erf(-math.sqrt(0.5) * torch.abs(Z))


def kurtosis_score(x, dim=0):
    '''Test whether a dataset has normal kurtosis.

    This function tests the null hypothesis that the kurtosis
    of the population from which the sample was drawn is that
    of the normal distribution: ``kurtosis = 3(n-1)/(n+1)``.
    ripoff from: `scipy.stats.kurtosistest`.

    Args:
        a: Array of the sample data
        axis: Axis along which to compute test. Default is 0. If None,
           compute over the whole array `a`.
    Returns:
        statistic: The computed z-score for this test.
        p-value: A 2-sided chi squared probability for the hypothesis test.
    '''
    x, n, dim = _x_n_dim(x, dim)
    if n < 20:
        raise ValueError(
            "Number of elements has to be >= 20 to compute kurtosis")
    b2 = (x**4).mean(dim) / (x**2).mean(dim)**2
    E = 3.0 * (n - 1) / (n + 1)
    varb2 = 24.0 * n * (n - 2) * (n - 3) / ((n + 1)**2 * (n + 3) * (n + 5))
    x = (b2 - E) / math.sqrt(varb2)
    sqrtbeta1 = 6.0 * (n * n - 5 * n + 2) / ((n + 7) * (n + 9)) *\
        math.sqrt((6.0 * (n + 3) * (n + 5)) / (n * (n - 2) * (n - 3)))
    A = 6.0 + 8.0 / sqrtbeta1 * \
        (2.0 / sqrtbeta1 + math.sqrt(1 + 4.0 / (sqrtbeta1**2)))
    term1 = 1 - 2 / (9.0 * A)
    denom = 1 + x * math.sqrt(2 / (A - 4.0))
    term2 = torch.sign(denom) * torch.pow((1 - 2.0 / A) /
                                          torch.abs(denom), 1 / 3.0)
    Z = (term1 - term2) / math.sqrt(2 / (9.0 * A))
    return Z, 1 + torch.erf(-math.sqrt(0.5) * torch.abs(Z))
