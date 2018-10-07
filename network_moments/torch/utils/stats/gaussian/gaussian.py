import math
import torch
from ..stats import (skewness_score, kurtosis_score, integrate_area,
                     estimate_pdf, pdf_similarity)


__all__ = ['density', 'normal_density', 'hypothesis_test',
           'fit', 'gaussianity']


def density(x, mean, std, pdf=True, cdf=True):
    '''The PDF and CDF of the Gaussian distribution.

    Args:
        x: The point of evaluation (tensor).
        mean: The mean of the distribution.
        std: The standard deviation.
        pdf: Whether to return the PDF.
        cdf: Whether to return the CDF.

    Returns:
        The Gaussian (PDF, CDF) at x.
    '''
    arg = math.sqrt(0.5) * (x - mean) / std
    if pdf:
        coef = 1.0 / (math.sqrt(2.0 * math.pi) * std)
        rpdf = coef * torch.exp(-torch.pow(arg, 2.0))
        if not cdf:
            return rpdf
    if cdf:
        rcdf = 0.5 + 0.5 * torch.erf(arg)
        if not pdf:
            return rcdf
    return rpdf, rcdf


def normal_density(x, pdf=True, cdf=True):
    '''The PDF and CDF of the normal distribution.

    Equivalent but a bit more efficient than `density(x, 0, 1, pdf, cdf)`.

    Args:
        x: The point of evaluation (tensor).
        pdf: Whether to return the PDF.
        cdf: Whether to return the CDF.

    Returns:
        The normal (PDF, CDF) at x.
    '''
    arg = math.sqrt(0.5) * x
    if pdf:
        coef = 1.0 / math.sqrt(2.0 * math.pi)
        rpdf = coef * torch.exp(-torch.pow(arg, 2.0))
        if not cdf:
            return rpdf
    if cdf:
        rcdf = 0.5 + 0.5 * torch.erf(arg)
        if not pdf:
            return rcdf
    return rpdf, rcdf


def hypothesis_test(x, dim=0):
    '''Test whether a sample differs from a normal distribution.

    This function tests the null hypothesis that a sample comes
    from a normal distribution.  It is based on D'Agostino and
    Pearson's test that combines skew and kurtosis to
    produce an omnibus test of normality.
    ripoff from: `scipy.stats.normaltest`.

    Args:
        a: The array containing the sample to be tested.
        dim: Axis along which to compute test. Default is 0. If None,
            compute over the whole array `a`.

    Returns:
        statistic: ``s^2 + k^2``, where ``s`` is the z-score returned by
            `skewness_score` and ``k`` is the z-score from `kurtosis_score`.
        p-value: A 2-sided chi squared probability for the hypothesis test.
    '''
    # https://machinelearningmastery.com/a-gentle-introduction-to-normality-tests-in-python/
    k2 = skewness_score(x, dim)[0]**2 + kurtosis_score(x, dim)[0]**2
    return k2, torch.exp(-k2 / 2.0)


def fit(x, sigmas=5, correct=True, sort=True):
    '''Estimate the PDF from samples and fit a Gaussian.

    Args:
        x: A 1D data tensor.
        sigmas: How many standard deviations away from the mean to consider.
        correct: Whether to normalize `pdf_fit` to account for sampling errors.
        sort: Set to False only if x is sorted to save computation.

    Returns:
        {xs: A linspace for the fitted PDFs
        pdf: The PDF of the data as 1D histogram
        fit: The histogram of the best Guassian fit
        mean: The mean of the Gaussian fit
        std: The standard deviation of the Gaussian fit
        similarity: The simialrity between `pdf` and `fit`}
    '''
    std = x.data.std()
    mean = x.data.mean()
    pdf = estimate_pdf(x.data, sort=sort)
    lower = min(pdf.bounds[0], mean - sigmas * std)
    upper = max(pdf.bounds[1], mean + sigmas * std)
    bounds = (lower, upper)
    xs = torch.linspace(*bounds, max(pdf.hist.size(0), 100),
                        device=x.device, dtype=x.dtype)
    pdf_hist = pdf(xs)
    gaussian_fit = density(xs, mean, std, cdf=False)
    if correct:  # normalize to correct sampling errors
        pdf_hist /= integrate_area(pdf_hist, bounds)
        gaussian_fit /= integrate_area(gaussian_fit, bounds)
    similarity = pdf_similarity(pdf_hist, gaussian_fit, bounds)
    if lower == upper:  # handle the degenerate case
        pdf_hist[0] = gaussian_fit[0] = similarity = 1
    results = {
        'xs': xs,
        'pdf': pdf_hist,
        'fit': gaussian_fit,
        'mean': mean,
        'std': std,
        'similarity': similarity,
    }
    return results


def gaussianity(x, dim=0, std_threshold=0, keepdim=False):
    '''Computes the closeness of data samples to Gaussian distribution.

    Args:
        x: The tensor containing the sample to be tested.
        dim: Dimension along which to do the operation.
        std_threshold: If the std of x is less than the threshold, output 0.
        keepdim: Whether to keep the dimension of operation.

    Returns:
        A tensor of similarity values between 0 to 1 (Gaussian).
    '''
    X = x.data.transpose(0, dim).view(x.size(dim), -1)

    def _fit(y):
        gf = fit(y)
        if gf['std'] < std_threshold:
            return 0
        return gf['similarity']
    out = torch.tensor(tuple(_fit(X[:, i]) for i in range(X.size(1))),
                       dtype=x.dtype, device=x.device)
    size = x.size()[:dim] + ((1,) if keepdim else ()) + x.size()[dim + 1:]
    return out.view(size)
