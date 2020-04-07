from fractions import Fraction
from random import choice

import matplotlib.pyplot as plt
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from torchvision.utils import make_grid
from tqdm import tqdm

from gaussian_relu_moments import forward_gaussian
from model_data import get_mnist, get_mnist_lenet
from relu_linearize import relu_linearize
from stat_utils import VarianceMeter, gaussian, rand_matrix


def popular_vote(logits):
    y = logits.data.argmax(-1)
    return y.histc(y.shape[-1]).max().item() / y.shape[0]


def plot_samples(samples, row_length=3, normalize=True, show=True, ax=None):
    samples = samples.data.cpu()
    samples = make_grid(samples, row_length, 2, normalize)
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(samples.clamp_(0, 1).permute(1, 2, 0))
    ax.axis('off')
    if show:
        plt.show(fig)
    return ax


def error(a, b, rtol=1e-5, atol=1e-8):
    return ((a - b).abs() - (atol + rtol * b.abs())).clamp_(0)


def gist(mean_ratio, bad_ratio, var_ratio, votes):
    mr = (mean_ratio.mean.tolist(), mean_ratio.std.tolist())
    print('mean ratios')
    print([round(x, 4) for x in mr[0]])
    print([round(x, 4) for x in mr[1]])
    print()
    br = (bad_ratio.mean.tolist(), bad_ratio.std.tolist())
    print('bad var ratios')
    print([round(x, 4) for x in br[0]])
    print([round(x, 4) for x in br[1]])
    print()
    vr = (var_ratio.mean.tolist(), var_ratio.std.tolist())
    print('good var ratios')
    print([round(x, 4) for x in vr[0]])
    print([round(x, 4) for x in vr[1]])
    print()
    vt = (votes.mean, votes.std)
    print('votes')
    print(f'{vt[0] * 100:.2f}%')
    print(f'{vt[1] * 100:.2f}%')
    print()
    return dict(mean_ratio=mr, bad_ratio=br, var_ratio=br, votes=vt)


def table_2(trials, trace, num_samples=1e4, terms=5):
    print(f'Table 2 [all images @ {Fraction(trace).limit_denominator()}]')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mnist = get_mnist()[1]  # testing set
    model = get_mnist_lenet(device=device).eval()
    votes = VarianceMeter(unbiased=True)
    bad_ratio = VarianceMeter(unbiased=True)
    var_ratio = VarianceMeter(unbiased=True)
    mean_ratio = VarianceMeter(unbiased=True)
    if trials < 1:
        data = (x.to(device) for x, _ in tqdm(mnist))
    else:
        data = (choice(mnist)[0].to(device) for _ in tqdm(range(trials)))
    for mean in data:
        cov = rand_matrix(mean.numel(), trace=trace, device=device)
        dist = gaussian(cov, mean)
        lin = relu_linearize(model, mean)
        with torch.no_grad():
            logits = model(dist.draw(int(num_samples)))
            m_var, m_mean = torch.var_mean(logits, dim=0, unbiased=False)
            l_var, l_mean = forward_gaussian(lin, cov, mean.view(-1), terms)
            b_var, _ = forward_gaussian(lin, cov, mean.view(-1), -1)
            vote = popular_vote(logits)
        votes.update(vote)
        bad_ratio.update(b_var / m_var)
        var_ratio.update(l_var / m_var)
        mean_ratio.update(l_mean / m_mean)
    return mean_ratio, bad_ratio, var_ratio, votes


def table_3(clusters, baseline, trace, num_samples=1e4, terms=5):
    name = str(clusters) + (' baseline' if baseline else '')
    print(f'Table 3 [{name} @ {Fraction(trace).limit_denominator()}]')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_mnist_lenet(device=device).eval()
    print('MNIST', end='... ')
    mnist = torch.stack([x for x, _ in get_mnist()[1]], dim=0)  # testing set
    print('PCA', end='... ')
    pca = PCA(n_components=40)
    pca_mnist = pca.fit_transform(mnist.flatten(1))
    print('KMeans', end='... ')
    kmeans = KMeans(clusters, random_state=42).fit(pca_mnist)
    print('Done!')
    centers = torch.from_numpy(pca.inverse_transform(kmeans.cluster_centers_))
    centers = centers.to(mnist.dtype).view(-1, *mnist.shape[1:])
    if baseline:
        centers = torch.stack([
            images[distance.argmax()] for c in range(clusters)
            for images in [mnist[kmeans.labels_ == c]]
            for distance in [(images - centers[c]).flatten(1).norm(dim=1)]
        ], 0)
    mnist, centers = mnist.to(device), centers.to(device)
    votes = VarianceMeter(unbiased=True)
    bad_ratio = VarianceMeter(unbiased=True)
    var_ratio = VarianceMeter(unbiased=True)
    mean_ratio = VarianceMeter(unbiased=True)
    for c, mean in zip(tqdm(kmeans.labels_), mnist):
        cov = rand_matrix(mean.numel(), trace=trace, device=device)
        dist = gaussian(cov, mean)
        lin = relu_linearize(model, centers[c])
        with torch.no_grad():
            logits = model(dist.draw(int(num_samples)))
            m_var, m_mean = torch.var_mean(logits, dim=0, unbiased=False)
            l_var, l_mean = forward_gaussian(lin, cov, mean.view(-1), terms)
            b_var, _ = forward_gaussian(lin, cov, mean.view(-1), -1)
            vote = popular_vote(logits)
        votes.update(vote)
        bad_ratio.update(b_var / m_var)
        var_ratio.update(l_var / m_var)
        mean_ratio.update(l_mean / m_mean)
    return mean_ratio, bad_ratio, var_ratio, votes


def run_all(trace):
    results = {}
    results['t2'] = gist(*table_2(0, trace))
    results['t3_250_baseline'] = gist(*table_3(250, True, trace))
    results['t3_250'] = gist(*table_3(250, False, trace))
    results['t3_500_baseline'] = gist(*table_3(500, True, trace))
    results['t3_500'] = gist(*table_3(500, False, trace))
    results['t3_1000'] = gist(*table_3(1000, False, trace))
    results['t3_2500'] = gist(*table_3(2500, False, trace))
    results['t3_5000'] = gist(*table_3(5000, False, trace))
    return results


if __name__ == '__main__':
    from argparse import ArgumentParser
    p = ArgumentParser(description='PAMI Tables Experiments')
    p.add_argument('-s', '--std', type=float, help='noise standard deviation')
    run_all(trace=784 * p.parse_args().std**2)
