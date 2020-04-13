from collections import defaultdict
from fractions import Fraction
from pathlib import Path

import torch
from tqdm import tqdm

from gaussian_relu_moments import forward_gaussian
from model_data import get_mnist, get_mnist_lenet, labeled_kmeans
from relu_linearize import relu_linearize
from stat_utils import gaussian, rand_matrix


def popular_vote(logits):
    y = logits.data.argmax(-1)
    return y.histc(y.shape[-1]).max().item() / y.shape[0]


def evaluate(model, images, points, trace, samples=1e4, terms=5):
    results = defaultdict(list)
    for mean, point in zip(tqdm(images), points):
        lin = relu_linearize(model, point)
        cov = rand_matrix(mean.numel(), trace=trace, device=mean.device)
        with torch.no_grad():
            logits = model(gaussian(cov, mean).draw(int(samples)))
            out_var, out_mean = torch.var_mean(logits, dim=0, unbiased=False)
            fg_var, fg_mean = forward_gaussian(lin, cov, mean.flatten(), terms)
            fg_bad = forward_gaussian(lin, cov, mean.flatten(), -1)[0]
        results['votes'].append(popular_vote(logits))
        results['out_mean'].append(out_mean)
        results['out_var'].append(out_var)
        results['fg_mean'].append(fg_mean)
        results['fg_var'].append(fg_var)
        results['fg_bad'].append(fg_bad)
    return results


def table_2(trials, trace, num_samples=1e4, terms=5):
    print(f'Table 2 [all images @ {Fraction(trace).limit_denominator()}]')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mnist = get_mnist()[1]  # testing set
    model = get_mnist_lenet(device=device).eval()
    images = mnist.data.float().div_(255).unsqueeze(1).to(device)
    if trials > 0:
        images = images[torch.randperm(len(images))[:trials]]
    return evaluate(model, images, images, trace, num_samples, terms)


def table_3(clusters, baseline, trace, num_samples=1e4, terms=5):
    name = str(clusters) + (' baseline' if baseline else '')
    print(f'Table 3 [{name} @ {Fraction(trace).limit_denominator()}]')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mnist = get_mnist()[1]  # testing set
    model = get_mnist_lenet(device=device).eval()
    images = mnist.data.float().div_(255).unsqueeze(1)
    centers, labels = labeled_kmeans(images, mnist.targets, clusters)
    if baseline:
        centers = torch.stack([
            cluster[distance.argmax()] for c in range(clusters)
            for cluster in [images[labels == c]]
            for distance in [(cluster - centers[c]).flatten(1).norm(dim=1)]
        ], 0)
    images, centers = images.to(device), centers.to(device)
    points = map(lambda i: centers[i], labels)
    return evaluate(model, images, points, trace, num_samples, terms)


def run_all(trace):
    path = Path(str(trace))
    path.mkdir(parents=True, exist_ok=True)
    if not (path / 't2.pt').exists():
        torch.save(table_2(0, trace), path / 't2.pt')
    if not (path / 't3_250_baseline.pt').exists():
        torch.save(table_3(250, True, trace), path / 't3_250_baseline.pt')
    if not (path / 't3_250.pt').exists():
        torch.save(table_3(250, False, trace), path / 't3_250.pt')
    if not (path / 't3_500_baseline.pt').exists():
        torch.save(table_3(500, True, trace), path / 't3_500_baseline.pt')
    if not (path / 't3_500.pt').exists():
        torch.save(table_3(500, False, trace), path / 't3_500.pt')
    if not (path / 't3_1000.pt').exists():
        torch.save(table_3(1000, False, trace), path / 't3_1000.pt')
    if not (path / 't3_2500.pt').exists():
        torch.save(table_3(2500, False, trace), path / 't3_2500.pt')
    if not (path / 't3_5000.pt').exists():
        torch.save(table_3(5000, False, trace), path / 't3_5000.pt')


if __name__ == '__main__':
    from argparse import ArgumentParser
    p = ArgumentParser(description='PAMI Tables Experiments')
    p.add_argument('-s', '--std', type=float, help='noise standard deviation')
    run_all(trace=784 * p.parse_args().std**2)
