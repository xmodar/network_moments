from collections import defaultdict
from fractions import Fraction
from pathlib import Path
from pprint import pprint

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from gaussian_relu_moments import forward_gaussian
from model_data import get_mnist, get_mnist_lenet, labeled_kmeans
from relu_linearize import relu_linearize
from stat_utils import gaussian, rand_matrix


def popular_vote(logits):
    y = logits.data.argmax(-1)
    return y.histc(y.shape[-1]).max().item() / y.shape[0]


def remove_extremes(x, k):
    lower = x.kthvalue(k, dim=0).values
    upper = x.kthvalue(x.shape[0] - k, dim=0).values
    mask = (x >= lower).all(1) & (x <= upper).all(1)
    return x[mask]


def ratio_std_mean(x, y):
    """Compute the std and mean of shifted ratios."""
    # relative_error = remove_extremes(x / y, 10)
    relative_error = 2 * (x - y).abs() / (x.abs() + y.abs())
    # See also: https://stats.stackexchange.com/a/201864
    return torch.std_mean(relative_error, dim=0)


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


def get_model_data(subset=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mnist = get_mnist()[1]  # testing set
    model = get_mnist_lenet(device=device).eval()
    images = mnist.data.float().div_(255).unsqueeze(1)
    return model, images[:subset].to(device), mnist.targets[:subset].to(device)


def table_2(trials, trace, num_samples=1e4, terms=5, subset=None):
    print(f'Table 2 [images @ {Fraction(trace).limit_denominator()}]')
    model, images, _ = get_model_data(subset)
    if trials > 0:
        images = images[torch.randperm(len(images))[:trials]]
    return evaluate(model, images, images, trace, num_samples, terms)


def table_3(clusters, baseline, trace, num_samples=1e4, terms=5, subset=None):
    name = str(clusters) + (' baseline' if baseline else '')
    print(f'Table 3 [{name} @ {Fraction(trace).limit_denominator()}]')
    model, images, targets = get_model_data(subset)
    centers, labels = labeled_kmeans(images.cpu(), targets.cpu(), clusters)
    if baseline:
        centers = torch.stack([
            cluster[distance.argmax()] for c in range(clusters)
            for cluster in [images[labels == c].cpu()]
            for distance in [(cluster - centers[c]).flatten(1).norm(dim=1)]
        ], 0)
    points = map(lambda i: centers[i].to(images.device), labels)
    return evaluate(model, images, points, trace, num_samples, terms)


def summarize(result_file):
    out = {}
    data = torch.load(result_file, 'cpu')
    out_mean = torch.stack(data['out_mean'])
    out_var = torch.stack(data['out_var'])
    fg_mean = torch.stack(data['fg_mean'])
    fg_var = torch.stack(data['fg_var'])
    fg_bad = torch.stack(data['fg_bad'])

    name = Path(result_file).stem
    k = len(fg_mean) if name == 't2' else int(name.split('_')[1])
    baseline = name.endswith('baseline')
    std = float(Path(result_file).absolute().parent.name)
    out = {'k': k, 'baseline': baseline, 'std': std}
    as_dict = lambda s, m: {'mean': m.tolist(), 'std': s.tolist()}
    out['mean_ratios'] = as_dict(*ratio_std_mean(fg_mean, out_mean))
    out['bad_ratios'] = as_dict(*ratio_std_mean(fg_bad, out_var))
    out['var_ratios'] = as_dict(*ratio_std_mean(fg_var, out_var))
    return out


def plot_hist(data, bins=50, columns=5, legend=False, figsize=(17, 5)):
    rows = -(-len(data) // columns)
    fig, axes = plt.subplots(rows, columns, figsize=figsize)
    for i, (ax, x) in enumerate(zip(axes.flatten(), data)):
        mn, mx = x.min(), x.max()
        ax.hist(x, bins, label=f'[{mn:.3f}, {mx:.3f}]')
        std, mean = torch.std_mean(x)
        ax.set_title(f'{i}: {mean:.3f}$\\pm${std:.3f}')
        if legend:
            ax.legend()
    list(map(fig.delaxes, axes.flatten()[len(data):]))
    fig.subplots_adjust(hspace=0.5, wspace=0.3)
    fig.tight_layout()
    # if legend:
    #     legend = fig.legend(loc='upper left', bbox_to_anchor=(1, 1))
    #     fig.get_tightbbox(fig.canvas.get_renderer(), [legend])
    plt.show(fig)


def run_all(path, typical_std, subset):
    trace = 784 * typical_std**2
    path = Path(path) / str(typical_std)
    path.mkdir(parents=True, exist_ok=True)

    def run(out_file, test, *args, **kwargs):
        if not Path(out_file).exists():
            torch.save(test(*args, trace, subset=subset, **kwargs), out_file)
        pprint(summarize(out_file))

    run(path / 't2.pt', table_2, 0)
    run(path / 't3_250_baseline.pt', table_3, 250, True)
    run(path / 't3_250.pt', table_3, 250, False)
    run(path / 't3_500_baseline.pt', table_3, 500, True)
    run(path / 't3_500.pt', table_3, 500, False)
    run(path / 't3_1000.pt', table_3, 1000, False)
    run(path / 't3_2500.pt', table_3, 2500, False)
    run(path / 't3_5000.pt', table_3, 5000, False)


if __name__ == '__main__':
    from argparse import ArgumentParser
    p = ArgumentParser(description='PAMI Tables Experiments')
    p.add_argument('-p', '--path', default='./results', help='output path')
    p.add_argument('-s', '--std', type=float, help='noise standard deviation')
    p.add_argument('-t', '--test', type=int, help='number of images to test')
    arg = p.parse_args()
    run_all(arg.path, arg.std, arg.test)
