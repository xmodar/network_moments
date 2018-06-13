# Network Moments

Network Moments is a toolkit that enables computing some probabilistic moments of deep neural networks given a specific input distribution. The current implemetation allows you to compute the first and second Gaussian network moments (GNM) of affine-ReLU-affine networks i.e., the output mean and variance subject to Gaussian input.

### Requirements

Network Moments was developed and tested with the following:

 - [Python](https://www.python.org/) v3.6.3+
 - [torch](https://pytorch.org/) v0.4.0+
 - (Optional) [torchvision](https://github.com/pytorch/vision) v0.2.1+

You need [Jupyter](https://jupyter.org/) to run [tightness](./tightness.ipynb). It is recommended that you have [Jupyter Lab](https://github.com/jupyterlab/jupyterlab).

### Installation

After cloning this repository run the following in the terminal:

```sh
cd /choose/some/permenant/path/for/the/package/
git clone https://github.com/ModarTensai/network_moments.git
pip install -e network_moments
```

Now go to the [tightness notebook](./tightness.ipynb) to see how to use this tool.

To uninstall the package:

```sh
pip uninstall network_moments
```

### Cite

This is the official implementation of the method described in this paper.

```bibtex
@inproceedings{bibi2018analytic,
  title={Analytic Expressions for Probabilistic Moments of PL-DNN with Gaussian Input},
  author={Bibi, Adel and Alfadly, Modar and Ghanem, Bernard},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={9099--9107},
  year={2018}
}
```

### License

MIT

### Author

[Modar M. Alfadly](https://github.com/ModarTensai/network_moments/)