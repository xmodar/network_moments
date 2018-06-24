# Network Moments

Network Moments is a toolkit that enables computing some probabilistic moments of deep neural networks given a specific input distribution. The current implemetation allows you to compute the first and second Gaussian network moments (GNM) of affine-ReLU-affine networks i.e., the output mean and variance subject to Gaussian input.

<img src="./static/theorems.svg" alt="theorems"/>

The main backend framework is [<img src="./static/pytorch-logo.png" alt="PyTorch" height="20px"/>](./network_moments/torch/) but also [<img src="./static/tensorflow-logo.png" alt="TensorFlow" height="20px"/>](./network_moments/tensorflow/) and [<img src="./static/matlab-logo.png" alt="MatLab" height="20px"/>](./matlab/) are supported.

### Requirements

Network Moments was developed and tested with the following:

 - [Python](https://www.python.org/) v3.6.3+
 - **Option 1**: [`PyTorch`](./network_moments/torch/)
   - [torch](https://pytorch.org/) v0.4.0+
   - (Optional) [torchvision](https://github.com/pytorch/vision) v0.2.1+
 - **Option 2**: [`TensorFlow`](./network_moments/tensorflow/)
   - [numpy](http://www.numpy.org) v1.14.2+
   - [tensorflow](https://www.tensorflow.org/) v1.8.0+
 - **Option 3**: [`MatLab`](./matlab/)
   - [matlab](https://www.mathworks.com/products/matlab.html) vR2017b+

You need [Jupyter](https://jupyter.org/) to run [tightness](./static/tightness.ipynb). It is recommended that you have [Jupyter Lab](https://github.com/jupyterlab/jupyterlab).

### Installation

After installing the requirements, clone this repository by running the following in the terminal:

```sh
cd /choose/some/permenant/path/for/the/package/
git clone https://github.com/ModarTensai/network_moments.git
pip install -e network_moments
```

Now go to the [tightness notebook](./static/tightness.ipynb) to see how to use this tool with the default backend framework.

To incorporate any new updates to the repository just pull the changes and they will be available for use automatically.

To uninstall the package:

```sh
pip uninstall network_moments
```

### Usage

To import the [`PyTorch`](./network_moments/torch/) sub-package:

```python
import network_moments.torch as nm
```

The basic usage is demonstrated in the [tightness notebook](./static/tightness.ipynb).

To import the [`TensorFlow`](./network_moments/tensorflow/) sub-package:

```python
import network_moments.tensorflow as nm
```
Please, refer to [tensorflow tests notebook](./static/tensorflow_tests.ipynb) for examples to compare [`PyTorch`](./network_moments/torch/) and [`TensorFlow`](./network_moments/tensorflow/) implementations.

### Cite

This is the official implementation of the method described in [this paper](http://openaccess.thecvf.com/content_cvpr_2018/html/Bibi_Analytic_Expressions_for_CVPR_2018_paper.html) (checkout the [poster](./static/poster.pdf)):

```bibtex
@InProceedings{Bibi_2018_CVPR,
    author = {Bibi, Adel and Alfadly, Modar and Ghanem, Bernard},
    title = {Analytic Expressions for Probabilistic Moments of PL-DNN With Gaussian Input},
    booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2018}
}
```

### License

MIT

### Author

[Modar M. Alfadly](https://github.com/ModarTensai/network_moments/)

### Contributors

I would gladly accept any pull request that improves any aspect of this repository.