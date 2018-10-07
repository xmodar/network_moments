from setuptools import setup, find_packages

# this package cloud benefit from namespace packages
# Link: https://packaging.python.org/guides/packaging-namespace-packages/
name = 'network_moments'
setup(
    name=name,
    version='0.9.5',
    description=(
        'A toolkit for computing some probabilistic moments '
        'of deep neural networks'
    ),
    url='https://github.com/ModarTensai/network_moments',
    author='Modar M. Alfadly',
    author_email='modar.alfadly@gmail.com',
    license='MIT',
    namespace_packages=[name],
    packages=[name] + [name + '.' + p for p in find_packages(name)],
    zip_safe=False
)
