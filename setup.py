from setuptools import setup

setup(
    name='HoloNet_0510',
    version='',
    packages=['HoloNet', 'HoloNet.tools', 'HoloNet.plotting', 'HoloNet.predicting', 'HoloNet.preprocessing'],
    url='',
    license='',
    author='lihaochen',
    author_email='',
    description='',
    install_requires=[
        'numpy~=1.22.3',
        'torch~=1.10.0',
        'pandas~=1.3.4',
        'anndata~=0.7.6',
        'tqdm~=4.50.2',
        'scipy~=1.7.1',
        'scikit-learn~=0.23.2',
        'scanpy~=1.7.1',
        'networkx~=2.5',
        'matplotlib~=3.4.3',
        'seaborn~=0.11.0'
        ],
)
