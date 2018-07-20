#from distutils.core import setup
from setuptools import setup

setup(
    name='GraphProt',
    description='Graph Neural network Scoring of protein-protein conformations',
    version='0.0',
    url='https://github.com/DeepRank/GraphProt',
    packages=['iScore'],


    install_requires=[
        'numpy >= 1.13',
        'scipy'],
        #'tarfiles',
        #'pickle'],

    extras_require= {
        'test': ['nose', 'coverage', 'pytest',
                 'pytest-cov','codacy-coverage','coveralls'],
    }
)
