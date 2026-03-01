#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

long_description = """
PyTorch implementation of DPWTE (Deep Learning Approach to Survival Analysis Using a Parsimonious Mixture of Weibull Distributions).

This code was developed during Achraf Bennis' PhD on survival analysis and implements the DPWTE approach described in:
- Bennis et al., "DPWTE: A Deep Learning Approach to Survival Analysis Using a Parsimonious Mixture of Weibull Distributions", ICANN 2021.
"""

requirements = [
    'torch>=1.7,<2.0',
    'lifelines==0.21.0',
    'scipy==1.3.0',
    'scikit-learn==0.21.2',
    'numpy==1.16.4',
    'progressbar2==3.50.1',
    'pandas==1.1.1',
]

setup(
    name='pydpwte',
    version='0.0.1',
    description="Survival analysis using deep learning and Weibull mixtures (DPWTE).",
    long_description=long_description,
    author="Achraf Bennis",
    author_email='achraf.bennis.b@gmail.com',
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    install_requires=requirements,
    classifiers=[
        'Natural Language :: English',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.6',
    ],
    python_requires='>=3.6'
)
