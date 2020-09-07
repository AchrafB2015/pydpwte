#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

long_description = """
"""

requirements = ['setuptools==49.2.0',
                'pytorch',
                'lifelines==0.21.0',
                'scipy==1.3.0',
                'scikit-learn==0.21.2',
                'numpy==1.16.4',
                'progressbar2==3.50.1',
                'pandas==0.25.0'
                ]

setup(
    name='pydpwte',
    version='0.0.1',
    description="Survival Analysis using Deep Learning and Weibull Distribution",
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
