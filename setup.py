#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages


long_description = """
"""

requirements = [
]

setup(
    name='pydpwte',
    version='0.0.1',
    description="Survival analysis using DeepWeiSurv",
    long_description=long_description,
    #long_description_content_type='text/markdown',
    author="Achraf Bennis",
    author_email='achraf.bennis.b@gmail.com',
    #url='https://github.com/havakv/pycox',
    packages=find_packages(),
    include_package_data=True,
    #install_requires=requirements,
    #license="BSD license",
    zip_safe=False,
    #keywords='pycox',
    classifiers=[
        'Natural Language :: English',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.6',
    ],
    python_requires='>=3.6'
)