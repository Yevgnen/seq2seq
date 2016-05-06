#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import find_packages, setup

setup(
    name='seq2seq',
    version='0.1',
    author='Yevgnen',
    packages=find_packages('seq2seq'),
    package_dir={'': 'seq2seq'},
    zip_safe=False
)
