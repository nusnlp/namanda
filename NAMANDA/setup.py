#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import setup, find_packages
import sys


with open('README.md') as f:
    readme = f.read()

setup(
    name='namanda',
    version='0.1.0',
    description='NAMANDA Model',
    long_description=readme,
    python_requires='>=3.5',
    packages=find_packages(exclude=('data')),
)
