#!/usr/bin/env python3
"""Setup file for NMRforMD package."""
# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
#
# Copyright (c) 2023-2026 Authors and contributors
# Simon Gravelle
#
# Released under the GNU Public Licence, v3 or any higher version
# SPDX-License-Identifier: GPL-3.0-or-later

from setuptools import setup

setup(name='nmrdfrommd',
      version='0.2.0',
      description='Calculate dipolar NMR relaxation rates from \
                   molecular dynamics trajectory file',
      long_description=open('LONG_DESCRIPTION.rst').read(),
      long_description_content_type='text/x-rst',
      url='https://github.com/NMRDfromMD/nmrdfrommd',
      download_url='https://github.com/NMRDfromMD/nmrdfrommd/releases/download/v0.2.0/nmrdfrommd-0.2.0.tar.gz',  # noqa
      author='Simon Gravelle',
      author_email='simon.gravelle@cnrs.fr',
      license='GNU GENERAL PUBLIC LICENSE',
      packages=['nmrdfrommd'],
      zip_safe=False,
      install_requires=[
       "mdanalysis",
       "pytest",
       "numpy",
       "coverage",
       "scipy",
      ]
      )
