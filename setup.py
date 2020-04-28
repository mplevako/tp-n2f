#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='TP-N2F',
      version='1.0',
      description='TP-N2F encoder-decoder model',
      author='Max Plevako',
      author_email='mplevako@gmail.com',
      install_requires=['pytorch-lightning'],
      packages=find_packages(),
      )
