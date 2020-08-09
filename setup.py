# Copyright (c) James Vuckovic. All rights reserved.
# Licensed under the MIT license.
"""Setup for pip package.

Adapted from https://github.com/deepmind/dm-haiku/blob/master/setup.py
"""

from setuptools import find_namespace_packages
from setuptools import setup


def _get_version():
  with open('jax_bayes/__init__.py') as fp:
    for line in fp:
      if line.startswith('__version__'):
        g = {}
        exec(line, g)  # pylint: disable=exec-used
        return g['__version__']
    raise ValueError('`__version__` not defined in `jax_bayes/__init__.py`')


def _parse_requirements(requirements_txt_path):
  with open(requirements_txt_path) as fp:
    return fp.read().splitlines()


_VERSION = _get_version()

EXTRA_PACKAGES = {
    'jax': ['jax>=0.1.74'],
    'jaxlib': ['jaxlib>=0.1.51'],
}

setup(
    name='jax-bayes',
    version=_VERSION,
    # url='https://github.com/deepmind/dm-haiku',
    license='MIT',
    author='James Vuckovic',
    description='jax-bayes is a bayesian inference library for JAX.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author_email='james@jamesvuckovic.com',
    # Contained modules and scripts.
    packages=find_namespace_packages(exclude=['*_test.py']),
    install_requires=_parse_requirements('requirements.txt'),
    extras_require=EXTRA_PACKAGES,
    # tests_require=_parse_requirements('requirements-test.txt'),
    requires_python='>=3.6',
    include_package_data=True,
    zip_safe=False,
    # PyPI package information.
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Software Development :: Libraries',
    ],
)