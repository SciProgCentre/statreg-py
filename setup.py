#!/usr/bin/python
"""
"""
import setuptools
import os

with open(os.path.join(os.path.dirname(__file__), "README.md"), "r") as fh:
    long_description = fh.read()


SCIPY_MIN_VERSION = '0.13.3'
NUMPY_MIN_VERSION = '1.8.2'

setuptools.setup(
    name="statreg",
    version="0.0.1",
    author="Nuclear Physics Methods", # "Alexey Khudyakov, Mikhail Zeleniy, Mariia Poliakova, Alexander Nozik",
    author_email="npm@mipt.ru",  #"alexey.skladnoy@gmail.com",
    url='http://npm.mipt.ru/',
    description="Implementation of Turchin's statistical regularization",
    license = "Apache License 2.0",
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords="statreg deconvolution regularization",
    packages=setuptools.find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    classifiers=(
        "Programming Language :: Python :: 3",
    ),
    project_urls={
        "Bug Tracker": "https://github.com/mipt-npm/statreg-py",
        "Documentation": "https://github.com/mipt-npm/statreg-py",
        "Source Code": "https://github.com/mipt-npm/statreg-py",
    },
    install_requires=[
        'numpy>={0}'.format(NUMPY_MIN_VERSION),
        'scipy>={0}'.format(SCIPY_MIN_VERSION)
    ],
    test_suite = 'tests'
)
