#!/usr/bin/env python
import os
from setuptools import setup, find_packages


def read_file(rel_path: str) -> str:
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, rel_path), "r") as f:
        return f.read()


def get_version(rel_path: str) -> str:
    for line in read_file(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


setup(
    name="classy_blocks",
    url="https://github.com/damogranlabs/classy_blocks",
    author="Nejc Jurkovic",
    author_email="kandelabr@gmail.com",
    description="Python classes for easier creation of openFoam's blockMesh dictionaries.",
    long_description=read_file("README.md"),
    keywords=["classy_blocks", "OpenFOAM", "blockMesh"],
    version=get_version("src/classy_blocks/__init__.py"),
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    license="MIT license",
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=["numpy", "scipy", "nptyping"],
    include_package_data=False,
    zip_safe=False,
)
