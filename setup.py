#!/usr/bin/env python
import os
from setuptools import setup, find_packages


def read_file(rel_path: str) -> str:
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, rel_path), "r") as f:
        return f.read()


with open("README.md") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()


def get_version(rel_path: str) -> str:
    for line in read_file(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


setup(
    author="Nejc Jurkovic",
    author_email="kandelabr@gmail.com",
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    description="Python classes for easier creation of openFoam's blockMesh dictionaries.",
    license="MIT license",
    long_description=read_file("README.md"),
    include_package_data=True,
    keywords=["classy_blocks", "openFoam", "blockMesh"],
    name="classy_blocks",
    version=get_version("src/classy_blocks/__init__.py"),
    packages=find_packages("src"),
    package_dir={"": "src"},
    url="https://github.com/damogranlabs/classy_blocks",
    zip_safe=False,
    install_requires=["numpy", "scipy", "Jinja2"],
)
