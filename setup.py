#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open("README.md") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

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
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="classy_blocks",
    name="classy_blocks",
    packages=["classy_blocks"],
    package_dir={"classy_blocks": "src/classy_blocks"},
    url="https://github.com/FranzBangar/classy_blocks",
    version="0.1.0",
    zip_safe=False,
    install_requires=["numpy", "scipy", "Jinja2"],
)
