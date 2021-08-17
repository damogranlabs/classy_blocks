#!/usr/bin/env python
import unittest
import os

from tests.fixtures import FixturedTestCase

def load_tests(loader, standard_tests, pattern):
    # top level directory cached on loader instance
    this_dir = os.path.dirname(__file__)
    package_tests = loader.discover(start_dir=this_dir, pattern=pattern)
    standard_tests.addTests(package_tests)
    return standard_tests

if __name__ == '__main__':
    load_tests()