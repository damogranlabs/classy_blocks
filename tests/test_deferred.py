import unittest
import os

from classy_blocks.classes.block import DeferredFunction

def simple_function(a, b=0):
    return a + b

class DeferredTestCase(unittest.TestCase):
    def test_deferred_function_1(self):
        """ a deferred function with all arguments passed """
        df = DeferredFunction(simple_function, 1, 2)
        self.assertEqual(df.call(), 3)

    def test_deferred_function_2(self):
        """ a deferred function with one keyword argument left default """
        df = DeferredFunction(simple_function, 1)
        self.assertEqual(df.call(), 1)

if __name__ == '__main__':
    unittest.main()