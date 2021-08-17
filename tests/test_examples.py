import unittest
import os

from tests.fixtures import ExecutedTestsBase

class PrimitiveTests(ExecutedTestsBase):
    def test_from_points(self):
        from examples.primitive import from_points as test_example
        self.run_and_check(test_example)

class OperationTests(ExecutedTestsBase):
    def test_extrude(self):
        from examples.operation import extrude as test_example
        self.run_and_check(test_example)

    def test_loft(self):
        from examples.operation import loft as test_example
        self.run_and_check(test_example)
    
    def test_revolve(self):
        from examples.operation import revolve as test_example
        self.run_and_check(test_example)
    
    def test_wedge(self):
        from examples.operation import wedge as test_example
        self.run_and_check(test_example)
    
    def test_airfoil2d(self):
        from examples.operation import airfoil_2d as test_example
        self.run_and_check(test_example)

class ShapeTests(ExecutedTestsBase):
    def test_elbow(self):
        from examples.shape import elbow as test_example
        self.run_and_check(test_example)
    
    def test_frustum(self):
        from examples.shape import frustum as test_example
        self.run_and_check(test_example)
    
    def test_ring(self):
        from examples.shape import ring as test_example
        self.run_and_check(test_example)

class ComplexTests(ExecutedTestsBase):
    def test_piping(self):
        from examples.complex import piping as test_example
        self.run_and_check(test_example)

    def  test_helmholtz_nozzle(self):
        from examples.complex import helmholtz_nozzle as test_example
        self.run_and_check(test_example)
    
    def test_karman(self):
        from examples.complex import karman as test_example
        self.run_and_check(test_example)

    def test_volute(self):
        from examples.complex import pump_volute as test_example
        self.run_and_check(test_example)