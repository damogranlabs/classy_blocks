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

    def test_extruded_ring(self):
        from examples.shape import extruded_ring as test_example

        self.run_and_check(test_example)

    def test_revolved_ring(self):
        from examples.shape import revolved_ring as test_example

        self.run_and_check(test_example)

    def test_hemisphere(self):
        from examples.shape import hemisphere as test_example

        self.run_and_check(test_example)

    def test_frustum_wall(self):
        from examples.shape import frustum_wall as test_example

        self.run_and_check(test_example)

    def test_elbow_wall(self):
        from examples.shape import elbow_wall as test_example

        self.run_and_check(test_example)


class ChainingTests(ExecutedTestsBase):
    def test_flywheel(self):
        from examples.chaining import flywheel as test_example

        self.run_and_check(test_example)

    def test_tank(self):
        from examples.chaining import tank as test_example

        self.run_and_check(test_example)

    def test_test_tube(self):
        from examples.chaining import test_tube as test_example

        self.run_and_check(test_example)

    def test_venturi_tube(self):
        from examples.chaining import venturi_tube as test_example

        self.run_and_check(test_example)

    def test_orifice_plate(self):
        from examples.chaining import orifice_plate as test_example

        self.run_and_check(test_example)

    def test_coriolis_flowmeter(self):
        from examples.chaining import coriolis_flowmeter as test_example

        self.run_and_check(test_example)


class ComplexTests(ExecutedTestsBase):
    def test_helmholtz_nozzle(self):
        from examples.complex import helmholtz_nozzle as test_example

        self.run_and_check(test_example)

    def test_karman(self):
        from examples.complex import karman as test_example

        self.run_and_check(test_example)


class AdvancedTests(ExecutedTestsBase):
    def test_project(self):
        from examples.advanced import project as test_example

        self.run_and_check(test_example)

    def test_sphere(self):
        from examples.advanced import sphere as test_example

        self.run_and_check(test_example)

    def test_merged(self):
        from examples.advanced import merged as test_example

        self.run_and_check(test_example)
