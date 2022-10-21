#!/usr/bin/env python
import os

# uncomment the example you wish to run

# primitive
# from examples.primitive import from_points as example

# operations
# from examples.operation import extrude as example
# from examples.operation import loft as example
# from examples.operation import revolve as example
# from examples.operation import wedge as example
# from examples.operation import airfoil_2d as example

# shapes
# from examples.shape import elbow as example
# from examples.shape import frustum as example
# from examples.shape import cylinder as example
# from examples.shape import revolved_ring as example
# from examples.shape import extruded_ring as example
# from examples.shape import hemisphere as example

# from examples.shape import elbow_wall as example
# from examples.shape import frustum_wall as example

# chaining
from examples.chaining import tank as example
# from examples.chaining import test_tube as example
# from examples.chaining import venturi_tube as example
# from examples.chaining import orifice_plate as example
# from examples.chaining import flywheel as example
# from examples.chaining import coriolis_flowmeter as example

# complex cases
# from examples.complex import helmholtz_nozzle as example
# from examples.complex import karman as example

# advanced
# from examples.advanced import project as example # projection to STL surface
# from examples.advanced import sphere as example # flow around sphere
# from examples.advanced import merged as example

# objects
#from examples.objects import t_pipe as example

try:
    geometry = example.geometry
except:
    geometry = None

mesh = example.get_mesh()

mesh.write(output_path=os.path.join('case', 'system', 'blockMeshDict'), geometry=geometry, debug=False)
os.system("case/Allrun.mesh")
