#!/usr/bin/env python
import os

geometry = None

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
# from examples.shape import ring as example

# complex cases
# from examples.complex import piping as example
# from examples.complex import helmholtz_nozzle as example
# from examples.complex import karman as example
from examples.complex import pump_volute as example

# advanced: projection
#from examples.advanced import project as example
#geometry = example.geometry

mesh = example.get_mesh()

mesh.write(output_path=os.path.join('case', 'system', 'blockMeshDict'), geometry=geometry)
os.system("case/Allrun.mesh")
