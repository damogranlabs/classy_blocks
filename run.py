#!/usr/bin/env python
import os

# uncomment the example you wish to run

# primitive
# import examples.primitive.from_points

# operations
# import examples.operation.extrude
# import examples.operation.loft
# import examples.operation.revolve
# import examples.operation.wedge
# import examples.operation.airfoil_2d

# shapes
# import examples.shape.elbow
# import examples.shape.frustum
# import examples.shape.cylinder
# import examples.shape.ring

# complex
# import examples.complex.piping
# import examples.complex.helmholtz_nozzle
import examples.complex.karman
# import examples.complex.pump_volute

os.system("case/Allrun.mesh")
