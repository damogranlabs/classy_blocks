### Geometry
# See docs/geometry.svg for a quick sketch

# Diameter and length of inlet pipe [mm]
D_INLET = 120
L_INLET = 500
# Body diameter and length
D_BODY = 300
L_BODY = 500
# Outlet pipe inner diameter and length;
# the 'in' part is inside body, out is extended away from the top surface.
# Keep in mind that L_OUTLET_IN must be greater than D_INLET
D_OUTLET = 140
L_OUTLET_IN = 140
L_OUTLET_OUT = 300
# outlet pipe wall thickness
T_PIPE = 10
# Length and end diameter of the conical section
L_CONE = 400
D_CONE = 120

DIM_SCALE = 1  # multiply all above dimensions with this
MESH_SCALE = 0.001  # goes into blockMeshDict.scale

# Mesh
# (use same dimensions as for geometry)
BULK_SIZE = DIM_SCALE * 10
BL_THICKNESS = DIM_SCALE * 0.5
C2C_EXPANSION = 1.2
