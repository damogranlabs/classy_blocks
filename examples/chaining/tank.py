import os
import classy_blocks as cb

# a cylindrical tank with round end caps
diameter = 0.5
length = 0.5 # including end caps

# refer to other examples for a proper-ish grading setup
h = 0.05

mesh = cb.Mesh()

cylinder = Cylinder(
    [0, 0, 0],
    [length, 0, 0],
    [0, diameter/2, 0]
)

wall_name = 'tank_wall'

cylinder.chop_axial(start_size=h)
cylinder.chop_radial(start_size=h)
cylinder.chop_tangential(start_size=h)
cylinder.set_outer_patch(wall_name)

start_cap = cb.Hemisphere.chain(cylinder, start_face=True)
start_cap.chop_axial(start_size=h)
start_cap.set_outer_patch(wall_name)

end_cap = cb.Hemisphere.chain(cylinder, start_face=False)
end_cap.chop_axial(start_size=h)
end_cap.set_outer_patch(wall_name)

mesh.add(cylinder)
mesh.add(start_cap)
mesh.add(end_cap)

mesh.write(os.path.join('..', 'case', 'system', 'blockMeshDict'), debug_path='debug.vtk')
