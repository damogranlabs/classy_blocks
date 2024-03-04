# classy_blocks_extensions

HexMesh

These python routines allow the user to create a hexahedral mesh from the classy_blocks mesh object
this hexahedral mesh contain the cell definition of the cells that make up the class_blocks geometry

The hexmesh is created by adding the following two lines at the bottom of the classy_blocks python script

hexmesh = cb.HexMesh(mesh, quality_metrics=True)
hexmesh.write_vtk(os.path.join("hexmesh.vtk"))

the resulting vtk file can be read by paraview
this allows viewing of the mesh and has attributes of cellid, blockid(if multiple blocks), and cell quality(if requested)

Note - THIS DOES NOT USE THE BOCKMESH ALOGRITHIM
this is a home grown alogrithim (although I found a reference that does something very similar afterwards)
created to allow additional properties to be added to the mesh on a cell by cell basis for a problem that
is not CFD

this extension works for all the example file in the classy_blocks examples folder
however at present no tests have been written

to allow projections of a face onto a surface that is defined in a an stl file then the following are required

numpy-stl (https://pypi.org/project/numpy-stl/)
python-utils ( install via anaconda or from https://github.com/WoLpH/python-utils)

no code in the classy blocks library has been changed
