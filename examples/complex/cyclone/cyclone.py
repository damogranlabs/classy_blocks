#!/usr/bin/env python
import os
from typing import List

import parameters as params
from geometry import geometry
from regions.body import ChainSketch, Cone, LowerBody, UpperBody
from regions.core import Core, Outlet
from regions.fillaround import FillAround
from regions.inlet import InletExtension, InletPipe
from regions.inner_ring import InnerRing
from regions.pipe import Pipe
from regions.region import Region
from regions.skirt import Skirt

import classy_blocks as cb

mesh = cb.Mesh()


# A Region is an independent part of the mesh, constructed from
# various other entities but each Region has the same methods and properties.
def add_regions(regions: List[Region]) -> None:
    for region in regions:
        for element in region.elements:
            mesh.add(element)

        region.chop()
        region.project()
        region.set_patches()


# First (bad) guess at blocking: inlet is a semicylinder
# with top faces snapped to body, then the two faces of core
# are moved towards the center
inlet = InletPipe()
# Skirt contains 4 blocks, extruded from displaced inlet top faces
skirt = Skirt(inlet.inlet.shell)
# Skirt and FillAround form a complete outer ring
fillaround = FillAround(skirt.lofts[0], skirt.lofts[-1])
# Inner ring adds another layer of blocks inside the above ring.
inner_ring = InnerRing([*inlet.inner_lofts, *skirt.elements, *fillaround.elements])

optimize_regions = [inlet, skirt, fillaround, inner_ring]
add_regions(optimize_regions)

# This bad blocking needs to be improved before
# making further blocks. It is done on a semi-finished mesh:
mesh.assemble()

# Also, when Mesh is assembled, vertices are created that have
# consistent indexes; those are used in Region.get_clamps() to quickly
# fetch points (no need for cb.*Finders)
optimizer = cb.Optimizer(mesh)

clamps = []
for region in optimize_regions:
    for clamp in region.get_clamps(mesh):
        optimizer.release_vertex(clamp)

optimizer.optimize(tolerance=1e-3, method="SLSQP")

# Now Block objects contain optimization results but those are not reflected in
# user-created Operations. Mesh.backport() will copy the data back.
mesh.backport()
# We'll add new stuff so we don't need those half-finished Blocks.
mesh.clear()

# The optimized inlet is short to aid optimization;
# it's time to extend it to specifications.
inlet_extension = InletExtension(inlet)
add_regions([inlet_extension])

# Upper body is an extension of inner and outer top rings
# (around outlet pipe)
upper_sketch = ChainSketch([skirt, fillaround, inner_ring])
upper = UpperBody(upper_sketch, geometry.l["upper"])

# Lower body begins where outlet pipe ends;
# it extends the above rings plus adds pipe thickness and core
lower_sketch = ChainSketch([upper])
lower = LowerBody(lower_sketch, geometry.l["lower"])

pipe = Pipe(lower)

add_regions([upper, lower, pipe])

# Core is a 12-block cylinder, created from 12 points of pipe ring.
# Again, instead of finding those 12 points (which are difficult to find and sort)
# it is easier to re-assemble the mesh and get the points from debug.vtk in ParaView.
mesh.assemble()
vindexes = [173, 169, 170, 179, 180, 182, 184, 186, 188, 190, 177, 175]
vindexes.reverse()
points = [mesh.vertices[i].position for i in vindexes]
mesh.clear()  # Again, we'll add new stuff right away

core = Core(points)

cone_sketch = ChainSketch([lower, pipe, core])
cone = Cone(cone_sketch)

outlet = Outlet(core.cylinder)

add_regions([core, cone, outlet])


# What's left is to mirror the inlet and upper rings to form a round inlet;
# Operation.mirror() is what we need but chopping needs to be done separately.
def mirror_region(region: Region):
    mirrored = []
    for element in region.elements:
        mirror = element.copy().mirror([0, 0, 1], [0, 0, 0])
        mesh.add(mirror)
        mirrored.append(mirror)

    return mirrored


fillaround_mirror = mirror_region(fillaround)
fillaround_mirror[1].unchop(2)
fillaround_mirror[1].chop(2, start_size=params.BULK_SIZE, end_size=params.BL_THICKNESS)

inner_ring_mirror = mirror_region(inner_ring)
inner_ring_mirror[0].unchop(2)
inner_ring_mirror[0].chop(1, start_size=params.BULK_SIZE)

mirror_region(skirt)
mirror_region(inlet)
mirror_region(inlet_extension)

mesh.set_default_patch("walls", "wall")
mesh.add_geometry(geometry.surfaces)
mesh.settings["scale"] = params.MESH_SCALE
mesh.write(os.path.join("..", "..", "case", "system", "blockMeshDict"), debug_path="debug.vtk")
