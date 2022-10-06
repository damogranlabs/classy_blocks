# classy_blocks

![Flywheel](https://raw.githubusercontent.com/damogranlabs/classy_examples/main/showcase/flywheel.png "Flywheel")

Python classes for easier creation of OpenFOAM's blockMesh dictionaries.

> Warning! This project is currently under development and is not yet very user-friendly. It still lacks some important
> features and probably features a lot of bugs. However, you're welcome to suggest features, improvements, and point out
> bugs.

> tl;dr: install the package with `pip install git+https://github.com/damogranlabs/classy_blocks.git`, 
> clone the [classy_examples](https://github.com/damogranlabs/classy_examples) repository and run `run.py`.

> For those that still want the submodule, clone [this commit (7e8e7bcd85b5bac40bcffabcddef6a220a4c6f9f)](https://github.com/damogranlabs/classy_blocks/tree/7e8e7bcd85b5bac40bcffabcddef6a220a4c6f9f).

# About

_blockMesh_ is a very powerful mesher but the amount of manual labour it requires to make even the simplest
meshes makes it mostly useless. Even attempts to simplify or parametrize _blockMeshDict_s with `#calc` or even
the dreadful `m4` quickly become unmanageable and cryptic.

classy_blocks' aim is to ease the burden of meticulous work by providing a
more intuitive workflow, off-the-shelf parts and some automatic helpers for building and optimization of block-structured hexahedral meshes.
Still it is not an automatic mesher and therefore some kinds of geometry are more suited than others.

## Useful For
### Fields
- Turbomachinery (impellers, propellers)
- Microfluidics
- Flow around buildings
- Heat transfer (PCB models, heatsinks)
- Airfoils (2D)
- Solids (heat transfer, mechanical stresses)

### Cases
- Simpler rotational geometry (immersed rotors, mixers)
- Pipes/channels
- Tanks/plenums/containers
- External aerodynamics of blunt bodies
- Modeling thin geometry (seals, labyrinths)
- Parametric studies
- Background meshes for snappy (cylindrical, custom)
- 2D and axisymmetric cases
- Overset meshes

## Not Good For
- External aerodynamics of vehicles (too complex to mesh manually, without refinement generates too many cells)
- Complex geometry in general
- One-off simulations (use automatic meshers)

# Features

## Workflow
As opposed to blockMesh, where the user is expected to manually enter pre-calculated vertices, edges, blocks and whatnot, classy_blocks tries to mimic procedural modeling of modern 3D CAD programs. Here, a Python script contains steps that describe geometry of blocks, their cell count, grading, patches and so on. At the end, the procedure is translated directly to blockMeshDict and no manual editing of the latter should be required.

## Building Elements
_Unchecked items are not implemented yet_

- [x] Manual definition of a Block with Vertices, Edges and Faces
- [x] Operations (Loft, Extrude, Revolve)
    - [x] Loft
    - [x] Extrude
    - [x] Revolve
    - [x] Wedge (a shortcut to Revolve for 2D axisymmetric cases)
- [x] Predefined Shapes
    - [x] Box (shortcut to Block aligned with coordinate system)
    - [x] Elbow (bent pipe of various diameters/cross-sections)
    - [x] Cone Frustum (truncated cone)
    - [x] Cylinder
    - [x] Ring (annulus)
    - [x] Hemisphere
    - [x] Elbow wall (thickened shell of an Elbow)
    - [x] Frustum wall
    - [ ] Hemisphere wall
- [ ] Predefined parametric Objects
    - [ ] T-joint (round pipes)
    - [ ] X-joint
    - [ ] N-joint (multiple pipes)
    - [ ] Box with hole

## Modifiers
After blocks have been placed, it is possible to create new geometry based on placed blocks or to modify them.

- [ ] Move Block's Vertex/Edge/Face
- [ ] Delete a Block created by a Shape or Object
- [x] Project Block's Vertex/Edge/Face*
- [ ] Chain Block's face to generate a new block
- [x] Chain Shape's surface (top/bottom/outer) to generate a new Shape
- [ ] Join two Blocks by extending their Edges
- [ ] Offset Block's faces to form new Blocks for easier creation of layouts with boundary layers that don't propagate into domain
- [ ] Optimize Vertex positions

## Meshing specification
- [x] Simple definition of edges: a single point for circular, a list of points for a spline edge, name of geometry for projecting
- [x] Automatic calculation of cell count and grading by specifying any of a number of parameters (cell-to-cell expansionr atio, start cell width, end cell width, total expansion ratio)
- [ ] [Edge grading](https://www.openfoam.com/documentation/user-guide/4-mesh-generation-and-conversion/4.3-mesh-generation-with-the-blockmesh-utility#x13-450004.3.1.3) (separate specification for each edge)
- [x] Automatic propagation of grading and cell count from a single block to all connected blocks as required by blockMesh
- [x] Projections of vertices, edges and block faces to geometry (triangulated and [searchable surfaces](https://www.openfoam.com/documentation/guides/latest/doc/guide-meshing-snappyhexmesh-geometry.html#meshing-snappyhexmesh-searchable-objects))*
- [x] Face merging as described by [blockMesh user guide](https://www.openfoam.com/documentation/user-guide/4-mesh-generation-and-conversion/4.3-mesh-generation-with-the-blockmesh-utility#x13-470004.3.2). Breaks the pure-hexahedral-mesh rule but can often save the day for trickier geometries.

* Not implemented: projected Vertex

# Examples

## Shapes
A simple Cylinder:

```python
inlet = Cylinder([x_start, 0, 0], [x_end, 0, 0], [0, 0, radius])
inlet.chop_radial(count=n_cells_radial, end_size=boundary_layer_thickness)
inlet.chop_axial(start_size=axial_cell_size, end_size=2*axial_cell_size)
inlet.chop_tangential(count=n_cells_tangential)

inlet.set_bottom_patch('inlet')
inlet.set_outer_patch('wall')
inlet.set_top_patch('outlet')
mesh.add(inlet)
```

> See [examples/shape](https://github.com/damogranlabs/classy_examples) for use of each _shape_
> and [examples/complex](https://github.com/damogranlabs/classy_examples) for a more real-life example usage of shapes.

## Operations

Analogous to a sketch in 3D CAD software, a Face is a set of 4 vertices and 4 edges.
An _Operation_ is a 3D shape obtained by swiping a Face into 3rd dimension by a specified rule; an example of Revolve:

```python
# a quadrangle with one curved side
base = Face(
    [ # quad vertices
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0]
    ],
    [ # edges: None specifies straight edge
        [0.5, -0.2, 0], # single point: arc
        None,
        None,
        None
  ]
)

revolve = Revolve(
    base, # face to revolve
    f.deg2rad(45), # revolve angle
    [0, -1, 0], # axis
    [-2, 0, 0]  # origin
)

revolve.chop(0, count=15) # first edge
revolve.chop(1, count=15) # second edge
revolve.chop(2, start_size=0.05) # revolve direction
mesh.add(revolve)
```

> See [examples/operations](https://github.com/damogranlabs/classy_examples) for an example of each operation.

## Projection To Geometry

[Any geometry that snappyHexMesh understands](https://www.openfoam.com/documentation/guides/latest/doc/guide-meshing-snappyhexmesh-geometry.html)
is also supported by blockMesh.
That includes searchable surfaces such as spheres and cylinders and triangulated surfaces.
Simply provide geometry definition as documentation specifies, then call a `project_face()` on Block object.

```python
geometry = {
    'terrain': [
        'type triSurfaceMesh',
        'name terrain',
        'file "terrain.stl"',
    ]
}

base = Face([
    [-1, -1, -1],
    [ 1, -1, -1],
    [ 1,  1, -1],
    [-1,  1, -1]
])

extrude = Extrude(base, [0, 0, 2])
extrude.block.project_face('bottom', 'terrain', edges=True)
extrude.chop(0, count=20)
extrude.chop(1, count=20)
extrude.chop(2, start_size=0.01, c2c_expansion=1.2)

extrude.set_patch('bottom', 'terrain')
mesh.add(extrude)
mesh.set_default_patch('atmosphere', 'patch')
```

## Face Merging

Simply provide names of patches to be merged and call `mesh.merge_patches(<master>, <slave>)`.
classy_blocks will take care of point duplication and whatnot.

```python
box = Box([-0.5, -0.5, 0], [0.5, 0.5, 1])
for i in range(3):
    box.chop(i, count=25)
box.set_patch('top', 'box_top')
mesh.add(box)

cylinder = Cylinder(
    [0, 0, 1],
    [0, 0, 2],
    [0.25, 0, 1]
)
cylinder.chop_axial(count=10)
cylinder.chop_radial(count=10)
cylinder.chop_tangential(count=20)

cylinder.set_bottom_patch('cylinder_bottom')
mesh.add(cylinder)

mesh.merge_patches('box_top', 'cylinder_bottom')
```

## Chaining and expanding/contracting

Useful for Shapes, mostly for piping and rotational geometry; An existing Shape's start or end sketch can be reused as a
starting sketch for a new Shape, as long as they are compatible.
For instance, an `Elbow` can be _chained_ to a `Cylinder` just like joining pipes in plumbing.

Moreover, most shapes* can be expanded to form a _wall_ version of the same shape. For instance, expanding a `Cylinder`
creates an `ExtrudedRing`.

> See [examples/chaining](https://github.com/damogranlabs/classy_examples) for an example of each operation.

## Debugging

By default, a `debug.vtk` file is created where each block represents a hexahedral cell.
By showing `block_ids` with a proper color scale the blocking can be visualized.
This is useful when blockMesh fails with errors reporting invalid/inside-out blocks but VTK will
happily show anything.

# Prerequisites

- numpy
- scipy
- jinja2
- blockMesh (OpenFOAM)

# Technical Information

There's no official documentation yet so here are some tips for easier navigation through source.

## Classes

These contain data to be written to blockMeshDict and also methods for point manipulation and output formatting.

- `Vertex`: an object containing (x, y, z) point and its index in blockMesh. Also formats output so OpenFOAM can read
  it.
- `Edge`: a collection of two `Vertex` indexes and a number of `Point`s in between
- `Face`: a collection of exactly 4 `Vertex`es and optionally 4 `Edge`s. If only some of the 4 edges are curved a `None`
  can be passed instead of a list of edge points.
- `Block`: contains `Vertex` and `Edge` objects and other block data: patches, number of cells, grading, cell zone,
  description.
- `Mesh`: holds lists of all blockMeshDict data: `Vertex`, `Edge`, `Block`.

## Block Creation

A blockMesh is created from a number of blocks, therefore a `Block` object is in the center of attention. It is always
created from 8 vertices - the order of which follows
the [sketch on openfoam.com user manual](https://www.openfoam.com/documentation/user-guide/blockMesh.php). Edges are
added to block by specifying vertex indexes and a list of points in between.

`Operation`s simply provide an interface for calculating those 8 vertices in a _user-friendly_ way, taking care of
intricate calculations of points and edges.

Once blocks are created, additional data must/may be added (number of cells, grading, patches, projections).
Finally, all blocks must be added to Mesh. That will prepare data for blockMesh and create a blockMeshDict from a
template.

## Data Preparation

After all blocks have been added, an instance of `Mesh` class only contains a list of blocks. Each block self-contains
its own data. Since blocks share vertices and edges and blockMesh needs separate lists of both, `Mesh.prepare_data()`
will translate all individual blocks' data to format blockMesh will understand:

- collect new vertices and re-use existing ones
- collect block edges, skipping duplicates
- run count and grading calculations on blocks where those are set
- assign neighbours of each block, then try to propagate cell counts and gradings through the whole mesh; if that fails,
  an `Exception` is raised
- gather a list of projected faces and edges

# TODO
- Unchecked list items from [Features](#features)
- Chaining:
    - *FrustumWall.expand()
    - *FrustumWall.contract()
    - *ElbowWall.contract()
    - Box.chain()
    - Block.chain() (low-level), or Block.get_face() -> Face
- Optimization
- Examples
    - Ramjet engine
- Technical stuff:
    - Redefine classes to support modification
    - Simplify imports (direct import from classy_blocks instead of specifying complete module paths)
    - Package to pypi
    - Debugging: connection between block and Shapes, naming
    - More tests
    - Documentation
