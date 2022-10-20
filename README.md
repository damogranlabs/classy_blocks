# classy_blocks

![Flywheel](https://raw.githubusercontent.com/damogranlabs/classy_examples/main/showcase/flywheel.png "Flywheel")

Python classes for easier creation of openFoam's blockMesh dictionaries.

> Warning! This project is currently under development and is not yet very user-friendly. It still lacks some important
> features and probably features a lot of bugs. However, you're welcome to suggest features, improvements, and point out
> bugs.

## What Is It
This is a collection of Python classes for creation of blockMeshDict files for OpenFOAM's blockMesh tool. Its purpose is
to avoid manual entering of numbers into blockMeshDict and also avoid the dreadful `m4` or `#calc`.

Since it is easier to crunch numbers and vectors and everything else with `numpy` it is a better idea to do that there
and then just throw everything into blockMeshDicts. This tool is a link between these two steps.

## When To Use It
- If your brain hurts during meticulous tasks such as manual copying of numbers from excel or even paper
- If you don't want to waste energy on low-level stuff such as numbering vertices
- If you have a rather simplish parametric model and would like to make a bunch of simulations with changing geometry (
  optimizations etc.)

## How To Use It
- If you just need the `classy_blocks`, install them with: `pip install git+https://github.com/damogranlabs/classy_blocks.git`
- If you want to run examples, follow instructions in [EXAMPLES_README.md](EXAMPLES_README.md)
- If you want to contribute, follow instructions in [CONTRIBUTING.md](CONTRIBUTING.md)
# Features

- Write your parametric model's geometry with a short Python script and translate it directly to `blockMeshDict`
- Predefined shapes like `Cylinder` or operations like `Extrude` and `Revolve`
- Simple specification of edges: a single point for circular or a list of points for a spline edge
- Automatic calculation of cell count and grading by specifying any of a number of parameters (cell-to-cell expansion
  ratio, start cell width, end cell width, total expansion ratio)
- Automatic propagation of grading and cell count from block to block as required by blockMesh
- projections of edges and block faces to `geometry` (STL meshes and searchable surfaces)

## Predefined Shapes

- Cone Frustum (truncated cone)
- Cylinder
- Ring (annulus)
- Elbow (bent pipe)
- Hemisphere
- Elbow wall (thickened elbow shell)
- Frustum wall

A simple example:

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

Analogous to a sketch in 3D CAD software, a `Face` is a collection of 4 vertices and 4 edges.
An _operation_ is a 3D shape obtained by swiping a Face into 3rd dimension following one of the rules below.

### Extrude

A block, created from a Face translated by an extrude vector.

### Revolve

A Face is revolved around a given axis so that a circular block with a constant cross-section is created.

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

### Wedge

A special case of revolve for 2D axisymmetric meshes. A list of `(x,y)` points is revolved symetrically around x-axis as
required by a `wedge` boundary condition in OpenFOAM.

### Loft

A block between two Faces. Edges in lofted direction can also be specified.

> See [examples/operations](https://github.com/damogranlabs/classy_examples) for an example of each operation.

## Projection To Geometry

[Any geometry that snappyHexMesh understands](https://www.openfoam.com/documentation/guides/latest/doc/guide-meshing-snappyhexmesh-geometry.html)
is also supported by blockMesh.
That includes searchable surfaces such as spheres and cylinders and triangulated surfaces.
Simply provide geometry definition as documentation specifies, then call a `project_face()` on Block object.

```python
geometry = {
    'terrain': [
        'type triSurfaceMesh;',
        'name terrain;',
        'file "terrain.stl";',
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

`.contract()` is only possible* on `ExtrudedRing` and produces another, smaller `ExtrudedRing`.

> See [examples/chaining](https://github.com/damogranlabs/classy_examples) for an example of each operation.

## Debugging

By default, `mesh.write(...)` creates a `debug.vtk` file where each block represents a hexahedral cell.
By showing `block_ids` with a proper color scale the blocking is clearly visible.
This is most useful when blockMesh fails with errors reporting invalid/inside-out blocks but VTK will
happily show anything.

This can be disabled by using `mesh.write(..., debug=False, ...)`.

# Prerequisites
Package (python) dependencies can be found in *setup.py* file.
Other dependencies that must be installed:
- blockMesh
- OpenFoam
- python

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

## Bonus:

### Geometry functions

Check out `util/geometry.py` for bonus functions. More about that is written in my blog
post, [https://damogranlabs.com/2019/11/points-and-vectors/].
There are also a couple of functions used in the example.
Of course you are free to use your own :)

# TODO

- New Shapes:
    - T-joint
    - X-joint
    - Hemisphere with given angle (a.k.a. cone cap)
- Wall versions of Shapes:
    - HemisphereWall
- Chaining:
    - *FrustumWall.expand()
    - *FrustumWall.contract()
    - *ElbowWall.contract()
    - Box.chain()
    - Block.chain() (low-level), or Block.get_face() -> Face
- Examples
    - Ramjet engine
- Technical stuff:
    - Debugging: connection between block and Shapes
    - More tests
    - Documentation
