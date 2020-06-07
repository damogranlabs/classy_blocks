# classy_blocks
![Elbows](showcase/piping.png "Elbows")

Python classes for easier creation of openFoam's blockMesh dictionaries.

> Warning! This project is currently under development and is not yet very user-friendly. It still lacks some important features and probably features a lof of bugs. However, you're welcome to suggest features, improvements, and point out bugs. There is also no other way to use it but copy the directory but in the future it might become a _pip package_.

# What is it
This is a collection of Python classes for creation of blockMeshDict files for OpenFOAM's blockMesh tool. Its purpose is to avoid manual entering of numbers into blockMeshDict and also avoid the dreadful `m4` or `#calc`.

Since it is easier to crunch numbers and vectors and everything else with `numpy` it is a better idea to do that there and then just throw everything into blockMeshDicts. This tool is a link between these two steps.

# When to use it
- If your brain hurts during meticulous tasks such as manual copying of numbers from excel or even paper
- If you don't want to waste energy on low-level stuff such as numbering vertices
- If you have a rather simplish parametric model and would like to make a bunch of simulations with changing geometry (optimizations etc.)

# Usage
The basis is the `Mesh` class. You can add vertices, edges, blocks and shapes to it. When you're done, a blockMeshDict is created using `mesh.write()`.

There are 3 different abstraction levels:
1. Shapes take points and vectors as parameters (depending on that shape) and returns an object that is passed to Mesh. Everything (including blocks) is created implicitly and you don't have to deal with any of the low-level stuff.
1. An _Operation_ combines a `Face` (4 points, 4 edges) and transforms it into a block using a rule based on the operation itself. `Revolve`, for instance, rotates the face around a specified axis and also creates circular edges between the two faces.
1. The lowest-level approach is to calculate vertex and edge points manually and create blocks from those.

## Predefined Shapes
 - Cone Frustum (truncated cone)
 - Cylinder
 - Ring (annulus)
 - Elbow (bent pipe)

> See `examples/shape` for example use of each 'shape'. See `examples/complex` for a more real-life example usage of shapes.

## Using Operations
A `Face` is a collection of 4 vertices and 4 edges. It is a 2D object analogous to a _sketch_ in most CAD modelers - once defined, it can be used to create 3D shapes with various _operations_.

### Extrude
A single block is created from a `Face` translated by an extrude vector.

### Revolve
A single face is revolved around a given axis so that a circular object with a constant cross-section is created.

### Wedge
A special case of revolve for 2D axisymmetric meshes. A list of `(x,y)` points is revolved symetrically around x-axis as required by a `wedge` boundary condition in OpenFOAM.

### Loft
A single block, created between two `Face`s. Edges between the same vertices on the two faces can also be specified.

> See `examples/operations` for an example of each operation.

**Basically any kind of block can be created with Loft so this is usually as low-level as you'll need to go.**

## Low level: Manual block creation using vertices, edges, etc.
Workflow is similar to bare-bones blockMeshDict but you don't have to track
vertex/edge/block indexes. You do:

 1. Calculate block vertices
 1. Calculate edge points, if any
 1. Create a Mesh object
 1. Create a Block object with vertices and edges for each block
 1. Add blocks to mesh
 1. Set block cell count and sizes
 1. Assign patches
 1. Pray you did everything right
 1. Cry because you didn't

> See `examples/primitive/` for a demonstration.

## Example usage
Uncomment the desired example in `./run_example.py`. Then run the file and open `examples/meshCase/case.foam` with ParaView.

## Other Examples
### Elbow
Run `python examples/elbow/example_elbow.py` from this repository's top-level directory.
Then open `examples/elbow/case.foam` in ParaView and check the mesh: it's a
square cross-section ventilation duct with two elbows. `block.set_cell_size()` is used to 
match cell size on block boundaries and to save on cell count where high resolution is not critical.

### Annulus & Taylor vortices
Run `python examples/annulus/example_annulus.py`. This is a simplified model of a wet-running electric motor.
The core cylinder is rotating and that creates not only rotating field but a complex array of so-called
[Taylor vortexes](https://www.google.com/search?tbm=isch&q=taylor+vortex). Here, blocks are graded as well
to save on cell count.

### Axisymmetric mesh: nozzle
After running `python examples/nozzle/example_nozzle.py` you get a nice axisymmetric wedge mesh
of a kind of a nozzle.

### Fully 3D mesh: cylinder
A cylindrical pipe made from 5 blocks.

## Prerequisites
 - numpy
 - scipy
 - jinja2

## Showcase
A single channel of an impeller, without volute and with infinitely thin blades:
![Impeller](https://github.com/FranzBangar/classyBlocks/blob/master/showcase/impeller.png?raw=true "Impeller")

A full volute and inlet (impeller mesh is created separately):
![Volute](https://github.com/FranzBangar/classyBlocks/blob/master/showcase/volute.png?raw=true "Volute")

# Technical information
## Classes
These contain data to be written to blockMeshDict and also methods for point manipulation and output formatting.

- `Vertex`: an object containing (x, y, z) point and its index in blockMesh. Also formats output so OpenFOAM can read it.
- `Edge`: a collection of two `Vertex` indexes and a number of `Point`s in between
- `Face`: a collection of exactly 4 `Vertex`es and optionally 4 `Edge`s. If only some of the 4 edges are curved a `None` can be passed instead of a list of edge points.
- `Block`: contains `Vertex` and `Edge` objects and other block data: patches, number of cells, grading, cell zone, description.
- `Mesh`: holds lists of all blockMeshDict data: `Vertex`, `Edge`, `Block`.

## Block (Geometry) creation
A blockMesh is created from a number of blocks, therefore a `Block` object is in the center of attention. A `Block` can be created in different ways:
 1. Directly from 8 vertices. The order of vertices follows the [sketch on openfoam.com user manual](https://www.openfoam.com/documentation/user-guide/blockMesh.php). Edges are added to block by specifying vertex indexes and a list of points in between.
 1. From 2 `Face` objects. Edges between the same vertex on both faces can be provided as well.
 1. From any of the Operations. Creation of a Block with an Operation depends on type of specific operation.
 1. By using a predefined Shape. Creation procedure again differs from shape to shape. Usually multiple blocks are returned.

Once blocks are created, additional data may be added (number of cells, grading, patches).
Finally, all blocks must be added to Mesh. That will prepare data for blockMesh and create a blockMeshDict from a template.

## Data preparation
`Mesh` only contains a list of blocks. Each block self-contains its data. Since blocks share vertices and edges and blockMesh needs separate lists of the latter, `Mesh.prepare_data()` will traverse its list of blocks and store vertices and edges to separate lists. During that it will check for duplicates and drop them. It will also collect patches from all blocks and store them into a template-readable list.

## Bonus: geometry functions
Check out `functions.py` for bonus functions. More about that is written in my blog post, [https://damogranlabs.com/2019/11/points-and-vectors/].
There are also a couple of functions used in the example.
Of course you are free to use your own :)

# TODO
 - More sophisticated tests
 - Neighbour awareness between blocks (consistent grading, cell count, etc.)
 - Block.size (multiple issues)
 - More examples from real life
    - heater/boiler
    - labyrinth seals
    - rotating annulus (Taylor-Couette vortex)
 - Package (`pip install...`)