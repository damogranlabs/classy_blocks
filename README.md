# classyBlocks
![Elbows](https://github.com/FranzBangar/classyBlocks/blob/master/showcase/elbows.png?raw=true "Elbows")

Python classes for easier creation of openFoam's blockMesh dictionaries.

## What is it
This is a collection of Python classes for creation of blockMeshDict files for OpenFOAM's blockMesh tool.
Its purpose is to avoid manual entering of numbers into blockMeshDict and also avoid the dreadful m4 or #calc.

Since it is easier to crunch numbers and vectors and everything else with numpy it is a better idea to do that
there and then just throw everything into blockMeshDicts. This tool is a link between these two steps.

## When to use it
- If your brain hurts during meticulous tasks such as manual copying of numbers from excel or even paper
- If you don't want to waste energy on low-level stuff such as numbering vertices
- If you have a rather simplish parametric model and would like to make a bunch of simulations with changing geometry (optimizations etc.)

## How to Use
- Calculate the important points of the model. Also calculate the arc point for edges, if needed.
- create a classy_blocks.Mesh() object
- for each block:
    - add vertices with Mesh.add_vertices()
    - add edge points with mesh.add_edge()
    - calculate number of cells and cell size
    - create block from the above vertices with Mesh.add_block()
    - set patches with block.set_patches()
- write the mesh with mesh.write()

## Prerequisites
 - numpy
 - scipy
 - jinja2

## Example usage
# TODO: update
Here's a rather simple example of a single-block box with curved edges.

```python
from functions import vector, rotate
import numpy as np

d = 0.1 # box size
a = np.pi/4 # revolve angle


# four box points on the ground
p1 = [
    vector(-d/2, -d/2, -d/2),
    vector( d/2, -d/2, -d/2),
    vector( d/2,  d/2, -d/2),
    vector(-d/2,  d/2, -d/2),
]

# for points above the ground; rotate them for 45 degrees around z-axis
# for a more interesting 'box'
p2 = [rotate(p + vector(0, 0, d), a, axis='z') for p in p1]

# four edge points in the middle
pe = [
    vector(-d/4, -d/4, 0),
    vector( d/4, -d/4, 0),
    vector( d/4,  d/4, 0),
    vector(-d/4,  d/4, 0),
]
pe = [rotate(p, a/2, axis='z') for p in pe]

# create a Mesh object
mesh = classy_blocks.Mesh()

# add vertices to block
vertices = mesh.add_vertices(p1 + p2)

# add edges between vertices
for i in range(4):
    mesh.add_edge(vertices[i], vertices[i+4], pe[i])

# create a block
block = mesh.add_block(vertices, [10, 10, 10])

# set block's cell size (optional for graded blocks)
block.set_cell_size(2, d/20) # 0, 1, 2 - x, y, z axis, respectively

# set block's patches
block.set_patch(['front', 'back', 'top', 'bottom'], 'walls')
block.set_patch(['left'], 'inlet')
block.set_patch(['right'], 'outlet')

# write mesh
mesh.write('blockMeshDict.template', 'system/blockMeshDict')

# run blockMesh
os.system("blockMesh")
```

## Other Examples
### Elbow
Run `python examples/elbow/example_elbow.py` from this repository's top-level directory.
Then open `examples/elbow/case.foam` in ParaView and check the mesh: it's a
square cross-section ventilation duct with two elbows. `block.set_cell_size()` is used to 
match cell size on block boundaries and to save on cell count where high resolution is not critical.

### Annulus & Taylor vortices
Run `python examples/annulus/example_annulus.py`. This is a simplified model of a wet-running electric motor.
The inner cylinder is rotating and that creates not only rotating field but a complex array of so-called
[Taylor vortexes](https://www.google.com/search?tbm=isch&q=taylor+vortex). Here, blocks are graded as well
to save on cell count.

### Axisymmetric mesh: nozzle
After running `python examples/nozzle/example_nozzle.py` you get a nice axisymmetric wedge mesh
of a kind of a nozzle.

### Fully 3D mesh: cylinder
A cylindrical pipe made from 5 blocks.

## Bonus: geometry functions
Check out `functions.py` for bonus functions. More about that is written in my blog post, [https://damogranlabs.com/2019/11/points-and-vectors/].
There are also a couple of functions used in the example.
Of course you are free to use your own :)

Have fun!

## Showcase
A single channel of an impeller, without volute and with infinitely thin blades:
![Impeller](https://github.com/FranzBangar/classyBlocks/blob/master/showcase/impeller.png?raw=true "Impeller")

A full volute and inlet (impeller mesh is created separately):
![Volute](https://github.com/FranzBangar/classyBlocks/blob/master/showcase/volute.png?raw=true "Volute")
