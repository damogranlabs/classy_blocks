# classy_examples
Example project, cases and tests for [classy_blocks](https://github.com/damogranlabs/classy_blocks).

# Usage
## Installation
1. Prerequisites
    1. OpenFOAM (just about any version)
    1. python3
    1. `pip install numpy scipy jinja2`
1. Fetch classy_blocks in one of the following ways:
    1. As a [git submodule](https://git-scm.com/book/en/v2/Git-Tools-Submodules): `git submodule add git@github.com:damogranlabs/classy_blocks.git`
    1. Just download the code as a `.zip` file and extract it to your project
1. Run `run.py`
1. Open `case/case.foam` with ParaView to inspect the mesh.

# Showcase
These are some screenshots of parametric models, built with classy_blocks.

Rectangular ducts (Extrude and Revolve Operations)
![Ducts](https://raw.githubusercontent.com/damogranlabs/classy_examples/main/showcase/elbows.png "Ducts")

3D pipes with twists and turns (Elbow and Cylinder Shapes)
![Piping](https://raw.githubusercontent.com/damogranlabs/classy_examples/main/showcase/piping.png "Piping")

A simple tank with rounded edges
![Tank](https://raw.githubusercontent.com/damogranlabs/classy_examples/main/showcase/tank.png "Tank")

A flywheel in a case. VTK Blocking output for debug is shown in the middle.
![Flywheel](https://raw.githubusercontent.com/damogranlabs/classy_examples/main/showcase/flywheel.png "Flywheel")

Venturi tube
![Venturi tube](https://raw.githubusercontent.com/damogranlabs/classy_examples/main/showcase/venturi_tube.png "Venturi tube")

Coriolis flowmeter with meshed fluid (blue) and solid section (white), ready for an FSI simulation.
![Coriolis Flowmeter](https://raw.githubusercontent.com/damogranlabs/classy_examples/main/showcase/coriolis_flowmeter.png "Coriolis Flowmeter")

2D mesh for studying Karman Vortex Street
![Karman Vortex Street](https://raw.githubusercontent.com/damogranlabs/classy_examples/main/showcase/karman.png "Karman vortex street")

Helmholtz nozzle, a resonator with sharp edges. See [this sketch](https://www.researchgate.net/figure/Schematic-diagram-of-a-Helmholtz-oscillator-and-its-operating-principles_fig6_305275686).
![Helmholtz nozzle](https://raw.githubusercontent.com/damogranlabs/classy_examples/main/showcase/resonator.png "Helmholtz resonator")

A real-life square volute with a blunt cutwater:
![Square Volute](https://raw.githubusercontent.com/damogranlabs/classy_examples/main/showcase/volute_square.png "Square Volute")

Edges and faces, projected to an STL surface
![Projected](https://raw.githubusercontent.com/damogranlabs/classy_examples/main/showcase/projected.png "Projected edges and faces")

Mesh for studying flow around a sphere, with projected edges and faces
![Sphere](https://raw.githubusercontent.com/damogranlabs/classy_examples/main/showcase/sphere.png "Flow around a sphere")

A parametric, Low-Re mesh of a real-life impeller *(not included in examples)*:
![Impeller - Low Re](https://raw.githubusercontent.com/damogranlabs/classy_examples/main/showcase/impeller_full.png "Low-Re Impeller")
