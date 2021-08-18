# classy_examples
Example project and cases for [classy_blocks](https://github.com/damogranlabs/classy_blocks).

# Usage
## Installation
1. Prerequisites
    1. OpenFOAM (just about any version)
    1. python3
    1. _Optional: use a virtual environment_
    1. `pip install numpy scipy jinja2`
1. Fetch classy_blocks in one of the following ways:
    1. As a [git submodule](https://git-scm.com/book/en/v2/Git-Tools-Submodules): `git submodule add git@github.com:damogranlabs/classy_examples.git`
    1. Just download the code as a `.zip` file and extract it to your project
1. Run `run.py`
1. Open `case/case.foam` with ParaView to inspect the mesh.

# Showcase
These are some screenshots of parametric models, built with classy_blocks.

3D pipes with twists and turns
![Piping](https://github.com/FranzBangar/classy_examples/blob/master/showcase/piping.png?raw=true "Piping")

A single channel of an impeller, without volute and with infinitely thin blades:
![Impeller](https://github.com/FranzBangar/classy_examples/blob/master/showcase/impeller.png?raw=true "Impeller")

2D mesh for studying Karman Vortex Street
![Karman Vortex Street](https://github.com/FranzBangar/classy_examples/blob/master/showcase/karman.png?raw=true "Karman vortex street")

Helmholtz nozzle, a resonator with sharp edges. See [this sketch](https://www.researchgate.net/figure/Schematic-diagram-of-a-Helmholtz-oscillator-and-its-operating-principles_fig6_305275686).

![Helmholtz nozzle](https://github.com/FranzBangar/classy_examples/blob/master/showcase/resonator.png?raw=true "Helmholtz resonator")

A full volute and suction pipe (impeller mesh is created separately). This has a sharp cutwater that causes
a little unphysical overlap but numerically it is just fine and also quite accurate.
![Volute](https://github.com/FranzBangar/classy_examples/blob/master/showcase/volute.png?raw=true "Volute")

A real-life square volute with a blunt cutwater:
![Square Volute](https://github.com/FranzBangar/classy_examples/blob/master/showcase/volute_square.png?raw=true "Square Volute")