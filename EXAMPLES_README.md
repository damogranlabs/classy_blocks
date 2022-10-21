# `classy_block` examples
Example project, cases and tests for `classy_blocks`.

## Usage
1. Prerequisites
    1. OpenFOAM (just about any version)
    2. Prepare python virtual environment:
        $ python -m venv venv
        $ "venv/bin/activate"
        $ python -m pip install -U pip
        $ python -m pip install -r requirements_dev.txt
        $ python -m pip install -e .
2. Run `examples/run.py`
3. Open `examples/case/case.foam` with ParaView to inspect the mesh.

## Showcase
These are some screenshots of parametric models, built with classy_blocks.

Rectangular ducts (Extrude and Revolve Operations)
![Ducts](showcase/elbows.png "Ducts")

3D pipes with twists and turns (Elbow and Cylinder Shapes)
![Piping](showcase/piping.png "Piping")

A simple tank with rounded edges
![Tank](showcase/tank.png "Tank")

A flywheel in a case. VTK Blocking output for debug is shown in the middle
![Flywheel](showcase/flywheel.png "Flywheel")

Venturi tube
![Venturi tube](showcase/venturi_tube.png "Venturi tube")

Coriolis flowmeter with meshed fluid (blue) and solid section (white), ready for an FSI simulation.
![Coriolis Flowmeter](showcase/coriolis_flowmeter.png "Coriolis Flowmeter")

2D mesh for studying Karman Vortex Street
![Karman Vortex Street](showcase/karman.png "Karman vortex street")

Helmholtz nozzle, a resonator with sharp edges. See [this sketch](https://www.researchgate.net/figure/Schematic-diagram-of-a-Helmholtz-oscillator-and-its-operating-principles_fig6_305275686).
![Helmholtz nozzle](showcase/resonator.png "Helmholtz resonator")

A real-life square volute with a blunt cutwater
![Square Volute](showcase/volute_square.png "Square Volute")

Edges and faces, projected to an STL surface
![Projected](showcase/projected.png "Projected edges and faces")

Mesh for studying flow around a sphere, with projected edges and faces
![Sphere](showcase/sphere.png "Flow around a sphere")

A parametric, Low-Re mesh of a real-life impeller *(not included in examples)*
![Impeller - Low Re](showcase/impeller_full.png "Low-Re Impeller")
