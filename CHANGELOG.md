# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.9.3]

### Added

- `trust-constr` optimization algorithm
- pass arbitrary options to `scipy.optimize.minimize`

### Changed

- Bugfix: always use grid quality when checking for rollback

## [1.9.2]

- Bugfix: mixed grid and junction qualities

## [1.9.1]

- Bugfix: multiple projections to the same surface

### Removed

- Mapper class (not exposed as a public API) and replaced with lookup classes

## [1.9.0]

### Added

- Optimization refactor: Optimizers now have a `.config` attribute
that the user can change for fine adjustments
- New classy_blocks logo

### Changed

- Optimization refactor: quality functions are now more reliable and yield better quality once finished

### Removed

- Removed Python 3.8 support

## [1.8.0]

### Added

- Hex*/QuadGrid objects now support `merge_tol` for merging approximately coincident points

### Changed

- Improvement on performance of mesh assembly (5x+)
- Improvement on performance of optimization (20x+)
- Minor API change: use `mesh.settings.property` instead of `mesh.settings['property']`
- Chopping propagation now doesn't overwrite already defined wires within the same block; it's possible to chop two surrounding blocks and the middle one will have different gradings on appropriate wires
- Bugfixes:
  - Correct center point of OneCoreDisk
  - Correct inner angle calculation for optimization

## [1.7.1]

### Changed

- Bugfix: wrong grading propagation on inverted wires

## [1.7.0]

### Added

- Automatic grading:
  - `FixedCountGrader`: simple and quick, grades all blocks in all directions with the same number. Useful while developing meshes - settings blocking etc.
  - `SimpleGrader`: quick and good for cases where blocks do not differ much in size. Sets simple-graded counts on blocks based on wanted cell size.
  - `SmoothGrader`: Tries to stretch and squeeze cells within blocks by using two graded chops in each direction. The idea is to try to keep difference in cell size between neighbouring blocks as little as possible. Blocks that cannot maintain required cell size are simply/uniformly chopped.
- Possibility to define rings by their inner radius
- Spline round shapes
- Cylinders:
  - Add symmetry patch to SemiCylinder
- New examples:
  - Quarter cylinder
  - One core cylinder
- `ShapeOptimizer` can optimize shapes _before_ they are added  to mesh

### Changed

- Renamed `classy_blocks.typing` module to `classy_blocks.cbtyping` due to name clash
- Bugfixes

## [1.6.4]

### Added

- MappedSketch.merge() method

### Changed

- Improved blocking in cyclone example

## [1.6.3]

### Changed

- Bugfix: sorting of cells by sensitivity
- Bugfix: sensitivity calculation moved clamps around
- Output of optimization reports
- Optimization will now skip cells that were made illegal during optimization

### Removed

- Quality caching (produced invalid results and did not speed up optimization)

## [1.6.2]

### Added

- QuarterSplineRound and QuarterSplineRoundRing; cylinders and rings with an arbitrary, parametrized spline cross-section
- Lofts and Shapes, created with spline side-edges (when multiple sketches for mid-sections are provided)
- Examples for splined shapes (as above)

### Changed

- Bugfixes

## [1.6.1]

### Added

- Assemblies and *Joints
  - Examples thereof

### Changed

- Some bugfixes

## [1.6.0]

### Added

- Array element; handling multiple points at once (makes transforms faster)
- Complete overhaul of Optimization:
  - Optimizer has become SketchOptimizer or MeshOptimizer
  - MappedSketch is now smoothed by SketchSmoother
  - Mesh can also be smoothed by MeshSmoother
- Gear example

### Changed

- Cell quality: adjusted calculation so that it works for quadrangles and hexahedrons
- Clamps no longer refer to Vertex objects but only store points as locations
- Links same as clamps above, locations only

### Removed

- QuadMap is no longer needed (handled by Grid classes)

## [1.5.3]

### Added

- RoundSolidShape.remove_inner_edges() can now remove edges from a specific face (start, end or both)

### Changed

- Bugfix: invalid cells in round shapes (due to wrong spline point calculation)

## [1.5.2]

### Added

- Spline edges on QuarterDisk, HalfDisk and Disk (FourCoreDisk) to improve mesh quality
- Face.remove_edges() to reset all edges to simple lines
- RoundSolidShape.remove_inner_edges() to get rid of splines in case the vertices need to be moved (in optimization, for example)

### Changed

- Updated affected tutorials (diffusers, cyclone)

## [1.5.1]

### Added

- Optimization:
  - Choice of solver (different problems require - work best - with different solvers)
  - Richer output (timing, relative improvement) for easier choice of solver
- Curves:
  - get_param_at_length()
  - Direct imports of various classes
- A new example: custom sketch
- Mesh:
  - assemble(): an option to skip creating edges; most useful when assembling mesh before optimization but later a backport() will be called

### Changed

- Improved optimization speed

## [1.5.0]

### Added

- Curves:
  - get_tangent()
  - get_normal()
  - get_binormal()
- Mapped sketch
- Definition of any fixed-blocking sketch
- Automatic laplacian smoothing (fixed outer edges, movable inner vertices)
- Sketches
- OneCoreDisk
- FourCoreDisk (the default Disk for Shapes)
- WrappedDisk
- Oval
- WrappedDisk
- Grid (A cartesian array of rectangular faces in XY plane)
- grid property of Sketch/Shape/Stack
- Stacks
- LoftedShape: a generic shape from 2 Sketches with the same number of faces
- Examples: Heater, Fusilli
- Mesh.delete() will omit given operation from blockMeshDict but its data stays in mesh (chops, patches, etc.)

### Changed

- Definitions of Disks sketches
- All off-the-shelf Shapes are now a LoftedShape
- Calling .transform() with a Mirror transformation will warn about creating an inverted block

## [1.4.1]

### Changed

- Improved optimization output

### Removed

- Relaxation within optimization
- Numerical integration of analytic curve lengths; use discretization instead

## [1.4.0]

### Added

- Channel example
- Cyclone example
- Mirror transform on points, operations, shapes, mirror example
- Operation:
  - `get_closest_face()`, `get_closest_side()`, `get_normal_face()`
  - Connector operation
- Geometric finders: find_on_plane()
- functions.point_to_line_distance()

### Changed

- Optimization improvements:
  - Default parameters for clamp optimization
  - Clamps are sorted by sensitivity, not "junction quality" as before (improves optimization speed)
  - Clamp parameters follow domain scale (Read more below)
  - Raise an Exception when adding more than one Clamp for the same vertex

#### Clamp Parameters

Previously:

- RadialClamp had a single parameter, the _angle_ of the point (and change thereof)
- In Linelamp the parameter _t_ went from 0 to 1 regardless of the distance between points

This created difficulties with optimization algorithms with extra large or very small domains.
Optimization speed also drastically changing with simply scaling the dimensions.

This has been changed:

- RadialClamp's parameter is now multiplied with radius so it means actual _distance_
- LineClamp's parameter now goes from 0 to _distance between points_
When working with CurveClamps, this kind of _automatic_ correction cannot be made so it is advisable
that parameter is of a similar magnitudes than points' coordinates.

### Removed

- Junction.delta() is now handled by optimization automatically

## [1.3.3]

### Added

- Airfoil example
- Translation and Rotational Link: move together with vertices being optimized (see the airfoil example)

### Changed

- Renamed ParametricCurveClamp to CurveClamp (takes Curve object of any kind)
- Interpolated curves' indexes are now between 0 and 1 (easier to work with than using len(points) every time)
- Optimization driver:
  - Termination tolerance is now based on initial improvement instead of quality
  - Relaxation starts at 0.5 by default and increases linearly to 1 in a given number of relaxed iterations

### Removed

- Curve.get_closest_param() now finds initial_param automatically

## [1.3.2] Curves

### Added

- *Curve objects for dealing with edge specification and optimization

## [1.3.1] Optimization/Backport

### Added

- mesh.clear() removes lists of all items that were populated during mesh.assemble()
- mesh.backport() updates user supplied operations' points with results of optimization/modification

### Changed

- Optimizer: under-relaxation for the first optimization iterations

## [1.3.0] Blocking Optimization

### Added

- **Blocking Optimization**
  - Finders for easier fetching vertices, generated by mesh.assemble()
    - GeometricFinder lists vertices inside searchable geometric entities
    - RoundSolidFinder identifies vertices on core/shell of round solids
  - An Optimizer class that handles blocking optimization
  - Clamp classes that define degrees of freedom of optimizing points:
    - Free (3 DoF)
    - Slide along a curve (line, parametric curve) (1 DoF)
    - Move on a surface (parametric surface) (2 DoF)
- **Reorienting Operations and Faces**
  - `Face`:
    - `shift()` method to rotate points around
    - `reorient()` method that rotates points so that they start nearest to given position
  - `ViewpointOrienter`: a class for auto-orienting operation's points based on specified points 'in front' and 'above' the operation.

### Changed

- Projection Behaviour
  - Calling .project() on a Point/Vertex will add the new geometry instead of replacing it.
  - Calling .project_edge() on an Operation will add the new geometry instead of replacing it.
  - Calling .project_side() on an Operation will add the new geometry to edges instead of replacing them (but will replace existing label for the side)

## [1.2.0] Shell Shape

### Added

- A Shell Shape, created from arbitrary set of faces

### Changed

- Operation.get_face() will not auto-reorder faces (causes confusion for users)

### Fixed

- A bug where Operation.project_face(..., points=True) won't project vertices

## [1. 1. 0] Default Extrude Direction

### Added

- Extrude now takes a vector of a float. If a float is given, direction is normal of the base face.

## [1. 0. 0] Refactor

A complete overhaul of all objects in an attempt to create a proper SOLID-obeying
package with type hinting, static typing and no python-ish duck-typing hacks.

### Added

- examples and showcases from `classy_examples` repo
- static type analysis, formatting, linting
- Origin and Angle edges (Foundation and ESI alternatives to arc)
- Projection of vertices to geometry
- Import convention `import classy_blocks as cb` and direct imports of user-usable objects from `cb`, like `cb.Mesh, cb.Loft, cb.Arc`
- `Operation.faces` property that creates new faces on-the-fly for easier chaining of new operations
- A Frame object that simplifies addressing edges/wires/other stuff between pairs of vertices on a hexahedron
- ExtrudedRing.fill() method has been added to create cylinders inside rings

### Changed

- Major package layout refactor
- Edge specification (Arc, Origin, Angle, Project, Spline, PolyLine objects)
- Reverted Face specification for operations
- The Block object is not directly available to the user as it makes no sense to do so
- Import convention: `import classy_blocks as cb` for examples
- Changed examples so that an example file runs directly instead of calling run.py (that created a lot of confusion)
- Box() is now an operation (previously Shape)
- simplified cylinder and sphere creation
- Chaining of elbows/cylinders/etc always with start_face parameters (instead of negative length)

### Removed

- *Wall shapes will be created later with a different approach
- Examples with *Wall shapes will be recreated later with new approaches
- airfoil2d example requires blocking optimization so it will be recreated when that feature is available
- sphere example will be recreated when 'offset' is available
- block.from_points has been removed (use Loft)
- T-joint will be added when a skew transform is implemented

## [0.0.1] Status Quo

### Added

examples and showcases from classy_examples repo
static type analysis, formatting, linting

### Changed

Major package layout refactor
Major CI refactor

### Removed

Some dependencies, docs and other stuff that is not ready ATM
