from classy_blocks.classes.operations import Face, Extrude
from classy_blocks.classes.mesh import Mesh

def get_mesh():
    base = Face(
        [ [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0] ],
        [ [0.5, -0.2, 0], None, None, None]
    )

    extrude = Extrude(base, [0.5, 0.5, 3])

    # direction 1
    extrude.chop(0, start_size=0.02, c2c_expansion=1.2, length_ratio=0.5, invert=False)
    extrude.chop(0, start_size=0.02, c2c_expansion=1.2, length_ratio=0.5, invert=True)

    # direction 2
    extrude.chop(1, start_size=0.02, c2c_expansion=1.2, length_ratio=0.5, invert=False)
    extrude.chop(1, start_size=0.02, c2c_expansion=1.2, length_ratio=0.5, invert=True)

    # extrude direction
    extrude.chop(2, c2c_expansion=1, count=20)

    mesh = Mesh()
    mesh.add(extrude)

    return mesh