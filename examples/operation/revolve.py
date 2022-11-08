from classy_blocks import Face, Revolve, Mesh

from classy_blocks.util import functions as f

def get_mesh():
    base = Face(
        [ [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0] ],
        [ [0.5, -0.2, 0], None, None, None]
    )

    revolve = Revolve(base, f.deg2rad(60), [0, -1, 0], [-2, 0, 0])

    # a shortcut for setting count only
    revolve.chop(0, count=10)
    revolve.chop(1, count=10)
    revolve.chop(2, count=30)

    mesh = Mesh()
    mesh.add(revolve)

    return mesh
