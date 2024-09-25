from classy_blocks.mesh import Mesh


class Probe:
    """Examines the mesh and gathers required data for auto chopping"""

    def __init__(self, mesh: Mesh):
        self.mesh = mesh
