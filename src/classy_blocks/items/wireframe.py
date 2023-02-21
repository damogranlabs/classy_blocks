from typing import List, Dict, Tuple

from classy_blocks.data.chop import AxisType, AxisChop, EdgeChop

from classy_blocks.items.vertex import Vertex
from classy_blocks.items.edges.edge import Edge
from classy_blocks.items.wire import Wire

from classy_blocks.util import constants

class Wireframe:
    """A handler for storing and retrieving pairs of vertices a.k.a. 'wires';
    Numbers of vertices, axes and everything else according to this sketch
    https://www.openfoam.com/documentation/user-guide/4-mesh-generation-and-conversion/4.3-mesh-generation-with-the-blockmesh-utility
    
    This object can be indexed so that the desired Wire can be accessed directly;
    for instance, an edge between vertices 2 and 6 is obtained with frame[2][6].edge"""
    def __init__(self, vertices:List[Vertex], edges:List[Edge]):
        self.wires:List[Wire] = []
        # the opposite side of each vertex
        self.couples:List[Dict[int, Wire]] = [{} for _ in range(8)]
        # wires of each axis
        self.axes:List[List[Wire]] = [[] for _ in range(3)]

        # create wires and connections for quicker addressing
        for axis in range(3):
            for pair in constants.AXIS_PAIRS[axis]:
                wire = Wire(vertices, axis, pair[0], pair[1])

                self.couples[pair[0]][pair[1]] = wire
                self.couples[pair[1]][pair[0]] = wire

                self.axes[axis].append(wire)

                self.wires.append(wire)

        # reverse-engineer corners of created edges
        # (they were in EdgeData but have been dropped and
        # this is ugly but better than passing that data around forever)
        # TODO: prettify somehow
        def find_corner(vertex:Vertex) -> int:
            for i, edge_vertex in enumerate(vertices):
                if edge_vertex == vertex:
                    return i

            raise RuntimeError("Edge is defined by different vertices")

        for edge in edges:
            corner_1 = find_corner(edge.vertex_1)
            corner_2 = find_corner(edge.vertex_2)

            self.couples[corner_1][corner_2].edge = edge

    def get_axis_wires(self, axis:AxisType) -> List[Wire]:
        """Returns a list of wires that run in the given axis"""
        return self.axes[axis]

    def chop_axis(self, axis:AxisType, chops:List[AxisChop]) -> None:
        """Creates Gradings from specified Axis chops"""
        for wire in self.get_axis_wires(axis):
            for chop in chops:
                for div in chop.divisions:
                    wire.grading.add_division(div)

    def broadcast_grading(self, axis:AxisType) -> bool:
        """Takes one wire with grading and copies it to all other wires
        in given axis;

        Returns:
        - False if nothing was done (no defined gradings/all already defined)
        - True if the method succeeded

        Raises:
        - RuntimeError if there is more than one grading and they are not the same"""
        wires = self.get_axis_wires(axis)
        defined = [wire.grading.is_defined for wire in wires]
        defined_count = sum(defined)

        if defined_count == 0:
            # there's nothing to copy
            return False

        if defined_count > 1:
            # check that counts match (but gradings can be different, eh?)
            if not all(w.grading.count == wires[0].grading.count for w in wires):
                message = f"Multiple different gradings are defined between vertices {wires}"
                raise RuntimeError(message)

        if defined_count == 4:
            # everything is defined already, also checked for consistency
            return False

        # choose the first defined grading and copy it to all other wires
        defined_index = defined.index(True)
        grading = wires[defined_index].grading

        for i, wire in enumerate(wires):
            if i == defined_index:
                continue

            wire.grading = grading.copy(invert=False)

        return True

    def pull_gradings(self) -> bool:
        """Traverses all wires and copies gradings from neighbours;
        Returns True when something has been copied and False when
        nothing changed"""
        changed = False

        for wire in self.wires:
            if wire.grading.is_defined:
                # no need to copy anything
                continue

            for coincident in wire.coincidents:
                # TODO: add checking for inconsistent counts
                if coincident.grading.is_defined:
                    wire.grading = coincident.grading.copy(wire.is_aligned(coincident))
                    changed = True
                    # continue with another wire
                    break

        return changed

    def __getitem__(self, index) -> Dict[int, Wire]:
        return self.couples[index]
