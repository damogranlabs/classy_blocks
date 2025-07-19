from typing import ClassVar

from classy_blocks.cbtyping import IndexType, NPPointListType
from classy_blocks.util.constants import EDGE_PAIRS

ConnectionType = tuple[int, int]


def get_key(index_1: int, index_2: int):
    return (min(index_1, index_2), max(index_1, index_2))


class Connection:
    """Connects two points by their index in unique point list"""

    def __init__(self, index_1: int, index_2: int):
        self.indexes = get_key(index_1, index_2)

    def get_other_index(self, index):
        if self.indexes[0] == index:
            return self.indexes[1]

        return self.indexes[0]

    def __repr__(self):
        return f"Connection {self.indexes[0]}-{self.indexes[1]}"

    def __hash__(self):
        return hash(self.indexes)


class ConnectionRegistryBase:
    edge_pairs: ClassVar[list[ConnectionType]]

    def __init__(self, points: NPPointListType, addressing: list[IndexType]):
        self.points = points
        self.addressing = addressing

        self.connections: dict[ConnectionType, Connection] = {}
        self.nodes: dict[int, set[Connection]] = {i: set() for i in range(len(self.points))}

        for i in range(len(self.addressing)):
            cell_indexes = self.addressing[i]
            for pair in self.edge_pairs:
                index_1 = cell_indexes[pair[0]]
                index_2 = cell_indexes[pair[1]]

                edge_key = get_key(index_1, index_2)

                if edge_key not in self.connections:
                    connection = Connection(index_1, index_2)
                    self.connections[edge_key] = connection
                    self.nodes[index_1].add(connection)
                    self.nodes[index_2].add(connection)

    def get_connected_indexes(self, index: int) -> set[int]:
        return {c.get_other_index(index) for c in self.nodes[index]}


class QuadConnectionRegistry(ConnectionRegistryBase):
    edge_pairs: ClassVar = [(0, 1), (1, 2), (2, 3), (3, 0)]


class HexConnectionRegistry(ConnectionRegistryBase):
    edge_pairs: ClassVar = EDGE_PAIRS
