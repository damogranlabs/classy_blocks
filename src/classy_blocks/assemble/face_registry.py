from typing import List, Tuple

FaceType = Tuple[int, int, int, int]


def get_key(indexes: List[int]):
    return tuple(sorted(indexes))


class Face:
    def __init__(self, indexes: FaceType):
        self.indexes = indexes

    def __hash__(self):
        return hash(self.indexes)

    def __repr__(self):
        return f"Face {self.indexes}"
