import dataclasses

import numpy as np

from classy_blocks.cbtyping import PointType, VectorType
from classy_blocks.util.constants import DTYPE, vector_format


def snake_to_camel(snake: str):
    camel = snake.split("_")
    return camel[0] + "".join(word.capitalize() for word in camel[1:])


@dataclasses.dataclass
class SearchableGeometry:
    label: str

    @property
    def kind(self) -> str:
        name = self.__class__.__name__
        return name[:1].lower() + name[1:]

    def get_data(self) -> dict[str, list[str]]:
        """Returns the data required to add the geometry to cb.Mesh"""
        data: list[str] = [
            f"type {self.kind}",
        ]

        for key, value in dataclasses.asdict(self).items():
            if key == "label":
                continue

            try:
                value = vector_format(np.asarray(value, dtype=DTYPE))
            except (ValueError, IndexError):
                value = str(value)

            data.append(f"{snake_to_camel(key)} {value}")

        return {self.label: data}


@dataclasses.dataclass
class SearchablePlanePointAndNormal(SearchableGeometry):
    base_point: PointType
    normal_vector: VectorType
    plane_type: str = dataclasses.field(init=False)
    base: PointType = dataclasses.field(init=False)
    normal: PointType = dataclasses.field(init=False)

    @property
    def kind(self):
        return "searchablePlane"

    def __post_init__(self):
        # duplicate keywords to be compatible with both OF branches
        # return {
        #     self.geometry_label: [
        #         "type searchableSphere",
        #         f"origin {constants.vector_format(self.center_point)}",
        #         f"centre {constants.vector_format(self.center_point)}",
        #         f"radius {self.radius}",
        #     ]
        # }
        #
        self.base = self.base_point
        self.normal = self.normal_vector
        self.plane_type = "pointAndNormal"


@dataclasses.dataclass
class SearchablePlaneEmbeddedPoints(SearchableGeometry):
    point_1: PointType
    point_2: PointType
    point_3: PointType
    plane_type: str = dataclasses.field(init=False)

    @property
    def kind(self):
        return "searchablePlane"

    def __post_init__(self):
        self.plane_type = "embeddedPoints"


@dataclasses.dataclass
class SearchableSphere(SearchableGeometry):
    origin: PointType
    radius: float
    centre: PointType = dataclasses.field(init=False)

    def __post_init__(self):
        self.centre = self.origin


@dataclasses.dataclass
class SearchableCylinder(SearchableGeometry):
    point_1: PointType
    point_2: PointType
    radius: float


@dataclasses.dataclass
class SearchableCone(SearchableGeometry):
    point_1: PointType
    radius_1: float
    inner_radius_1: float
    point_2: PointType
    radius_2: float
    inner_radius_2: float


@dataclasses.dataclass
class TriSurface(SearchableGeometry):
    file: str

    @property
    def kind(self):
        return "triSurface"

    def __post_init__(self):
        self.file = '"' + self.file + '"'
