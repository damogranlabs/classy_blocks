import dataclasses
from typing import Dict, List

from parameters import (
    D_BODY,
    D_CONE,
    D_INLET,
    D_OUTLET,
    DIM_SCALE,
    L_BODY,
    L_CONE,
    L_INLET,
    L_OUTLET_IN,
    L_OUTLET_OUT,
    T_PIPE,
)

from classy_blocks.base.exceptions import GeometryConstraintError
from classy_blocks.cbtyping import NPPointType
from classy_blocks.util import functions as f
from classy_blocks.util.constants import vector_format as fvect


@dataclasses.dataclass
class Geometry:
    """Holds user-provided parameters and conversions to SI units;
    simple relations and shortcuts for easier construction"""

    def __post_init__(self):
        # constraints check
        if self.r["inlet"] >= self.r["body"] / 2:
            raise GeometryConstraintError("Inlet must be smaller than body/2")

        if self.z["skirt"] <= self.z["upper"]:
            raise GeometryConstraintError("Outlet inside is too short")

    @property
    def r(self) -> Dict[str, float]:
        """Radii as defined in parameters"""
        return {
            "inlet": DIM_SCALE * D_INLET / 2,
            "outlet": DIM_SCALE * D_OUTLET / 2,
            "body": DIM_SCALE * D_BODY / 2,
            "pipe": DIM_SCALE * (D_OUTLET / 2 + T_PIPE),
            "cone": DIM_SCALE * D_CONE / 2,
        }

    @property
    def l(self) -> Dict["str", "float"]:  # noqa: E743
        """Lengths that matter"""
        skirt = 0.1 * self.r["inlet"]
        l_body = DIM_SCALE * L_BODY - skirt - self.r["inlet"]
        l_outlet_in = DIM_SCALE * L_OUTLET_IN
        upper = l_outlet_in - self.r["inlet"] - skirt

        return {
            "inlet": DIM_SCALE * L_INLET,
            "outlet": DIM_SCALE * (L_OUTLET_IN + L_OUTLET_OUT),
            "skirt": skirt,
            "upper": upper,
            "lower": l_body - upper,
            "body": l_body,
            "cone": DIM_SCALE * L_CONE,
            "pipe": T_PIPE * DIM_SCALE,  # pipe thickness
        }

    @property
    def inlet(self) -> List[NPPointType]:
        point_2 = f.vector(0, -self.r["body"] + self.r["inlet"], 0)
        point_0 = point_2 + f.vector(-self.l["inlet"], 0, 0)
        point_1 = point_2 + f.vector(-self.r["body"] * 1.1, 0, 0)

        return [point_0, point_1, point_2]

    @property
    def z(self) -> Dict[str, float]:
        """z-coordinates of cyclone parts"""
        z_in_sk = self.r["inlet"] + self.l["skirt"]
        return {
            "skirt": -z_in_sk,
            "upper": -z_in_sk - self.l["upper"],
            "lower": -self.l["body"] - z_in_sk,
            "cone": -self.l["body"] - self.l["cone"] - z_in_sk,
        }

    @property
    def surfaces(self):
        """Returns definitions of searchable geometries for projections"""
        delta_in = f.vector(-2 * self.l["inlet"], 0, 0)
        p_body = f.vector(0, 0, -2 * self.z["cone"])

        def cylinder(name, point_1, point_2, radius):
            return {
                name: [
                    "type searchableCylinder",
                    f"point1 {fvect(point_1)}",
                    f"point2 {fvect(point_2)}",
                    f"radius {radius}",
                ],
            }

        return {
            **cylinder("inlet", self.inlet[0] - delta_in, self.inlet[2] + delta_in, self.r["inlet"]),
            **cylinder("body", p_body, -p_body, self.r["body"]),
            **cylinder("outlet", p_body, -p_body, self.r["outlet"]),
            **cylinder("pipe", p_body, -p_body, self.r["pipe"]),
            "cone": [
                "type searchableCone",
                f"point1 {fvect([0, 0, self.z['lower']])}",
                f"radius1 {self.r['body']}",
                "innerRadius1 0",
                f"point2 {fvect([0, 0, self.z['cone']])}",
                f"radius2 {self.r['cone']}",
                "innerRadius2 0",
            ],
        }


geometry = Geometry()
