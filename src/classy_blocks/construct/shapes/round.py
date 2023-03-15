import abc
from typing import Type, Callable, List, TypeVar

from classy_blocks.types import AxisType, OrientType

from classy_blocks.construct.edges import Arc
from classy_blocks.construct.shapes.shape import Shape
from classy_blocks.construct.flat.circle import Circle
from classy_blocks.construct.flat import circle
from classy_blocks.construct.operations.operation import Operation
from classy_blocks.construct.operations.loft import Loft

ShapeT = TypeVar('ShapeT', bound='RoundShape')

class RoundShape(Shape, abc.ABC):
    """An object, lofted between 2 or more sketches;
    to form blocks, sketches are transformed with specified
    functions (and so are side edges) and loft operations
    are created between.
    
    Solid round shapes: Elbow, Frustum, Cylinder;
    they are created using an OH-grid (see Circle), have a
    'start' and 'end' sketch and an 'outer' surface."""
    sketch_class = Circle # Sketch class to be used for construction of the Shape

    axial_axis:AxisType = 2 # Axis along which 'outer sides' run
    radial_axis:AxisType = 0 # Axis that goes from center to 'outer side'
    tangential_axis:AxisType = 1 # Axis that goes around the circumference of the shape"""

    start_patch:OrientType ='bottom' # Sides of blocks that define the start patch
    end_patch:OrientType = 'top' # Sides of blocks that define the end patch"""
    outer_patch:OrientType = 'right' # Sides of blocks that define the outer surface

    def __init__(self, args_1, transform_2_args, transform_mid_args=None):
        # start with sketch_1 and transform it
        # using self.transform_function(transform_2_args) to obtain sketch_2;
        # use self.transform_function(transform_mid_args) to obtain mid sketch
        # (only if applicable)
        self.sketch_1 = self.sketch_class(*args_1)
        self.sketch_2 = self.transform_function(**transform_2_args)

        # TODO: TEST
        if transform_mid_args is not None:
            self.sketch_mid = self.transform_function(**transform_mid_args)
        else:
            self.sketch_mid = None

        self.operations = []

        for i, face_1 in enumerate(self.sketch_1.faces):
            face_2 = self.sketch_2.faces[i]

            loft = Loft(face_1, face_2)

            # add edges, if applicable
            if self.sketch_mid:
                face_mid = self.sketch_mid.faces[i]

                for i, point in enumerate(face_mid.points):
                    loft.add_side_edge(i, Arc(point))

            self.operations.append(loft)

    @abc.abstractmethod
    def transform_function(self, **kwargs) -> Callable:
        """A function that transforms sketch_1 to sketch_2;
        a Loft will be made from those"""

    @property
    def core(self) -> List[Operation]:
        """Operations in the center of the shape"""
        return self.operations[:len(self.sketch_1.core)]

    @property
    def shell(self) -> List[Operation]:
        """Operations on the outside of the shape"""
        return self.operations[len(self.sketch_1.core):]

    def chop_axial(self, **kwargs):
        """Chop the shape between start and end face"""
        self.operations[0].chop(self.axial_axis, **kwargs)

    def chop_radial(self, **kwargs):
        """Chop the outer 'ring', or 'shell';
        core blocks will be defined by tangential chops"""
        # scale all radial sizes to this ratio or core cells will be
        # smaller than shell's
        c2s_ratio = max(circle.CORE_DIAGONAL_RATIO, circle.CORE_SIDE_RATIO)
        if "start_size" in kwargs:
            kwargs["start_size"] *= c2s_ratio
        if "end_size" in kwargs:
            kwargs["end_size"] *= c2s_ratio

        self.shell[0].chop(self.radial_axis, **kwargs)

    def chop_tangential(self, **kwargs):
        """Circumferential chop; also defines core sizes"""
        for i in (0, 1, 2, -1): # minimal number of blocks that need to be set
            self.shell[i].chop(self.tangential_axis, **kwargs)

    def set_start_patch(self, name:str) -> None:
        """Assign the faces of start sketch to a named patch"""
        for operation in self.operations:
            operation.set_patch(self.start_patch, name)

    def set_end_patch(self, name:str) -> None:
        """Assign the faces of end sketch to a named patch"""
        for operation in self.operations:
            operation.set_patch(self.end_patch, name)
    
    def set_outer_patch(self, name:str) -> None:
        """Assign the faces of end sketch to a named patch"""
        for operation in self.shell:
            operation.set_patch(self.outer_patch, name)
