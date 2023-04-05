from typing import Dict, List


class GeometryList:
    """Handling of the 'geometry' part of blockMeshDict"""

    def __init__(self) -> None:
        self.geometry: Dict[str, List[str]] = {}

    def add(self, geometry: dict) -> None:
        """Adds named entry in the 'geometry' section of blockMeshDict;
        'g' is in the form of dictionary {'geometry_name': [list of properties]};
        properties are as specified by searchable* class in documentation.
        See examples/advanced/project for an example."""
        # concatenate the two dictionaries
        self.geometry = {**self.geometry, **geometry}

    @property
    def description(self) -> str:
        """Formats a string to be inserted into blockMeshDict"""
        # nothing to output?
        if len(self.geometry.items()) == 0:
            return ""

        out = "geometry\n{\n"

        for name, properties in self.geometry.items():
            out += f"\t{name}\n\t{{\n"

            for prop in properties:
                out += f"\t\t{prop};\n"

            out += "\t}\n"

        out += "};\n\n"

        return out
