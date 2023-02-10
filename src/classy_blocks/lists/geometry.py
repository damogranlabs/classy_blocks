
class GeometryList:
    """Handling of the 'geometry' part of blockMeshDict"""
    def __init__(self):
        self.geometry:dict = {}

    def add(self, geometry:dict) -> None:
        """Adds named entry in the 'geometry' section of blockMeshDict;
        'g' is in the form of dictionary {'geometry_name': [list of properties]};
        properties are as specified by searchable* class in documentation.
        See examples/advanced/project for an example."""
        # concatenate the two dictionaries
        self.geometry = {**self.geometry, **geometry}

    def output(self):
        """Formats a string to be inserted into blockMeshDict"""
        # nothing to output?
        if len(self.geometry.items()) == 0:
            return ""

        glist = "geometry\n{\n"

        for name, properties in self.geometry.items():
            glist += f"\t{name}\n\t{{\n"

            for prop in properties:
                glist += f"\t\t{prop};\n"

            glist += "\t}\n"

        glist += "};\n\n"

        return glist
