import unittest

from classy_blocks.lists.geometry_list import GeometryList


class GeometryListTests(unittest.TestCase):
    def setUp(self):
        self.cone_data = [
            "type cone",
            "point1 (0 0 0)",
            "radius1 1",
            "innerRadius1 (0.5)",
            "point2 (1 0 0)",
            "radius2 0.4",
            "innerRadius2 0.3",
        ]

        self.sphere_data = [
            "type searchableSphere",
            "centre (0 0 0)",
            "radius 3",
        ]

        self.glist = GeometryList()

    def test_add_single(self):
        """Add two different geometries to the list"""
        self.glist.add({"cone": self.cone_data})

        self.assertEqual(len(self.glist.geometry), 1)

    def test_add_single_two(self):
        """Add two geometries one after the other"""
        self.glist.add({"cone": self.cone_data})
        self.glist.add({"sphere": self.sphere_data})

        self.assertEqual(len(self.glist.geometry), 2)

    def test_add_two(self):
        """Add two geometries at once"""
        self.glist.add({"cone": self.cone_data, "sphere": self.sphere_data})

        self.assertEqual(len(self.glist.geometry), 2)

    def test_description(self):
        self.glist.add({"sphere": self.sphere_data})
        description = self.glist.description

        for data in self.sphere_data:
            self.assertIn(f"{data};", description)

    def test_description_empty(self):
        """Ouput nothing when list is empty"""
        self.assertEqual(self.glist.description, "")
