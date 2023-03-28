import unittest

from classy_blocks.mesh import Mesh

class MeshTests(unittest.TestCase):
    def setUp(self):
        self.mesh = Mesh()

    def test_settings_output(self):
        """Proper formatting of settings"""
        self.mesh.settings['prescale'] = 1
        self.mesh.settings['scale'] = 0.001
        self.mesh.settings['mergeType'] = 'points'

        expected = "prescale 1;\nscale 0.001;\nmergeType points;\n\n"

        self.assertEqual(self.mesh.format_settings(), expected)
