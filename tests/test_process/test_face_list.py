from tests.fixtures import FixturedTestCase

class FaceListTests(FixturedTestCase):
    def test_block_project_face(self):
        self.mesh.blocks[0].project_face("bottom", "terrain")
        self.mesh.blocks[0].project_face("left", "building")

        self.mesh.prepare()

        formatted_faces = self.mesh.faces.output()

        self.assertTrue("project (0 1 2 3) terrain" in formatted_faces)
        self.assertTrue("project (4 0 3 7) building" in formatted_faces)