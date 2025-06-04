from classy_blocks.base.exceptions import BlockNotFoundError, NoInstructionError
from classy_blocks.construct.shapes.cylinder import Cylinder
from classy_blocks.grading.autograding.catalogue import Instruction, RowCatalogue
from classy_blocks.mesh import Mesh
from tests.fixtures.block import BlockTestCase


class InstructionTests(BlockTestCase):
    def setUp(self):
        self.instruction = Instruction(self.make_block(0))

    def test_not_defined(self):
        self.instruction.directions[1] = True
        self.assertFalse(self.instruction.is_defined)

    def test_defined(self):
        self.instruction.directions = [True] * 3
        self.assertTrue(self.instruction.is_defined)

    def test_hash(self):
        _ = {0: self.instruction}


class RowCatalogueTests(BlockTestCase):
    def setUp(self):
        self.mesh = Mesh()

        self.mesh.add(Cylinder([0, 0, 0], [1, 0, 0], [0, 1, 0]))
        self.mesh.assemble()

        self.catalogue = RowCatalogue(self.mesh)

    def test_row_blocks_exception(self):
        with self.assertRaises(BlockNotFoundError):
            # Try to find a block that's not a part of this mesh
            self.catalogue.get_row_blocks(self.make_block(1), 0)

    def test_find_instruction_exception(self):
        with self.assertRaises(NoInstructionError):
            self.catalogue._find_instruction(self.make_block(1))
