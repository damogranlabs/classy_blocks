import unittest

from classy_blocks.grading.graders.inflation import (
    BufferLayer,
    BulkLayer,
    InflationLayer,
    InflationParams,
    LayerStack,
    sum_length,
)


class ParamsTests(unittest.TestCase):
    def setUp(self):
        self.params = InflationParams(0.002, 0.1, 1.2, 30, 2)

    def test_bl_thickness(self):
        self.assertEqual(self.params.bl_thickness, 0.06)

    def test_inflation_count(self):
        self.assertEqual(self.params.inflation_count, 11)

    def test_inflation_end_size(self):
        self.assertAlmostEqual(self.params.inflation_end_size, 0.012383, places=5)

    def test_buffer_start_size(self):
        self.assertAlmostEqual(self.params.buffer_start_size, 0.012383 * 2, places=5)

    def test_buffer_count(self):
        self.assertEqual(self.params.buffer_count, 3)

    def test_sum_length_simple(self):
        self.assertAlmostEqual(sum_length(0.1, 10, 1), 1)

    def test_sum_length_graded(self):
        self.assertAlmostEqual(sum_length(0.1, 10, 2), 102.3)

    def test_buffer_thickness(self):
        self.assertAlmostEqual(self.params.buffer_thickness, 0.173369, places=5)


class LayerStackTests(ParamsTests):
    def test_inflation_layer_count(self):
        layer = InflationLayer(self.params, 0.6)

        self.assertEqual(layer.count, 12)

    def test_inflation_layer_size(self):
        layer = InflationLayer(self.params, 0.6)

        self.assertLess(layer.length, self.params.bl_thickness + self.params.buffer_start_size)

    def test_inflation_layer_size_truncated(self):
        layer = InflationLayer(self.params, 0.05)

        self.assertLess(layer.length, 0.05 + self.params.buffer_start_size)

    def test_inflation_count_zero(self):
        layer = InflationLayer(self.params, 0)

        self.assertEqual(layer.count, 1)

    def test_buffer_count(self):
        layer = BufferLayer(self.params, 1)

        self.assertEqual(layer.count, 4)

    def test_buffer_length_truncated(self):
        max_length = 1 - self.params.bl_thickness
        layer = BufferLayer(self.params, max_length)

        self.assertLess(layer.length, max_length + self.params.bulk_cell_size)

    def test_buffer_length(self):
        layer = BufferLayer(self.params, 100)

        self.assertLess(layer.length, 100)

    def test_bulk_layer(self):
        layer = BulkLayer(self.params, 1)

        self.assertEqual(layer.count, 11)

    def test_stack_inflation_short(self):
        # block size is less than boundary layer thickness
        stack = LayerStack(self.params, 0.5 * self.params.bl_thickness)

        self.assertEqual(len(stack.layers), 1)

    def test_stack_inflation_exact(self):
        # block size is exactly boundary layer size;
        stack = LayerStack(self.params, self.params.bl_thickness)

        self.assertEqual(len(stack.layers), 1)

    def test_stack_buffer(self):
        stack = LayerStack(self.params, self.params.bl_thickness + 2 * self.params.buffer_start_size)

        self.assertEqual(len(stack.layers), 2)

    def test_stack_bulk(self):
        stack = LayerStack(self.params, 10)

        self.assertEqual(len(stack.layers), 3)

    def test_length_ratios(self):
        stack = LayerStack(self.params, 10)

        self.assertAlmostEqual(sum(layer.length_ratio for layer in stack.layers), 1)
