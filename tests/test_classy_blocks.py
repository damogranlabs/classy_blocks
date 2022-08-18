#!/usr/bin/env python

"""Tests for `classy_blocks` package."""

import pytest


@pytest.fixture
def response():
    """Sample pytest fixture.

    See more at: http://doc.pytest.org/en/latest/fixture.html
    """
    # import requests
    # return requests.get('https://github.com/audreyr/cookiecutter-pypackage')


def test_content(response):
    """Sample pytest test function with the pytest fixture as an argument."""
    # from bs4 import BeautifulSoup
    # assert 'GitHub' in BeautifulSoup(response.content).title.string


def test_get_mesh():
    from classy_blocks.classes.mesh import Mesh
    from classy_blocks.classes.shapes import Cylinder

    axis_point_1 = [0, 0, 0]
    axis_point_2 = [5, 5, 0]
    radius_point_1 = [0, 0, 2]

    def get_mesh():
        cylinder = Cylinder(axis_point_1, axis_point_2, radius_point_1)

        cylinder.set_bottom_patch('inlet')
        cylinder.set_top_patch('outlet')
        cylinder.set_outer_patch('walls')

        bl_thickness = 0.05
        core_size = 0.2

        cylinder.chop_axial(count=30)
        cylinder.chop_radial(start_size=core_size, end_size=bl_thickness)
        cylinder.chop_tangential(start_size=core_size)

        mesh = Mesh()
        mesh.add(cylinder)

        return mesh

    mesh = get_mesh()
