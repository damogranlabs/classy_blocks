# -*- coding: utf-8 -*-
import os, jinja2
import numpy as np
import scipy.optimize

from util import geometry as g
from util import tools, constants

# see README for terminology, terminolology, lol
class Point():
    """ vertex without an index (for, like, edges and stuff) """
    def __init__(self, coordinates):
        self.coordinates = coordinates
        
    @property
    def short_output(self):
        # output for edge definitions
        return "({} {} {})".format(self.coordinates[0], self.coordinates[1], self.coordinates[2]) 
    
    @property
    def long_output(self):
        # output for Vertex definitions
        return "({}\t{}\t{})".format(
            self.coordinates[0], self.coordinates[1], self.coordinates[2]
        )

    def __add__(self, other):
        return np.array(self.coordinates) + np.array(other.coordinates)

    def __sub__(self, other):
        return np.array(self.coordinates) - np.array(other.coordinates)


class Vertex():
    """ point with an index that's used in block and face definition """
    def __init__(self, point, index):
        self.point = Point(point)
        self.index = index
        
    @property
    def output(self):
        return self.point.long_output + " // {}".format(self.index)
    
    @property
    def printout(self):
        return "Vertex {} {}".format(self.index, self.point.short_output)
    
    def __repr__(self):
        return self.output

    def __str__(self):
        return self.output
        
    def __unicode__(self):
        return self.output


class Edge():
    def __init__(self, vertex_1, vertex_2, point, index):
        """ an edge is defined by two vertices and a point in between;
            all edges are treated as circular arcs
        """
        self.vertex_1 = vertex_1
        self.vertex_2 = vertex_2
        self.point = Point(point)

    @property
    def output(self):
        return "arc {} {} {}".format(
            self.vertex_1.index,
            self.vertex_2.index,
            self.point.short_output
        )

    @property
    def is_valid(self):
        # returns True if the edge is not an arc;
        # in case the three points are collinear, blockMesh
        # will find an arc with infinite radius and crash.
        OA = np.array(self.vertex_1.point.coordinates)
        OB = np.array(self.vertex_2.point.coordinates)
        OC = np.array(self.point.coordinates)

        # if point C is on the same line as A and B:
        # OC = OA + k*(OB-OA)
        AB = OB - OA
        AC = OC - OA

        k = g.norm(AC)/g.norm(AB)
        d = g.norm((OA+AC) - (OA + k*AB))

        return d > constants.tol

    def __repr__(self):
        return self.output
