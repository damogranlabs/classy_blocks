from classy_blocks import Face, Loft, Mesh

def get_mesh():
    # Example geometry using Loft:
    bottom_face = Face(
        [ # vertices
            [0, 0, 0], # 0
            [1, 0, 0], # 1
            [1, 1, 0], # 2
            [0, 1, 0]  # 3
        ],
        [
            [0.5, -0.25, 0], # edge between 0 and 1
            None,            # 1 and 2
            [0.5, 1.25, 0],  # 2 and 3
            None             # 3 and 0
        ]
    )

    top_face = Face(
        [ #
            [0, 0, 2], # 4
            [1, 0, 2], # 5
            [1, 1, 2], # 6
            [0, 1, 2]  # 7
        ],
        [
            None,
            [1.25, 0.5, 2],
            None,
            [-0.25, 0.5, 2]
        ]
    )

    side_edges = [
        [ [0.1, 0.1, 0.5], [0.2, 0.2, 1.0], [0.1, 0.1, 1.5], ], # 0-4 (spline edge)
        [0.9, 0.1, 1], # 1-5
        [0.9, 0.9, 1], # 2-6
        [0.1, 0.9, 1]  # 3-7
    ]

    loft = Loft(bottom_face, top_face, side_edges)
    loft.chop(0, start_size=0.1, c2c_expansion=1)
    loft.chop(1, c2c_expansion=1, count=20)
    loft.chop(2, c2c_expansion=1, count=30)

    mesh = Mesh()
    mesh.add(loft)
    mesh.set_default_patch('walls', 'wall')
    
    return mesh
