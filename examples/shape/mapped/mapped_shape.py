import classy_blocks as cb

mesh = cb.Mesh()


shape = cb.MappedShape.from_file("port.shape")

mesh.add(shape)

cb.FixedCountGrader(mesh).grade()
mesh.write("../../case/system/blockMeshDict")
