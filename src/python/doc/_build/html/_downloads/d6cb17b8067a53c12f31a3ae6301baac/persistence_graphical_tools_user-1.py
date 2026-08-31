import matplotlib.pyplot as plt
import gudhi

off_file = gudhi.__root_source_dir__ + '/data/points/tore3D_300.off'
point_cloud = gudhi.read_points_from_off_file(off_file=off_file)

rips_complex = gudhi.RipsComplex(points=point_cloud, max_edge_length=0.7)
simplex_tree = rips_complex.create_simplex_tree(max_dimension=3)
diag = simplex_tree.persistence(min_persistence=0.4)

gudhi.plot_persistence_barcode(diag)
plt.show()