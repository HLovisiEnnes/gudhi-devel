import matplotlib.pyplot as plt
import gudhi

# rips_on_tore3D_1307.pers obtained from write_persistence_diagram method
persistence_file=gudhi.__root_source_dir__ + \
    '/data/persistence_diagram/rips_on_tore3D_1307.pers'
ax = gudhi.plot_persistence_diagram(persistence_file=persistence_file)
# We can modify the title, aspect, etc.
ax.set_title("Persistence diagram of a torus")
ax.set_aspect("equal")  # forces to be square shaped
plt.show()