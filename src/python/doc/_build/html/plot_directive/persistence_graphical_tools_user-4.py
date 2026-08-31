import matplotlib.pyplot as plt
import gudhi
# rips_on_tore3D_1307.pers obtained from write_persistence_diagram method
persistence_file=gudhi.__root_source_dir__ + \
    '/data/persistence_diagram/rips_on_tore3D_1307.pers'
birth_death = gudhi.read_persistence_intervals_in_dimension(
    persistence_file=persistence_file, only_this_dim=1)
# Use subplots to display diagram and density side by side
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
gudhi.plot_persistence_diagram(persistence=birth_death, axes=axes[0])
gudhi.plot_persistence_density(persistence=birth_death, axes=axes[1])
plt.show()