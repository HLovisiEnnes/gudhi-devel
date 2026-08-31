import matplotlib.pyplot as plt
import gudhi
import numpy as np
d = np.array([[0., 1.], [1., 2.], [1., np.inf]])
gudhi.plot_persistence_diagram(d)
plt.show()