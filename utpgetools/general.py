# ...existing code...
def mat_build(dimensions, values):
	"""
	Build a numpy matrix from dimensions and a flat list of values.
	Args:
		dimensions (tuple): (rows, cols)
		values (list): flat list of values, left-to-right, top-to-bottom
	Returns:
		numpy.ndarray: matrix as a numpy array
	"""
	import numpy as np
	rows, cols = dimensions
	if len(values) != rows * cols:
		raise ValueError("Number of values does not match matrix dimensions.")
	return np.array(values).reshape(rows, cols)
# ...existing code...
