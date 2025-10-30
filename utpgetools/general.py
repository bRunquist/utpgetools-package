"""
General Utilities Module

This module contains general-purpose utility functions that are commonly used 
across various petroleum engineering calculations and analyses. These functions 
provide basic mathematical operations, data structure manipulations, and other 
general utilities that support the broader utpgetools package functionality.

Functions:
    mat_build: Constructs a numpy matrix from dimensions and a flat list of values
    
Notes:
    This module focuses on generic utility functions rather than domain-specific
    petroleum engineering calculations, making it useful for supporting various
    computational tasks throughout the package.
"""

def mat_build(dimensions, values):
	"""
	Build a numpy matrix from dimensions and a flat list of values.
	
	This function takes a tuple specifying matrix dimensions and a flat list of values,
	then constructs a numpy array by reshaping the values into the specified matrix shape.
	Values are arranged in row-major order (left-to-right, top-to-bottom).
	
	Args:
		dimensions (tuple): A 2-element tuple (rows, cols) specifying the matrix dimensions.
			rows (int): Number of rows in the resulting matrix
			cols (int): Number of columns in the resulting matrix
		values (list or array-like): Flat list/array of values to populate the matrix.
			Must contain exactly (rows × cols) elements. Values should be ordered
			left-to-right, top-to-bottom as they would appear in the final matrix.
	
	Returns:
		numpy.ndarray: A 2D numpy array with shape (rows, cols) containing the
			reshaped input values.
	
	Raises:
		ValueError: If the number of values does not match the specified matrix dimensions
			(i.e., len(values) ≠ rows × cols).
		TypeError: If dimensions is not a 2-element tuple or if values cannot be
			converted to a numpy array.
	
	Examples:
		>>> # Create a 2x3 matrix
		>>> mat_build((2, 3), [1, 2, 3, 4, 5, 6])
		array([[1, 2, 3],
		       [4, 5, 6]])
		
		>>> # Create a 3x2 matrix with the same values
		>>> mat_build((3, 2), [1, 2, 3, 4, 5, 6])
		array([[1, 2],
		       [3, 4],
		       [5, 6]])
		
		>>> # Using with floating point values
		>>> mat_build((2, 2), [1.5, 2.7, 3.1, 4.9])
		array([[1.5, 2.7],
		       [3.1, 4.9]])
	
	Notes:
		- The function uses numpy's reshape method, which creates a view of the
		  original data when possible, making it memory efficient.
		- Input values can be any numeric type that numpy can handle (int, float, complex).
		- The resulting matrix follows numpy's standard indexing convention (0-based).
	"""
	import numpy as np
	rows, cols = dimensions
	if len(values) != rows * cols:
		raise ValueError("Number of values does not match matrix dimensions.")
	return np.array(values).reshape(rows, cols)
