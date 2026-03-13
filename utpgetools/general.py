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
def calculate_gpa_percentile(gpa, college='Cockrell'):
    """
    Calculate your percentile based on your GPA for a given college.
    
    Args:
        gpa (float): The GPA to evaluate.
        college (str): The college name. Default is 'Cockrell'.
        
    Returns:
        float: The percentile corresponding to the GPA.
    """
    import numpy as np
    from scipy.optimize import curve_fit

    # GPA cutoffs for each college: [80th percentile, 90th percentile, 96th percentile]
    cutoff_dict = {
        'Architecture': [3.30, 3.30, 3.30],
        'McCombs': [3.50, 3.65, 3.80],
        'Moody': [3.465, 3.665, 3.865],
        'Education': [3.50, 3.65, 3.80],
        'Cockrell': [3.50, 3.70, 3.85],
        'Fine Arts': [3.30, 3.60, 3.85],
        'Jackson': [3.30, 3.667, 3.867],
        'COLA': [3.30, 3.667, 3.867],
        'CNS': [3.30, 3.667, 3.867],
        'Nursing': [3.30, 3.30, 3.30],
        'Pharmacy': [3.30, 3.30, 3.30],
        'Social Work': [3.30, 3.30, 3.30]
    }
    
    colleges = list(cutoff_dict.keys())
    blacklist = ['Architecture', 'Nursing', 'Pharmacy', 'Social Work']
    
    if college in blacklist:
        raise ValueError(f"GPA percentile calculation is not meaningful for the {college} college, all cutoffs are the same.")
        return None
    elif college not in colleges:
        raise ValueError(f"College '{college}' not recognized. Available colleges: {', '.join(colleges)}")
        return None
    
    # Get cutoffs for the specified college
    gpas = cutoff_dict[college]
    percentiles = [80, 90, 96]
    
    def func(x,a,b):
        return a*x**2 + b*x
    popt, pcov = curve_fit(func, gpas, percentiles)

    residuals = percentiles - func(np.array(gpas), *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((percentiles - np.mean(percentiles))**2)
    r_squared = 1 - (ss_res / ss_tot)
    print(f"R^2 of the fit: {r_squared:.4f}")
    percentile = func(gpa, *popt)
    if gpa >= gpas[0] and gpa < gpas[1]:
        print(f"\n\nCongratulations! Your GPA of {gpa} is enough to graduate with Honors!")
    elif gpa >= gpas[1] and gpa < gpas[2]:
        print(f"\n\nCongratulations! Your GPA of {gpa} is enough to graduate with High Honors!")
    elif gpa >= gpas[2]:
        print(f"\n\nCongratulations! Your GPA of {gpa} is enough to graduate with Highest Honors!")

    print(f"For the {college} school, your GPA of {gpa} corresponds to approximately the {percentile:.2f}th percentile, or top {100 - percentile:.2f}%.")
    print("Note: This is a trend line fit, and it's not going to be perfect. Use it as a general guideline only.")
    return percentile