
import csv
from typing import Dict, List, Any

def read_dev(dev_file_path: str) -> Dict[str, List[Any]]:
	"""
	Reads a .dev file and returns a dictionary where each key is a column name and each value is a list of column values (floats if possible, otherwise strings).
	Args:
		dev_file_path (str): Path to the .dev file.
	Returns:
		Dict[str, List[Any]]: Dictionary of columns with lists of values.
	"""
	columns = {}
	with open(dev_file_path, 'r', newline='') as f:
		reader = csv.DictReader(f, delimiter='\t')
		for header in reader.fieldnames:
			columns[header] = []
		for row in reader:
			for key in reader.fieldnames:
				value = row[key]
				# Try to convert to float, else keep as string
				try:
					columns[key].append(float(value))
				except (ValueError, TypeError):
					columns[key].append(value)
	return columns

def fault_stress_visualization(sv,
							   shmax,
							   shmin,
							   pore_pressure,
							   fault_strike,
							   fault_dip,
							   friction_coefficient=None,
							   shmin_strike=None,
							   shmax_strike=None):
	# package imports
	import numpy as np
	import matplotlib.pyplot as plt

	# Function to normalize angles to 0-360 degrees
	def normalize_angle(angle):
		"""Normalize angle to 0-360 degrees range"""
		return angle % 360
	
	# Define non-present strike of principal stresses
	if shmin_strike is None:
		shmin_strike = normalize_angle(shmax_strike + 90)
	if shmax_strike is None:
		shmax_strike = normalize_angle(shmin_strike - 90)
	
	# Normalize all input angles
	fault_strike = normalize_angle(fault_strike)
	shmax_strike = normalize_angle(shmax_strike)
	shmin_strike = normalize_angle(shmin_strike)

	# Early colinearity check - before any plotting
	dip_direction = normalize_angle(fault_strike + 90)
	
	# Calculate angular differences with horizontal stresses
	diff_to_shmax = min(abs(dip_direction - shmax_strike), 360 - abs(dip_direction - shmax_strike))
	diff_to_shmin = min(abs(dip_direction - shmin_strike), 360 - abs(dip_direction - shmin_strike))
	
	# Check if fault dip direction is colinear with one of the principal horizontal stresses
	# Allow a small tolerance (e.g., 1 degrees) for practical purposes
	tolerance = 1.0
	is_colinear_shmax = diff_to_shmax <= tolerance
	is_colinear_shmin = diff_to_shmin <= tolerance
	
	if not (is_colinear_shmax or is_colinear_shmin) and fault_dip != 90:
		error_message = (f"Error: The fault dip direction ({dip_direction:.1f}°) is not colinear with either "
						f"SHmax ({shmax_strike:.1f}°) or Shmin ({shmin_strike:.1f}°). "
						f"The fault cannot be properly analyzed using the 3D Mohr's circle. "
						)
		return error_message

	# Calculate effective stresses
	sigma_v = sv - pore_pressure
	sigma_hmax = shmax - pore_pressure
	sigma_hmin = shmin - pore_pressure

	# 3d Mohr's circle calculations and logic
	
	# Circle 1 Calculations
	C1 = (sigma_hmax + sigma_hmin) / 2
	R1 = (sigma_hmax - sigma_hmin) / 2
	theta = np.linspace(0, np.pi, 100)  # Fixed: start from 0, not 9
	X1 = C1 + R1 * np.cos(theta)
	Y1 = R1 * np.sin(theta)
	# the x and y curves for circle 1 (shmin to shmax) are now defined
	# Circle 2 Calculations
	C2 = (sigma_v + sigma_hmax) / 2
	R2 = abs(sigma_v - sigma_hmax) / 2  # Use absolute value to ensure positive radius
	X2 = C2 + R2 * np.cos(theta)
	Y2 = R2 * np.sin(theta)
	# Circle 3 Calculations
	C3 = (sigma_v + sigma_hmin) / 2
	R3 = (sigma_v - sigma_hmin) / 2
	X3 = C3 + R3 * np.cos(theta)
	Y3 = R3 * np.sin(theta)
	# Shear Stress line (only if friction coefficient is provided)
	stresses = [sigma_v, sigma_hmax, sigma_hmin]
	sigman = np.linspace(0, np.max(stresses), 100)
	if friction_coefficient is not None:
		shear_stress = sigman * friction_coefficient
	
	# Create the figure with three subplots (custom layout to make subplot 3 larger)
	fig = plt.figure(figsize=(18, 8))
	
	# Subplot 1: 3D Cubic Volume with Principal Stress Arrows (top left)
	ax1 = fig.add_subplot(221, projection='3d')
	
	# Create cube vertices
	cube_size = 1
	vertices = np.array([
		[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # bottom face
		[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]   # top face
	]) * cube_size
	
	# Define cube edges
	edges = [
		[0, 1], [1, 2], [2, 3], [3, 0],  # bottom face
		[4, 5], [5, 6], [6, 7], [7, 4],  # top face
		[0, 4], [1, 5], [2, 6], [3, 7]   # vertical edges
	]
	
	# Draw cube edges
	for edge in edges:
		points = vertices[edge]
		ax1.plot3D(points[:, 0], points[:, 1], points[:, 2], 'k-', alpha=0.6)
	
	# Add principal stress arrows
	# Sv (vertical) - from center bottom to center top
	ax1.quiver(0.5, 0.5, 0, 0, 0, sv/max(stresses), color='blue', arrow_length_ratio=0.1, linewidth=3)
	ax1.text(0.5, 0.5, 1.2, f'Sv = {sv:.1f}', fontsize=10, ha='center')
	
	# Convert strikes to radians for arrow directions (geological convention: 0° = North, clockwise)
	# For geological strikes: x = sin(strike), y = cos(strike)
	shmax_rad = np.deg2rad(shmax_strike)
	shmin_rad = np.deg2rad(shmin_strike)
	
	# Shmax arrow - from center of appropriate face
	shmax_x_dir = np.sin(shmax_rad)  # East component
	shmax_y_dir = np.cos(shmax_rad)  # North component
	ax1.quiver(0.5, 0.5, 0.5, shmax_x_dir * shmax/max(stresses), shmax_y_dir * shmax/max(stresses), 0, 
			   color='red', arrow_length_ratio=0.1, linewidth=3)
	ax1.text(0.5 + shmax_x_dir * 0.6, 0.5 + shmax_y_dir * 0.6, 0.5, f'SHmax = {shmax:.1f}', 
			 fontsize=10, ha='center')
	
	# Shmin arrow - perpendicular to Shmax
	shmin_x_dir = np.sin(shmin_rad)  # East component
	shmin_y_dir = np.cos(shmin_rad)  # North component
	ax1.quiver(0.5, 0.5, 0.5, shmin_x_dir * shmin/max(stresses), shmin_y_dir * shmin/max(stresses), 0, 
			   color='green', arrow_length_ratio=0.1, linewidth=3)
	ax1.text(0.5 + shmin_x_dir * 0.6, 0.5 + shmin_y_dir * 0.6, 0.5, f'Shmin = {shmin:.1f}', 
			 fontsize=10, ha='center')
	
	# Add fault plane to 3D volume with intersection lines
	fault_strike_rad = np.deg2rad(fault_strike)
	fault_dip_rad = np.deg2rad(fault_dip)
	
	# Calculate fault plane normal vector (geological convention)
	# Strike direction vector (along fault)
	strike_x = np.sin(fault_strike_rad)  # East
	strike_y = np.cos(fault_strike_rad)  # North
	strike_z = 0  # Horizontal
	
	# Dip direction (perpendicular to strike, rotated 90° clockwise)
	dip_dir_rad = fault_strike_rad + np.pi/2
	dip_x = np.sin(dip_dir_rad) * np.cos(fault_dip_rad)  # East component
	dip_y = np.cos(dip_dir_rad) * np.cos(fault_dip_rad)  # North component
	dip_z = -np.sin(fault_dip_rad)  # Depth component (negative because dip goes down)
	
	# Calculate fault plane equation: ax + by + cz = d
	# Normal vector to fault plane
	normal_x = strike_y * dip_z - strike_z * dip_y
	normal_y = strike_z * dip_x - strike_x * dip_z
	normal_z = strike_x * dip_y - strike_y * dip_x
	
	# Plane passes through cube center (0.5, 0.5, 0.5)
	d = normal_x * 0.5 + normal_y * 0.5 + normal_z * 0.5
	
	# Function to find intersection of plane with cube edges
	def plane_line_intersect(p1, p2, nx, ny, nz, d):
		"""Find intersection of plane (nx*x + ny*y + nz*z = d) with line segment p1-p2"""
		x1, y1, z1 = p1
		x2, y2, z2 = p2
		
		# Direction vector of line
		dx, dy, dz = x2 - x1, y2 - y1, z2 - z1
		
		# Check if line is parallel to plane
		denom = nx * dx + ny * dy + nz * dz
		if abs(denom) < 1e-10:
			return None
		
		# Calculate parameter t for intersection
		t = (d - nx * x1 - ny * y1 - nz * z1) / denom
		
		# Check if intersection is within line segment
		if 0 <= t <= 1:
			return (x1 + t * dx, y1 + t * dy, z1 + t * dz)
		return None
	
	# Find all intersections with cube edges
	cube_edges = [
		# Bottom face edges
		([0, 0, 0], [1, 0, 0]), ([1, 0, 0], [1, 1, 0]), ([1, 1, 0], [0, 1, 0]), ([0, 1, 0], [0, 0, 0]),
		# Top face edges
		([0, 0, 1], [1, 0, 1]), ([1, 0, 1], [1, 1, 1]), ([1, 1, 1], [0, 1, 1]), ([0, 1, 1], [0, 0, 1]),
		# Vertical edges
		([0, 0, 0], [0, 0, 1]), ([1, 0, 0], [1, 0, 1]), ([1, 1, 0], [1, 1, 1]), ([0, 1, 0], [0, 1, 1])
	]
	
	intersection_points = []
	for p1, p2 in cube_edges:
		intersection = plane_line_intersect(p1, p2, normal_x, normal_y, normal_z, d)
		if intersection is not None:
			intersection_points.append(intersection)
	
	# Remove duplicate points (within tolerance)
	unique_intersections = []
	for point in intersection_points:
		is_duplicate = False
		for existing in unique_intersections:
			if (abs(point[0] - existing[0]) < 1e-6 and 
				abs(point[1] - existing[1]) < 1e-6 and 
				abs(point[2] - existing[2]) < 1e-6):
				is_duplicate = True
				break
		if not is_duplicate:
			unique_intersections.append(point)
	
	# Draw intersection lines on cube faces if we have enough points
	if len(unique_intersections) >= 3:
		# Sort points to create a proper polygon
		# For simplicity, we'll connect consecutive points
		points = np.array(unique_intersections)
		
		# Draw the fault plane as lines connecting intersection points
		from mpl_toolkits.mplot3d.art3d import Poly3DCollection
		
		# Create a polygon from the intersection points
		if len(points) >= 3:
			# Try to order points to form a proper polygon
			center_point = np.mean(points, axis=0)
			
			# Sort points by angle around center (in strike direction plane)
			angles = []
			for point in points:
				vec = point - center_point
				angle = np.arctan2(np.dot(vec, [dip_x, dip_y, dip_z]), 
								  np.dot(vec, [strike_x, strike_y, strike_z]))
				angles.append(angle)
			
			# Sort points by angle
			sorted_indices = np.argsort(angles)
			sorted_points = points[sorted_indices]
			
			# Draw the fault plane
			fault_plane = Poly3DCollection([sorted_points], alpha=0.3, facecolor='purple', edgecolor='purple', linewidth=2)
			ax1.add_collection3d(fault_plane)
			
			# Draw bold lines around the fault plane edges
			for i in range(len(sorted_points)):
				p1 = sorted_points[i]
				p2 = sorted_points[(i + 1) % len(sorted_points)]
				ax1.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
						'purple', linewidth=3, alpha=0.8)
	
	ax1.set_xlabel('East (X)')
	ax1.set_ylabel('North (Y)')
	ax1.set_zlabel('Z (Depth)')
	ax1.set_title('3D Visualization')
	ax1.set_xlim([0, 1])
	ax1.set_ylim([0, 1])
	ax1.set_zlim([0, 1])

	# Subplot 2: Top-down view with rotated square and fault (bottom left)
	ax2 = fig.add_subplot(223)
	
	# Create square aligned with principal stress orientation
	# Using geological convention: X = East, Y = North, 0° = North, clockwise
	square_size = 1.0
	
	# Simple square aligned with stress directions (no rotation needed)
	# Square edges should align with SHmax and Shmin directions
	shmax_dir_x = np.sin(np.deg2rad(shmax_strike))  # East component
	shmax_dir_y = np.cos(np.deg2rad(shmax_strike))  # North component
	shmin_dir_x = np.sin(np.deg2rad(shmin_strike))  # East component  
	shmin_dir_y = np.cos(np.deg2rad(shmin_strike))  # North component
	
	# Create square corners aligned with stress directions
	corners = np.array([
		[-square_size/2 * shmax_dir_x - square_size/2 * shmin_dir_x, -square_size/2 * shmax_dir_y - square_size/2 * shmin_dir_y],
		[square_size/2 * shmax_dir_x - square_size/2 * shmin_dir_x, square_size/2 * shmax_dir_y - square_size/2 * shmin_dir_y],
		[square_size/2 * shmax_dir_x + square_size/2 * shmin_dir_x, square_size/2 * shmax_dir_y + square_size/2 * shmin_dir_y],
		[-square_size/2 * shmax_dir_x + square_size/2 * shmin_dir_x, -square_size/2 * shmax_dir_y + square_size/2 * shmin_dir_y],
		[-square_size/2 * shmax_dir_x - square_size/2 * shmin_dir_x, -square_size/2 * shmax_dir_y - square_size/2 * shmin_dir_y]  # Close the square
	])
	
	# Plot square
	ax2.plot(corners[:, 0], corners[:, 1], 'k-', linewidth=2)
	
	# Draw SHmax arrow
	shmax_arrow_length = 0.3
	shmax_arrow_x = shmax_dir_x * shmax_arrow_length
	shmax_arrow_y = shmax_dir_y * shmax_arrow_length
	ax2.arrow(0, 0, shmax_arrow_x, shmax_arrow_y, head_width=0.02, head_length=0.02, 
			  fc='red', ec='red', linewidth=2, label=f'SHmax = {shmax:.1f} (Strike: {shmax_strike:.1f}°)')
	
	# Draw Shmin arrow
	shmin_arrow_length = 0.2
	shmin_arrow_x = shmin_dir_x * shmin_arrow_length
	shmin_arrow_y = shmin_dir_y * shmin_arrow_length
	ax2.arrow(0, 0, shmin_arrow_x, shmin_arrow_y, head_width=0.02, head_length=0.02, 
			  fc='green', ec='green', linewidth=2, label=f'Shmin = {shmin:.1f} (Strike: {shmin_strike:.1f}°)')
	
	# Draw fault line that intersects with square edges only
	fault_angle = np.deg2rad(fault_strike)
	
	# Calculate fault direction vectors
	fault_dir_x = np.sin(fault_angle)  # East component
	fault_dir_y = np.cos(fault_angle)  # North component
	
	# Function to find line-segment intersection
	def line_intersect(p1, p2, p3, p4):
		"""Find intersection of line segments p1-p2 and p3-p4"""
		x1, y1 = p1
		x2, y2 = p2
		x3, y3 = p3
		x4, y4 = p4
		
		denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
		if abs(denom) < 1e-10:
			return None
		
		t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / denom
		u = -((x1-x2)*(y1-y3) - (y1-y2)*(x1-x3)) / denom
		
		if 0 <= u <= 1:  # Check if intersection is on the square edge
			return (x1 + t*(x2-x1), y1 + t*(y2-y1))
		return None
	
	# Find intersections with square edges
	# Extend fault line far beyond square for intersection calculation
	fault_extend = 2.0
	fault_start = (-fault_dir_x * fault_extend, -fault_dir_y * fault_extend)
	fault_end = (fault_dir_x * fault_extend, fault_dir_y * fault_extend)
	
	intersections = []
	# Check intersection with each square edge
	for i in range(4):  # 4 edges of the square
		edge_start = corners[i]
		edge_end = corners[i+1]
		intersection = line_intersect(fault_start, fault_end, edge_start, edge_end)
		if intersection is not None:
			intersections.append(intersection)
	
	# Draw fault line between the two intersection points (if found)
	if len(intersections) >= 2:
		ax2.plot([intersections[0][0], intersections[1][0]], 
				 [intersections[0][1], intersections[1][1]], 'purple', linewidth=3, 
				 label=f'Fault (Strike: {fault_strike:.1f}°)')
	else:
		# Fallback: draw a short fault line if no intersections found
		fault_length = 0.2
		fault_x1 = -fault_dir_x * fault_length
		fault_y1 = -fault_dir_y * fault_length
		fault_x2 = fault_dir_x * fault_length
		fault_y2 = fault_dir_y * fault_length
		ax2.plot([fault_x1, fault_x2], [fault_y1, fault_y2], 'purple', linewidth=3, 
				 label=f'Fault (Strike: {fault_strike:.1f}°)')
	
	ax2.set_xlim([-0.7, 0.7])
	ax2.set_ylim([-0.7, 0.7])
	ax2.set_aspect('equal')
	ax2.set_xlabel('East')
	ax2.set_ylabel('North')
	ax2.set_title('Top-Down View')
	ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
	ax2.grid(True, alpha=0.3)
	
	# Subplot 3: Mohr's Circles with fault analysis (right side - larger)
	ax3 = fig.add_subplot(122)
	
	# Plot the three Mohr's circles
	ax3.plot(X1, Y1, 'r-', linewidth=2, label='Circle 1 (Shmin-SHmax)')
	ax3.plot(X2, Y2, 'g-', linewidth=2, label='Circle 2 (SHmax-Sv)')
	ax3.plot(X3, Y3, 'b-', linewidth=2, label='Circle 3 (Shmin-Sv)')

	# Plot shear failure line (only if friction coefficient is provided)
	if friction_coefficient is not None:
		ax3.plot(sigman, shear_stress, 'k--', linewidth=2, label='Shear Failure Line')
	
	# Determine which circle to plot the fault angle on
	# (Note: colinearity check was already performed at function start)
	
	if fault_dip == 90:
		# Vertical fault - use circle 1
		circle_choice = 1
		circle_center = C1
		circle_radius = R1
		circle_color = 'red'
		
		# Calculate angular difference between fault strike and closest horizontal stress
		diff_to_shmax_strike = min(abs(fault_strike - shmax_strike), 360 - abs(fault_strike - shmax_strike))
		diff_to_shmin_strike = min(abs(fault_strike - shmin_strike), 360 - abs(fault_strike - shmin_strike))
		
		# Use the smaller angular difference (closest horizontal stress)
		angular_diff = min(diff_to_shmax_strike, diff_to_shmin_strike)
		angle_on_circle = np.deg2rad(angular_diff * 2)  # For Mohr's circle, angle is doubled
	elif diff_to_shmin < diff_to_shmax:
		# Dip direction closer to Shmin - use circle 3
		circle_choice = 3
		circle_center = C3
		circle_radius = R3
		circle_color = 'blue'
		angle_on_circle = np.deg2rad(fault_dip * 2)  # For Mohr's circle, angle is doubled
	else:
		# Dip direction closer to SHmax - use circle 2
		circle_choice = 2
		circle_center = C2
		circle_radius = R2
		circle_color = 'green'
		angle_on_circle = np.deg2rad((90 - fault_dip) * 2)  # For Mohr's circle, angle is doubled, using 90-dip
	
	# Plot fault point on appropriate circle
	fault_normal_stress = circle_center + circle_radius * np.cos(angle_on_circle)
	fault_shear_stress = circle_radius * np.sin(angle_on_circle)
	
	ax3.plot(fault_normal_stress, fault_shear_stress, 'o', color=circle_color, 
			 markersize=10, markeredgecolor='black', markeredgewidth=2,
			 label=f'Fault on Circle {circle_choice}')
	
	# Add dotted line from circle center to fault point
	ax3.plot([circle_center, fault_normal_stress], [0, fault_shear_stress], 
			 '--', color='gray', linewidth=2, alpha=0.7)
	
	# Add text annotation for fault point
	ax3.annotate(f'Fault\n(Dip: {fault_dip:.1f}°)', 
				xy=(fault_normal_stress, fault_shear_stress),
				xytext=(fault_normal_stress + max(stresses)*0.1, fault_shear_stress + max(stresses)*0.05),
				arrowprops=dict(arrowstyle='->', color='black'),
				fontsize=10, ha='center')
	
	ax3.set_xlabel('Normal Stress (σₙ)')
	ax3.set_ylabel('Shear Stress (τ)')
	ax3.set_title('3D Mohr\'s Circle')
	ax3.legend()
	ax3.grid(True, alpha=0.3)
	ax3.set_xlim([0, max(stresses) * 1.1])
	ax3.set_ylim([0, max(stresses) * 0.6])
	ax3.set_aspect('equal')  # Make circles appear as circles
	
	plt.subplots_adjust(wspace=0.4)  # Increase horizontal spacing between subplots
	plt.show()

	# Determine if fauly will slip
	stress_ratio = fault_shear_stress / fault_normal_stress if fault_normal_stress != 0 else np.inf
	if friction_coefficient is not None:
		if stress_ratio >= friction_coefficient:
			slip_status = "The fault is likely to SLIP."
		else:
			slip_status = "The fault is STABLE and unlikely to slip."
	
	# Print analysis results
	print(f"\nFault Analysis Results:")
	print(f"Fault Strike: {fault_strike:.1f}°")
	print(f"Fault Dip: {fault_dip:.1f}°")
	print(f"Dip Direction: {dip_direction:.1f}°")
	print(f"SHmax Strike: {shmax_strike:.1f}°")
	print(f"Shmin Strike: {shmin_strike:.1f}°")
	print(f"Fault plotted on Circle {circle_choice}")
	print(f"Normal Stress on Fault: {fault_normal_stress:.2f}")
	print(f"Shear Stress on Fault: {fault_shear_stress:.2f}")
	if friction_coefficient is not None:
		print(f"Friction Coefficient: {friction_coefficient:.2f}")
		print(f"Shear to Normal Stress Ratio: {stress_ratio:.2f}")
		print(slip_status)
	else:
		print("Friction Coefficient not provided; slip analysis not performed.")
	
	return fig