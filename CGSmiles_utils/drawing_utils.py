
import numpy as np
import matplotlib.pyplot as plt
from rdkit.Chem import rdMolTransforms

def _rotate_molecule_around_center(mol, theta):
	"""
	Rotate a molecule in the XY plane around its centroid.
	
	Parameters
	----------
	mol : rdkit.Chem.rdchem.Mol
		The RDKit molecule to rotate. Must have 2D conformer.
	theta : float
		The rotation angle in degrees.

	Returns
	-------
	None
		The molecule is modified in place.
	"""
	conf = mol.GetConformer(0)

	coords = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
	centroid = coords.mean(axis=0)

	# translation to origin
	T1 = np.eye(4)
	T1[:3, 3] = -centroid

	# rotation in xy plane around origin 
	theta = np.deg2rad(theta)
	c, s = np.cos(theta), np.sin(theta)
	R = np.array([
		[c,  s, 0, 0],
		[-s, c, 0, 0],
		[0,  0, 1, 0],
		[0,  0, 0, 1]
	])

	# translate centroid back to original position
	T2 = np.eye(4)
	T2[:3, 3] = centroid

	# Full transform = T2 * R * T1
	M = T2 @ R @ T1

	rdMolTransforms.TransformConformer(conf, M)

def _flip_molecule(mol):
	"""
	Flip a molecule horizontally in the XY plane.
	
	Parameters
	----------
	mol : rdkit.Chem.rdchem.Mol
		The RDKit molecule to flip. Must have 2D conformer.

	Returns
	-------
	None
		The molecule is modified in place.
	"""
	conf = mol.GetConformer(0)

	# Horizontal flip
	R = np.array([	
		[-1., 0., 0., 0.],
		[ 0., 1., 0., 0.],
		[ 0., 0., 1., 0.],
		[ 0., 0., 0., 1.]
	])

	rdMolTransforms.TransformConformer(conf, R)

def _bead_radius(bead_type):
	'''
	Determine radius for the bead circle based on bead size given in the bead type.

	Parameters
	----------
	bead_type : str
	
	Returns
	-------
	float
		The radius of the bead circle.
	'''
	match bead_type[0]:
		case 'U': # use small circles for dummy beads
			bead_radius = 7.5
		case 'T':
			bead_radius = 15
		case 'S':
			bead_radius = 20
		case 'R':
			bead_radius = 23.5
		case _:  # default to 'R' if size is not explicitly given
			bead_radius = 23.5 
	return float(bead_radius)

def _bead_color(bead_type):
	'''
	Determine color for the bead circle based on bead type.

	Parameters
	----------
	bead_type : str

	Returns
	-------
	str
		The color of the bead circle in hex format.
	'''
	if bead_type[0] in ['T', 'S', 'R']: # remove size designation if present
		bead_type = bead_type[1]
	else:
		bead_type = bead_type[0]
		
	match bead_type:
		case 'C':
			color = '#cccccc' # light gray
		case 'N':
			color = '#80b3ff' # blue
		case 'P':
			color = '#ff0000' # red
		case 'X':
			color = '#ccff00' # lime
		case 'Q':
			color = '#ffcc00' # yellow
		case 'D':
			color = '#ff8000' # orange
		case 'U':
			color = '#8e008e' # purple
		case _:
			color = '#808080' # gray
	return color

def _create_bead_circle(bead_position, bead_type, bead_name, scale_factor, dashed=False):
	'''
	Create SVG circle element for a bead.
	
	Parameters
	----------
	bead_position : list or np.array
		The (x, y) position of the bead.
	bead_type : str
		The type of the bead.
	bead_name : str
		The name of the bead.
	dashed : bool
		Draw the outline dashed (used for virtual sites).

	Returns
	-------
	str
		The SVG circle element as a string.
	'''
	bead_name = bead_name if bead_name is not None else bead_type
	dasharray = f'{2.5 * scale_factor},{1.75 * scale_factor}' if dashed else 'none'
	circle_svg = ['<circle\n']
	circle_svg += f'\tstyle="opacity:0.5;fill:{_bead_color(bead_type)};fill-opacity:1;stroke:#6e6e6e;stroke-width:{1.75 * scale_factor};stroke-miterlimit:4;stroke-dasharray:{dasharray}"\n'
	circle_svg += f'\tid="circle_bead_{bead_name}"\n'
	circle_svg += f'\tcx="{bead_position[0]}"\n'
	circle_svg += f'\tcy="{bead_position[1]}"\n'
	circle_svg += f'\tr="{_bead_radius(bead_type)*scale_factor}" />\n'
	return ''.join(circle_svg)

def _create_bond_line(start, end, color='#000000', width=2.0):
	'''
	Create SVG line element for a bond or constraint between two beads.

	Parameters
	----------
	start, end : list or np.array
		The (x, y) positions of the two beads.
	color : str
		Line color.
	width : float
		Line width.

	Returns
	-------
	str
		The SVG line element as a string.
	'''
	line_svg = ['<line\n']
	line_svg += f'\tstyle="fill:none;stroke:{color};stroke-width:{width};stroke-linecap:round"\n'
	line_svg += f'\tx1="{start[0]}"\n'
	line_svg += f'\ty1="{start[1]}"\n'
	line_svg += f'\tx2="{end[0]}"\n'
	line_svg += f'\ty2="{end[1]}" />\n'
	return ''.join(line_svg)

def _create_bead_label(bead_position, bead_type, bead_name=None, label_position=None, scale_factor=1.0, type_color='#666666', name_color='#000000'):
	'''
	Create SVG text element for a bead label.

	Parameters
	----------
	bead_position : list or np.array
		The (x, y) position of the center of the bead.
	bead_type : str
		The bead type/name.
	bead_name : str, optional, default: None
		The bead name to display above the bead type. If None, only the bead type is shown.
	type_color : str, optional, default: '#666666'
		The color of the bead type text in hex format.
	name_color : str, optional, default: '#000000'
		The color of the bead name text in hex format.   
	
	Returns
	-------
	str
		The SVG text element as a string.
	'''
	fontsize = 13 * scale_factor  # base font size scaled by the scale factor
	if bead_type == 'U' and bead_name is not None: # only show bead name in black
		type_color = name_color # use black color for virtual nodes
		bead_type = bead_name

	if label_position is None:
		offset = 1 / np.sqrt(2)
		label_position = lambda pos, r, bead_name: (pos[0] + r * offset, pos[1] - r * offset)
	position = label_position(bead_position, _bead_radius(bead_type) * scale_factor, bead_name)
	
	# define text element
	text_svg = ['<text\n']
	text_svg.append(f'\tid="text_bead_{bead_name if bead_name is not None else bead_type}"\n')
	text_svg.append(f'\tstyle="font-size:{fontsize}px;dominant-baseline:bottom;text-anchor:start;font-family:sans-serif;"\n')
	text_svg.append(f'\tx="{position[0]}"\n')
	text_svg.append(f'\ty="{position[1]}">\n')

	# define tspan element for bead name if given
	if not bead_type.startswith('U') and bead_name is not None:
		text_svg.append(f'\t<tspan x="{position[0]}" y="{position[1]-fontsize}" font-size="{fontsize}px" fill="{name_color}" font-family="sans-serif">{bead_name}</tspan>\n')
	# define tspan element for bead type
	text_svg.append(f'\t<tspan x="{position[0]}" y="{position[1]}" font-size="{fontsize}px" fill="{type_color}" font-family="sans-serif">{bead_type}</tspan>\n')
	text_svg.append('</text>\n')
	return ''.join(text_svg)

def detect_virtual_nodes(resgraph):
	"""
	Detect virtual edges (nodes only connected by edges with 'order' == 0)

	Parameters
	resgraph : networkx.Graph
		The residue graph to analyze.

	Returns
	dict
		A dictionary where keys are the index of the virtual nodes,
		and values are sorted lists of the indices of the nodes they are connected to.
	"""
	virtual_edges = {}
	for node in resgraph.nodes:
		edges = resgraph.edges(node, data=True)
		# If node has no edges, skip
		if not list(edges):
			continue
		# if all edges have 'order' == 0, node is VS
		if all(edge_data.get('order', None) == 0 for _, _, edge_data in edges):
			# save the indices of the edges
			virtual_edges[node] = sorted([v if u == node else u for u, v, _ in edges])
	return virtual_edges

def _check_vertical_lines(svg, highlight=True, threshold=0.5, tolerance=0.02):
	'''
	Check for vertical lines in the SVG and highlight them in red if requested.
	Vertical lines are defined as lines with an angle of less than `threshold` degrees to the y-axis.
	A tolerance can be set to account for small deviations from perfect verticality.

	Parameters
	----------
	svg : str
		The SVG string to check.
	highlight : bool, optional, default: True
		Whether to highlight vertical lines in red.
	threshold : float, optional, default: 0.5
		The angle threshold in degrees to consider a line as vertical.
	tolerance : float, optional, default: 0.02
		The tolerance in degrees to account for small deviations from perfect verticality.
	
	Returns
	-------
	str or None
	'''
	min = 180
	min_index = []
	sign = '+'
	for i, line in enumerate(svg.split('\n')):
		if line.strip().startswith('<path') and 'd=' in line and line.strip().endswith('/>'):
			# get the part after d=" and before the next "
			d_part = line.split("d='")[1].split("'")[0]
			tokens = d_part.replace(",", " ").split()
			if len(tokens) > 6:
				continue
			x1 = float(tokens[1])
			y1 = float(tokens[2])
			x2 = float(tokens[4])
			y2 = float(tokens[5])

			# calculate the angle between the y-axis and the line
			if y1-y2 != 0:
				angle = np.arctan((x1-x2)/(y1-y2)) * 180 / np.pi
				# print(f"angle: {angle:.2f} degrees")
			if abs(angle) <= min+tolerance:
				min = abs(angle)
				if abs(min - abs(angle))<tolerance:
					min_index.append(i)
					if angle < 0:
						sign = '-'
					else:
						sign = '+'
				else:
					min_index = [i]
	if min < threshold:
		print(f"Found vertical lines with angle {sign}{min:.2f} degrees")
	else:
		print(f"No vertical lines found, min angle: {sign}{min:.2f} degrees")
	if highlight:
		lines = svg.split('\n')
		for i in min_index:
			lines[i] = lines[i].replace('stroke:#000000', 'stroke:#FF0000')
		svg = '\n'.join(lines)
	return svg

def _get_bead_coordinate(atoms, idx_map, drawer_coords, opts, bead):
	coords = []
	weights = []
	for atom, data in atoms.nodes(data=True): # loop over atoms in the bead
		if not opts.hydrogens_in_bead_position and data['element'] == 'H':
			continue
		atom = idx_map[atom]
		coords.append(drawer_coords.GetDrawCoords(atom))
		weights.append(data['weight'])
	if np.all(np.array(weights) == 0): 
		raise ValueError(f"All weights are zero for bead {bead}. Cannot compute bead position.")
	bead_position = np.average(coords, axis=0, weights=weights)
	return bead_position

def _strip_background(svg):
	'''
	Strip the background rectangle from the SVG to make it transparent.

	Parameters
	----------
	svg : str
		The SVG string to modify.

	Returns
	-------
	str
		The modified SVG string without the background rectangle.
	'''
	lines = svg.split('\n')
	new_lines = []
	for line in lines:
		if line.strip().startswith('<rect') and ('fill:#ffffff' in line or 'fill:#FFFFFF' in line):
			continue
		new_lines.append(line)
	return '\n'.join(new_lines)