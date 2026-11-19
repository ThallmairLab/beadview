
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

def _create_polyline(points, color='#000000', width=2.0):
	'''
	Create SVG polyline element, used for angles and dihedrals.

	Parameters
	----------
	points : list of list or np.array
		The (x, y) positions the line passes through.
	color : str
		Line color.
	width : float
		Line width.

	Returns
	-------
	str
		The SVG polyline element as a string.
	'''
	points_str = ' '.join(f'{p[0]},{p[1]}' for p in points)
	polyline_svg = ['<polyline\n']
	polyline_svg += f'\tstyle="fill:none;stroke:{color};stroke-width:{width};stroke-linecap:round;stroke-linejoin:round"\n'
	polyline_svg += f'\tpoints="{points_str}" />\n'
	return ''.join(polyline_svg)

def _offset_path(points, distance, side):
	'''
	Points of a line that follows the path through the given points, shifted by distance to one side.
	The line ends next to the centers of the first and last bead.

	Parameters
	----------
	points : list of list or np.array
		The (x, y) positions of the beads along the path (3 for angles, 4 for dihedrals).
	distance : float
		Distance between the line and the path.
	side : int
		+1 to shift to the left of the path direction (first to last point), -1 to the right.

	Returns
	-------
	list of np.array
		The points of the shifted line.
	'''
	points = [np.asarray(p, dtype=float) for p in points]
	directions = [b - a for a, b in zip(points[:-1], points[1:])]
	normals = [side * np.array([-d[1], d[0]]) / np.linalg.norm(d) for d in directions]
	path = [points[0] + distance * normals[0]]
	for point, d1, d2, n1, n2 in zip(points[1:-1], directions[:-1], directions[1:], normals[:-1], normals[1:]):
		cos = np.dot(n1, n2)
		outer = side * (d1[0] * d2[1] - d1[1] * d2[0]) < 0 # the line runs around the outside of this corner
		if outer and 1 + cos < 2 / 1.5**2: # miter longer than 1.5 * distance: cut the corner off
			path += [point + distance * n1, point + distance * n2]
		elif 1 + cos > 1e-6:
			# corner where the two shifted segments meet; limited for very sharp turns
			path.append(point + distance * min(1 / (1 + cos), 3 / np.linalg.norm(n1 + n2)) * (n1 + n2))
		else:
			path.append(point + distance * n1)
	path.append(points[-1] + distance * normals[-1])
	return path

def _offset_side(points, offset, away_from):
	'''
	Side (+1 left, -1 right of the path direction) to which a path has to be shifted to move away
	from the point away_from, or towards it for a negative offset. See _offset_path.
	'''
	away_from = np.asarray(away_from, dtype=float)
	def mean_distance(side):
		return np.mean([np.linalg.norm(p - away_from) for p in _offset_path(points, abs(offset), side)])
	further = 1 if mean_distance(1) >= mean_distance(-1) else -1
	return further if offset >= 0 else -further

def _bead_label_lines(bead_type, bead_name=None):
	'''
	Lines of a bead label from top to bottom as (text, is_type) tuples.
	Virtual nodes ('U') with a name only show the name, other beads show the name above the type.
	'''
	if bead_type == 'U' and bead_name is not None:
		return [(bead_name, True)]
	if not bead_type.startswith('U') and bead_name is not None:
		return [(bead_name, False), (bead_type, True)]
	return [(bead_type, True)]

def _bead_label_extent(bead_type, bead_name=None, scale_factor=1.0):
	'''
	Approximate extent of a bead label relative to its anchor (left end of the baseline of the last line).

	Returns
	-------
	tuple of float
		(width, above, below): the label spans x to x + width and y - above to y + below.
	'''
	fontsize = 13 * scale_factor
	lines = _bead_label_lines(bead_type, bead_name)
	width = 0.62 * fontsize * max(len(text) for text, _ in lines) # average glyph width of sans-serif fonts
	above = (len(lines) - 1) * fontsize + 0.75 * fontsize # cap height of the top line
	below = 0.2 * fontsize # descenders
	return width, above, below

def _label_above_bead(bead_type, bead_name=None, scale_factor=1.0, gap=1.0):
	'''
	label_position function (see _create_bead_label) that centers the label horizontally on the bead
	with its baseline just above the top edge of the bead.
	'''
	width, _, _ = _bead_label_extent(bead_type, bead_name, scale_factor)
	return lambda pos, r, bead_name: (pos[0] - width / 2, pos[1] - r - gap * scale_factor)

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
	label_position : callable, optional, default: None
		Function (bead_position, bead_radius, bead_name) -> (x, y) of the label anchor
		(left end of the baseline of the last line). Defaults to the upper right of the bead.
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
	if label_position is None:
		offset = 1 / np.sqrt(2)
		label_position = lambda pos, r, bead_name: (pos[0] + r * offset, pos[1] - r * offset)
	position = label_position(bead_position, _bead_radius(bead_type) * scale_factor, bead_name)
	lines = _bead_label_lines(bead_type, bead_name)
	
	# define text element
	text_svg = ['<text\n']
	text_svg.append(f'\tid="text_bead_{bead_name if bead_name is not None else bead_type}"\n')
	text_svg.append(f'\tstyle="font-size:{fontsize}px;dominant-baseline:bottom;text-anchor:start;font-family:sans-serif;"\n')
	text_svg.append(f'\tx="{position[0]}"\n')
	text_svg.append(f'\ty="{position[1]}">\n')

	# one tspan per line, the bead type in type_color (virtual nodes show only their name, in name_color)
	for i, (text, is_type) in enumerate(lines):
		y = position[1] - (len(lines) - 1 - i) * fontsize
		color = type_color if is_type and not (bead_type == 'U' and bead_name is not None) else name_color
		text_svg.append(f'\t<tspan x="{position[0]}" y="{y}" font-size="{fontsize}px" fill="{color}" font-family="sans-serif">{text}</tspan>\n')
	text_svg.append('</text>\n')
	return ''.join(text_svg)

def _point_segment_distances(points, segments):
	'''
	Distances between each point and each line segment.

	Parameters
	----------
	points : np.array, shape (N, 2)
	segments : np.array, shape (S, 2, 2)

	Returns
	-------
	np.array, shape (N, S)
	'''
	a, b = segments[:, 0], segments[:, 1]
	ab = b - a
	length2 = np.maximum(np.sum(ab**2, axis=1), 1e-12)
	ap = points[:, None, :] - a[None, :, :]
	t = np.clip(np.sum(ap * ab[None], axis=2) / length2[None], 0, 1)
	closest = a[None] + t[..., None] * ab[None]
	return np.linalg.norm(points[:, None, :] - closest, axis=2)

def _optimize_label_positions(bead_positions, bead_radii, label_sizes, circles, segments, canvas_size, gap,
							  fixed_centers=None, n_directions=24, n_rings=3, max_sweeps=20):
	'''
	Place one rectangular label next to each bead such that the labels overlap as little as possible
	with each other, with the beads and with the obstacles of the underlying structure.

	Each label can be placed at n_directions directions around its bead and n_rings distances from it.
	The overlap of a candidate with the beads, circles, segments and the area outside the canvas is
	estimated on a grid of points inside the label, the overlap with other labels from the exact
	intersection area. Labels are first placed independently and then repeatedly moved to their best
	position given the other labels until no label moves anymore.

	Parameters
	----------
	bead_positions : np.array, shape (B, 2)
		Centers of the beads.
	bead_radii : np.array, shape (B,)
		Radii of the beads.
	label_sizes : np.array, shape (B, 2)
		Width and height of the label of each bead.
	circles : np.array, shape (C, 3)
		Circular obstacles (x, y, radius), e.g. atom labels.
	segments : np.array, shape (S, 5)
		Line obstacles (x1, y1, x2, y2, half width), e.g. bonds.
	canvas_size : tuple
		Width and height of the drawing; area outside is avoided.
	gap : float
		Clearance kept between labels and their own bead and to all obstacles.
	fixed_centers : dict, optional, default: None
		Centers of labels that are not moved, keyed by bead index. They are still avoided by the other labels.

	Returns
	-------
	np.array, shape (B, 2)
		Centers of the labels.
	'''
	bead_positions = np.asarray(bead_positions, dtype=float).reshape(-1, 2)
	bead_radii = np.asarray(bead_radii, dtype=float)
	label_sizes = np.asarray(label_sizes, dtype=float).reshape(-1, 2)
	circles = np.asarray(circles, dtype=float).reshape(-1, 3)
	segments = np.asarray(segments, dtype=float).reshape(-1, 5)
	n_labels = len(bead_positions)
	if n_labels == 0:
		return bead_positions.copy()

	# candidate directions, starting at the upper right (the default position), y points down in SVG
	angles = -np.pi / 4 + 2 * np.pi * np.arange(n_directions) / n_directions
	directions = np.stack([np.cos(angles), np.sin(angles)], axis=1)
	# shift the label by half its size along the direction so that its edge (or corner) faces the bead
	corner = np.clip(directions * np.sqrt(2), -1, 1)
	ring_steps = np.arange(n_rings)

	# weights of the different kinds of overlap and of small preferences that break ties
	w_label, w_canvas, w_bead, w_circle, w_segment = 4.0, 4.0, 2.0, 1.5, 1.0
	w_ring, w_direction = 0.05, 0.02

	grid = np.stack(np.meshgrid(np.linspace(-0.5, 0.5, 8), np.linspace(-0.5, 0.5, 4)), axis=-1).reshape(-1, 2)
	candidates, static_costs = [], []
	fixed_centers = fixed_centers or {}
	for i in range(n_labels):
		if i in fixed_centers:
			candidates.append(np.asarray(fixed_centers[i], dtype=float).reshape(1, 2))
			static_costs.append(np.zeros(1))
			continue
		size = label_sizes[i]
		ring_distance = bead_radii[i] + gap + ring_steps * 0.5 * size[1]
		centers = (bead_positions[i][None, None] + ring_distance[:, None, None] * directions[None]
				   + corner[None] * size / 2).reshape(-1, 2) # (rings * directions, 2)
		points = (centers[:, None] + grid[None] * size).reshape(-1, 2)
		n_points = len(grid)

		def fraction(inside):
			return inside.reshape(len(centers), n_points).mean(axis=1)

		cost = np.zeros(len(centers))
		# beads, including the own one when a label would be pushed into it
		d = np.linalg.norm(points[:, None] - bead_positions[None], axis=2)
		cost += w_bead * fraction(np.any(d < bead_radii[None] + gap, axis=1))
		if len(circles):
			d = np.linalg.norm(points[:, None] - circles[None, :, :2], axis=2)
			cost += w_circle * fraction(np.any(d < circles[None, :, 2] + gap, axis=1))
		if len(segments):
			d = _point_segment_distances(points, segments[:, :4].reshape(-1, 2, 2))
			cost += w_segment * fraction(np.any(d < segments[None, :, 4] + gap, axis=1))
		outside = (points[:, 0] < 0) | (points[:, 1] < 0) | (points[:, 0] > canvas_size[0]) | (points[:, 1] > canvas_size[1])
		cost += w_canvas * fraction(outside)
		cost += w_ring * np.repeat(ring_steps, n_directions)
		cost += w_direction * np.tile((1 - np.cos(angles - angles[0])) / 2, n_rings)
		candidates.append(centers)
		static_costs.append(cost)

	def label_overlap(i, centers, choice):
		'''Intersection area of candidate labels of label i with all other placed labels, relative to the label area.'''
		others = [j for j in range(n_labels) if j != i]
		if not others:
			return np.zeros(len(centers))
		other_centers = np.array([candidates[j][choice[j]] for j in others])
		half = (label_sizes[i] + gap) / 2
		other_half = (label_sizes[others] + gap) / 2
		overlap = np.clip(np.minimum(centers[:, None] + half, other_centers[None] + other_half)
						  - np.maximum(centers[:, None] - half, other_centers[None] - other_half), 0, None)
		return np.prod(overlap, axis=2).sum(axis=1) / np.prod(label_sizes[i] + gap)

	choice = [int(np.argmin(cost)) for cost in static_costs]
	for _ in range(max_sweeps):
		costs = [static_costs[i] + w_label * label_overlap(i, candidates[i], choice) for i in range(n_labels)]
		moved = False
		# the labels in the worst position move first
		for i in sorted(range(n_labels), key=lambda i: -costs[i][choice[i]]):
			cost = static_costs[i] + w_label * label_overlap(i, candidates[i], choice)
			best = int(np.argmin(cost))
			if cost[best] < cost[choice[i]] - 1e-9:
				choice[i] = best
				moved = True
		if not moved:
			break
	return np.array([candidates[i][choice[i]] for i in range(n_labels)])

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

def _parse_wvs(bead_graph, atom):
	'''
	Return the ``wVS`` annotation of an atom as float, or None if it has none.
	'''
	value = bead_graph.nodes[atom].get('wVS')
	if value is None:
		return None
	try:
		return float(value)
	except (TypeError, ValueError):
		raise ValueError(f"wVS='{value}' of atom {bead_graph.nodes[atom].get('element')} "
						 "is not a number.") from None

def _vs_weight(bead_graph, atom):
	'''
	Weight of an atom of a parent bead when it is used for a virtual site (mirrors fast_forward).
	The atom's own wVS if given; for a hydrogen without one, the wVS of the heavy atom
	it is bonded to; otherwise the atom's normal mapping weight (w, default 1).
	'''
	wvs = _parse_wvs(bead_graph, atom)
	if wvs is None and bead_graph.nodes[atom].get('element') == 'H':
		for anchor in bead_graph.neighbors(atom):
			if bead_graph.nodes[anchor].get('element') != 'H':
				wvs = _parse_wvs(bead_graph, anchor)
				break
	if wvs is None:
		return bead_graph.nodes[atom].get('weight', 1.0)
	return wvs

def _vs_atom_weights(cg_graph, vnode, virtual_nodes):
	'''
	Atoms that determine the position of a virtual site and their weights.
	The atoms are those of the neighboring beads that are not virtual sites. An atom
	shared between several of these beads gets the sum of its weights.

	Returns
	dict
		{atom node in the AA graph: weight}
	'''
	parents = [n for n in cg_graph.neighbors(vnode) if n not in virtual_nodes]
	if not parents:
		data = cg_graph.nodes[vnode]
		raise ValueError(f"Virtual site {data.get('n', data.get('fragname'))} (bead {vnode}) "
						 "is not bonded to any bead with atoms, so its position is undefined.")
	weights = {}
	for parent in parents:
		bead_graph = cg_graph.nodes[parent]['graph']
		for atom in bead_graph.nodes:
			weights[atom] = weights.get(atom, 0.0) + _vs_weight(bead_graph, atom)
	return weights

def _get_vs_coordinate(atom_weights, coord_of, aa_graph, vnode, include_hydrogens=True):
	'''
	Weighted average of the atom coordinates given by coord_of(atom) for a virtual site.
	'''
	coords, weights = [], []
	for atom, weight in atom_weights.items():
		if not include_hydrogens and aa_graph.nodes[atom]['element'] == 'H':
			continue
		coords.append(coord_of(atom))
		weights.append(weight)
	if not weights or np.all(np.array(weights) == 0):
		raise ValueError(f"All wVS weights are zero for virtual site {vnode}. Cannot compute its position.")
	return np.average(coords, axis=0, weights=weights)

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