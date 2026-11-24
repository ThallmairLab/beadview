'''
SVG elements for beads, interactions and labels, and fitting the canvas to the drawing.
'''
import math
import re
from xml.sax.saxutils import escape, quoteattr

import numpy as np

from .geometry import distance_to_skeleton
from .labels import LABEL_FONT_SIZE, label_lines


def svg_id(prefix, name):
	'''id attribute (including the leading space) for an SVG element.'''
	return f' id={quoteattr(f"{prefix}_{name}")}'


def bead_circle(bead_position, radius, color, element_id, scale_factor, dashed=False, border_width=1.75, border_color='#6e6e6e',
				fill_alpha=0.5, border_alpha=0.5):
	'''
	SVG element for a bead drawn as a circle.

	Parameters
	----------
	bead_position : list or np.array
		The (x, y) position of the bead.
	radius : float
		Radius of the circle at scale_factor=1.
	color : str
		Fill color.
	element_id : str
		Unique name of the bead, used in the id of the SVG element.
	dashed : bool
		Draw the border dashed (used for virtual sites).
	border_width : float
		Width of the border at scale_factor=1.
	border_color : str
		Color of the border.
	fill_alpha, border_alpha : float
		Opacity of the filling and of the border.

	Returns
	-------
	str
		The SVG group element as a string.
	'''
	dasharray = f'{2.5 * scale_factor},{1.75 * scale_factor}' if dashed else 'none'
	x, y = bead_position[0], bead_position[1]
	r = radius * scale_factor
	border_width = border_width * scale_factor
	# the filling ends where the border starts, so the two do not overlap
	svg = [f'<g{svg_id("circle_bead", element_id)}>\n']
	svg.append(f'\t<circle style="fill:{color};fill-opacity:{fill_alpha};stroke:none" cx="{x}" cy="{y}" r="{max(r - border_width / 2, 0)}" />\n')
	svg.append(f'\t<circle style="fill:none;stroke:{border_color};stroke-opacity:{border_alpha};stroke-width:{border_width};stroke-dasharray:{dasharray}" cx="{x}" cy="{y}" r="{r}" />\n')
	svg.append('</g>\n')
	return ''.join(svg)


def _contour_path(x, y, distance, level):
	'''
	SVG path data of the closed contour lines where the distance equals level.
	'''
	from contourpy import contour_generator
	lines = contour_generator(x, y, distance).lines(level)
	return ' '.join('M ' + ' L '.join(f'{px:.2f},{py:.2f}' for px, py in line) + ' Z' for line in lines if len(line) > 2)


def bead_outline(atom_positions, bonds, color, element_id, width, border_width, border_color='#000000',
				 fill_alpha=0.5, border_alpha=0.5):
	'''
	SVG element for a bead drawn as a rounded outline along the bonds between its atoms.
	Atoms that are not bonded to another atom of the bead are drawn as dots.
	The filling and the border are computed as shapes (the region closer than width/2 to the bonds and the ring
	around it), so overlapping segments do not add up and the two do not overlap.

	Parameters
	----------
	atom_positions : dict
		Drawing coordinates (x, y) of the atoms of the bead, keyed by atom.
	bonds : list of tuple
		Pairs of atoms (keys of atom_positions) that are bonded.
	color : str
		Fill color.
	element_id : str
		Unique name of the bead, used in the id of the SVG element.
	width : float
		Width of the outline (without border).
	border_width : float
		Width of the border around the outline.
	border_color : str
		Color of the border around the outline.
	fill_alpha, border_alpha : float
		Opacity of the filling and of the border.

	Returns
	-------
	str
		The SVG group element as a string.
	'''
	bonded = {atom for bond in bonds for atom in bond}
	segments = [(atom_positions[i], atom_positions[j]) for i, j in bonds]
	points = [atom_positions[atom] for atom in atom_positions if atom not in bonded]
	inner, outer = width / 2, width / 2 + border_width
	# distance field on a grid that is fine compared to the border and extends beyond the outline
	step = max(min(border_width, width) / 4, 0.05) if border_width > 0 else max(width / 40, 0.05)
	coords = np.array(list(atom_positions.values()), dtype=float).reshape(-1, 2)
	low, high = coords.min(axis=0) - outer - 2 * step, coords.max(axis=0) + outer + 2 * step
	x, y = np.meshgrid(np.arange(low[0], high[0] + step, step), np.arange(low[1], high[1] + step, step))
	distance = distance_to_skeleton(x, y, segments, points)
	inner_path = _contour_path(x, y, distance, inner)
	svg = [f'<g{svg_id("outline_bead", element_id)}>\n']
	svg.append(f'\t<path style="fill:{color};fill-opacity:{fill_alpha};fill-rule:evenodd;stroke:none" d="{inner_path}" />\n')
	if border_width > 0:
		ring_path = _contour_path(x, y, distance, outer) + ' ' + inner_path
		svg.append(f'\t<path style="fill:{border_color};fill-opacity:{border_alpha};fill-rule:evenodd;stroke:none" d="{ring_path}" />\n')
	svg.append('</g>\n')
	return ''.join(svg)


def line(start, end, color='#000000', width=2.0):
	'''
	SVG line element, used for bonds and constraints between two beads.
	'''
	return (f'<line\n'
			f'\tstyle="fill:none;stroke:{color};stroke-width:{width};stroke-linecap:round"\n'
			f'\tx1="{start[0]}"\n'
			f'\ty1="{start[1]}"\n'
			f'\tx2="{end[0]}"\n'
			f'\ty2="{end[1]}" />\n')


def polyline(points, color='#000000', width=2.0):
	'''
	SVG polyline element through the given (x, y) points, used for angles and dihedrals.
	'''
	points_str = ' '.join(f'{p[0]},{p[1]}' for p in points)
	return (f'<polyline\n'
			f'\tstyle="fill:none;stroke:{color};stroke-width:{width};stroke-linecap:round;stroke-linejoin:round"\n'
			f'\tpoints="{points_str}" />\n')


def bead_label(anchor, bead_type, element_id, bead_name=None, scale_factor=1.0, type_color='#666666', name_color='#000000'):
	'''
	SVG text element for a bead label: the bead name (if given) above the bead type.
	Dummy beads with a name only show the name.

	Parameters
	----------
	anchor : tuple of float
		(x, y) of the left end of the baseline of the last line of the label.
	bead_type : BeadType
		The bead type.
	element_id : str
		Unique name of the bead, used in the id of the SVG element.
	bead_name : str, optional, default: None
		The bead name.
	type_color : str, optional, default: '#666666'
		The color of the bead type text.
	name_color : str, optional, default: '#000000'
		The color of the bead name text.

	Returns
	-------
	str
		The SVG text element as a string.
	'''
	fontsize = LABEL_FONT_SIZE * scale_factor
	x, y = anchor
	lines = label_lines(bead_type, bead_name)
	text_svg = ['<text\n']
	text_svg.append(f'\t{svg_id("text_bead", element_id).strip()}\n')
	text_svg.append(f'\tstyle="font-size:{fontsize}px;text-anchor:start;font-family:sans-serif;"\n')
	text_svg.append(f'\tx="{x}"\n')
	text_svg.append(f'\ty="{y}">\n')
	for i, (text, is_name) in enumerate(lines):
		line_y = y - (len(lines) - 1 - i) * fontsize
		color = name_color if is_name else type_color
		text_svg.append(f'\t<tspan x="{x}" y="{line_y}" font-size="{fontsize}px" fill="{color}" font-family="sans-serif">{escape(text)}</tspan>\n')
	text_svg.append('</text>\n')
	return ''.join(text_svg)


_SHAPE = re.compile(r'<(path|circle|ellipse|line|polyline)\b[^>]*>')
_NUMBER = r'-?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?'


def _attribute(tag, name):
	match = re.search(rf'\s{name}\s*=\s*([\'"])(.*?)\1', tag, re.S)
	return match.group(2) if match else None


def _path_points(d):
	'''Points of SVG path data with absolute commands (M, L, Q, C, S, T, H, V, A, Z), as RDKit and this package write them.'''
	points, x, y = [], 0.0, 0.0
	for command, args in re.findall(r'([A-Za-z])([^A-Za-z]*)', d):
		numbers = [float(n) for n in re.findall(_NUMBER, args)]
		if command == 'H':
			points += [(n, y) for n in numbers]
		elif command == 'V':
			points += [(x, n) for n in numbers]
		elif command == 'A': # rx ry rotation large-arc sweep x y
			points += [(numbers[i + 5], numbers[i + 6]) for i in range(0, len(numbers) - 6, 7)]
		else:
			points += list(zip(numbers[0::2], numbers[1::2]))
		if points:
			x, y = points[-1]
	return points


def content_bbox(svg_text, extra_boxes=()):
	'''
	Bounding box (x_min, y_min, x_max, y_max) of the shapes (path, circle, ellipse, line, polyline) of an SVG
	and of extra_boxes (e.g. text). The background rectangle and stroke widths are not included.
	'''
	boxes = list(extra_boxes)
	for match in _SHAPE.finditer(svg_text):
		kind, tag = match.group(1), match.group(0)
		if kind == 'path':
			points = _path_points(_attribute(tag, 'd') or '')
		elif kind == 'polyline':
			numbers = [float(n) for n in re.findall(_NUMBER, _attribute(tag, 'points') or '')]
			points = list(zip(numbers[0::2], numbers[1::2]))
		elif kind == 'line':
			points = [(float(_attribute(tag, 'x1')), float(_attribute(tag, 'y1'))),
					  (float(_attribute(tag, 'x2')), float(_attribute(tag, 'y2')))]
		else:
			cx, cy = float(_attribute(tag, 'cx')), float(_attribute(tag, 'cy'))
			rx = float(_attribute(tag, 'r') or _attribute(tag, 'rx'))
			ry = float(_attribute(tag, 'r') or _attribute(tag, 'ry'))
			points = [(cx - rx, cy - ry), (cx + rx, cy + ry)]
		if points:
			xs, ys = zip(*points)
			boxes.append((min(xs), min(ys), max(xs), max(ys)))
	if not boxes:
		return None
	x_min, y_min, x_max, y_max = zip(*boxes)
	return min(x_min), min(y_min), max(x_max), max(y_max)


def fit_canvas(svg_text, extra_boxes=(), padding=10.0):
	'''
	Resize the canvas of an SVG written by RDKit to the bounding box of its content (see content_bbox) plus padding,
	by setting width, height and viewBox and moving the background rectangle (if any).
	'''
	box = content_bbox(svg_text, extra_boxes)
	if box is None:
		return svg_text
	x0, y0 = math.floor(box[0] - padding), math.floor(box[1] - padding)
	width, height = math.ceil(box[2] + padding) - x0, math.ceil(box[3] + padding) - y0
	svg_text = re.sub(r"width='[^']*' height='[^']*' viewBox='[^']*'",
					  f"width='{width}px' height='{height}px' viewBox='{x0} {y0} {width} {height}'", svg_text, count=1)
	return re.sub(r"(<rect\b[^>]*?)width='[^']*' height='[^']*' x='[^']*' y='[^']*'",
				  rf"\g<1>width='{width}' height='{height}' x='{x0}' y='{y0}'", svg_text, count=1)
