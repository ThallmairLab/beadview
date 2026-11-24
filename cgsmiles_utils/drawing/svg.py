'''
SVG elements for beads, interactions and labels.
'''
from xml.sax.saxutils import escape, quoteattr

import numpy as np

from .geometry import distance_to_skeleton
from .labels import LABEL_FONT_SIZE, label_lines


def svg_id(prefix, name):
	'''id attribute (including the leading space) for an SVG element.'''
	return f' id={quoteattr(f"{prefix}_{name}")}'


def bead_circle(bead_position, bead_type, element_id, scale_factor, dashed=False, border_width=1.75, border_color='#6e6e6e',
				fill_alpha=0.5, border_alpha=0.5):
	'''
	SVG element for a bead drawn as a circle.

	Parameters
	----------
	bead_position : list or np.array
		The (x, y) position of the bead.
	bead_type : BeadType
		The type of the bead, determines size and color.
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
	r = bead_type.radius * scale_factor
	border_width = border_width * scale_factor
	# the filling ends where the border starts, so the two do not overlap
	svg = [f'<g{svg_id("circle_bead", element_id)}>\n']
	svg.append(f'\t<circle style="fill:{bead_type.color};fill-opacity:{fill_alpha};stroke:none" cx="{x}" cy="{y}" r="{max(r - border_width / 2, 0)}" />\n')
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


def bead_outline(atom_positions, bonds, bead_type, element_id, width, border_width, border_color='#000000',
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
	bead_type : BeadType
		The type of the bead, determines the color.
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
	svg.append(f'\t<path style="fill:{bead_type.color};fill-opacity:{fill_alpha};fill-rule:evenodd;stroke:none" d="{inner_path}" />\n')
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
