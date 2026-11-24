'''
Drawing of a CGMapping: the atomistic molecule (RDKit), the beads, the interactions of the loaded ITP and the bead labels.
'''
import io
import subprocess
import sys
from dataclasses import dataclass

import numpy as np
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Geometry.rdGeometry import Point2D

from ..beads import BeadType, bead_positions as compute_bead_positions, detect_virtual_nodes
from ..rdkit_conv import cgsmiles_to_rdkit
from . import svg
from .geometry import flip_molecule, rotate_molecule, most_vertical_bonds, offset_path, offset_side
from .labels import label_extent, label_above_bead, optimize_label_positions
from .options import REFERENCE_SCALE


@dataclass
class DrawnBead:
	'''A bead as it is drawn: cg_graph node, drawing position and what is shown for it.'''
	node: int
	position: np.ndarray
	bead_type: BeadType
	name: str | None
	element_id: str # unique name used in the ids of the SVG elements
	dashed: bool = False # virtual site in the loaded ITP

	def radius(self, scale_factor):
		return self.bead_type.radius * scale_factor


class Renderer:
	'''
	Draws a CGMapping with the given DrawOptions, see Renderer.svg().
	'''
	def __init__(self, mapping, opts):
		self.mapping = mapping
		self.opts = opts
		self.mol, self.idx_map = self._build_mol()
		self.drawer = self._make_drawer()

	@property
	def topology(self):
		return self.mapping.topology

	def coords(self, atom):
		'''Drawing coordinates of an atom of the aa_graph.'''
		return self.drawer.GetDrawCoords(self.idx_map[atom])

	def _build_mol(self):
		mol, idx_map = cgsmiles_to_rdkit(self.mapping.aa_graph, add_hydrogens=self.opts.show_hydrogens)
		rdDepictor.Compute2DCoords(mol)
		if self.opts.flip:
			flip_molecule(mol)
		rotate_molecule(mol, self.opts.rotate_by)
		return mol, idx_map

	def _make_drawer(self):
		opts = self.opts
		drawer = rdMolDraw2D.MolDraw2DSVG(*opts.canvas_size)
		# SetScale only takes integer sizes, so scale factors are effectively rounded to multiples of 1/REFERENCE_SCALE
		size = max(int(round(opts.scale_factor * REFERENCE_SCALE)), 1)
		drawer.SetScale(size, size, Point2D(1000, 1000), Point2D(-1000, -1000))
		o = drawer.drawOptions()
		o.addAtomIndices = opts.show_atom_indices
		o.bondLineWidth = opts.bond_width * opts.scale_factor
		o.clearBackground = not opts.remove_background
		if not opts.color_atoms:
			o.useBWAtomPalette()
		return drawer

	def svg(self):
		'''The drawing as SVG string.'''
		opts = self.opts
		if opts.check_orientation:
			angle, vertical_bonds = most_vertical_bonds(self.mol, tolerance=0.02)
			if abs(angle) < 0.5:
				print(f"Found vertical lines with angle {angle:+.2f} degrees")
			else:
				print(f"No vertical lines found, min angle: {angle:+.2f} degrees")
			self.drawer.drawOptions().continuousHighlight = False # draw the highlighted bonds in red instead of underlaying them
			self.drawer.DrawMolecule(self.mol, highlightAtoms=[], highlightBonds=vertical_bonds,
									 highlightBondColors={bond: (1.0, 0.0, 0.0) for bond in vertical_bonds})
		else:
			self.drawer.DrawMolecule(self.mol)
		self.drawer.FinishDrawing()
		text = self.drawer.GetDrawingText()
		if opts.show_mapping:
			text = text.replace('</svg>', self._mapping_svg() + '</svg>')
		return text

	# beads
	def _layout_beads(self):
		'''
		Positions of all beads in drawing coordinates and the beads that are drawn.

		Returns
		-------
		positions : dict
			Drawing coordinates of all beads (including hidden virtual sites), keyed by cg_graph node.
		drawn : list of DrawnBead
			The beads that are drawn, virtual sites last.
		hidden : set
			cg_graph nodes that are not drawn.
		'''
		cg_graph = self.mapping.cg_graph
		positions = compute_bead_positions(cg_graph, self.mapping.aa_graph, self.coords,
										   include_hydrogens=self.opts.hydrogens_in_bead_position)
		itp_vs = self.topology.virtual_sites(cg_graph) if self.opts.indicate_vs and self.topology is not None else set()
		hidden = set() if self.opts.show_vs else set(detect_virtual_nodes(cg_graph))
		drawn, seen_ids = [], set()
		for node, position in positions.items():
			if node in hidden:
				continue
			data = cg_graph.nodes[node]
			name = data.get('n')
			element_id = name if name is not None else data['fragname']
			if element_id in seen_ids: # unnamed beads of the same type or duplicate names
				element_id = f'{element_id}_{node}'
			seen_ids.add(element_id)
			drawn.append(DrawnBead(node, position, BeadType(data['fragname']), name, element_id, dashed=node in itp_vs))
		return positions, drawn, hidden

	def _bead_svg(self, bead):
		'''SVG of a drawn bead: an outline along its drawn atoms (style='outline') or a circle.'''
		opts = self.opts
		atoms = []
		if opts.style == 'outline':
			atoms = [atom for atom in self.mapping.cg_graph.nodes[bead.node]['graph'].nodes if atom in self.idx_map]
		if atoms: # beads without drawn atoms (e.g. virtual sites) are drawn as circles
			atom_positions = {atom: list(self.coords(atom)) for atom in atoms}
			bonds = list(self.mapping.aa_graph.subgraph(atoms).edges)
			return svg.bead_outline(atom_positions, bonds, bead.bead_type, bead.element_id,
				opts.outline_width * opts.scale_factor, opts.border_width * opts.scale_factor,
				opts.border_color, fill_alpha=opts.fill_alpha, border_alpha=opts.border_alpha)
		return svg.bead_circle(bead.position, bead.bead_type, bead.element_id, opts.scale_factor, dashed=bead.dashed,
			border_width=opts.border_width, border_color=opts.border_color,
			fill_alpha=opts.fill_alpha, border_alpha=opts.border_alpha)

	def _mapping_svg(self):
		'''SVG of the beads, the interactions of the loaded ITP and the bead labels.'''
		opts = self.opts
		positions, drawn, hidden = self._layout_beads()
		beads = [self._bead_svg(bead) for bead in drawn]

		below, above = [], [] # bonds and angles are drawn below the beads, dihedrals on top
		if opts.show_bonds:
			below += self._bonds_svg(positions, hidden)
		angles, dihedrals = self._angles_dihedrals_svg(positions, hidden)
		below += angles
		above += dihedrals

		labels = []
		if opts.show_bead_labels:
			anchors = self._label_anchors(drawn, positions, hidden)
			labels = [svg.bead_label(anchor, bead.bead_type, bead.element_id, bead_name=bead.name,
									 scale_factor=opts.scale_factor,
									 type_color=opts.bead_type_label_color,
									 name_color=opts.bead_name_label_color)
					  for bead, anchor in zip(drawn, anchors)]
		return ''.join(below) + ''.join(beads) + ''.join(above) + ''.join(labels)

	# interactions
	def _interactions(self, interaction_type, hidden):
		'''cg_graph nodes of the interactions of a type in the loaded ITP (none without ITP), see Topology.interactions.'''
		if self.topology is None:
			return []
		return self.topology.interactions(self.mapping.cg_graph, interaction_type, hidden)

	def _bonds_svg(self, positions, hidden):
		'''SVG lines for the bonds and constraints of the loaded ITP.'''
		opts = self.opts
		lines = []
		for interaction_type, color in (('bonds', opts.bond_color), ('constraints', opts.constraint_color)):
			for i, j in self._interactions(interaction_type, hidden):
				lines.append(svg.line(positions[i], positions[j], color=color, width=opts.cg_line_width * opts.scale_factor))
		return lines

	def _angles_dihedrals_svg(self, positions, hidden):
		'''
		SVG polylines for the angles and dihedrals (including impropers) of the loaded ITP, depending on
		opts.show_angles and opts.show_dihedrals. Each one is drawn as a line along its bonds, shifted by
		opts.cg_line_offset away from the centroid of all beads. Where lines would overlap
		on a bond, the offset of the later line is increased in steps of opts.line_spacing; angles are placed first.

		Returns
		-------
		tuple of list of str
			The SVG polyline elements for the angles and for the dihedrals.
		'''
		opts = self.opts
		centroid = np.mean(list(positions.values()), axis=0)
		spacing = opts.line_spacing * opts.scale_factor
		offset = opts.cg_line_offset * opts.scale_factor
		occupied = {} # (bond, side) -> distances of the lines already placed along that bond on that side
		def place(nodes, color):
			points = [positions[n] for n in nodes]
			side = offset_side(points, offset, centroid)
			# side relative to each bond in a fixed direction (lower node first), so lines on different paths can be compared
			lanes = [((a, b), side) if a < b else ((b, a), -side) for a, b in zip(nodes[:-1], nodes[1:])]
			taken = [d for lane in lanes for d in occupied.get(lane, [])]
			distance = abs(offset)
			while any(abs(distance - d) < spacing - 1e-9 for d in taken):
				distance += spacing
			for lane in lanes:
				occupied.setdefault(lane, []).append(distance)
			return svg.polyline(offset_path(points, distance, side), color=color, width=opts.cg_line_width * opts.scale_factor)

		angles, dihedrals = [], []
		if opts.show_angles:
			angles = [place(nodes, opts.angle_color) for nodes in self._interactions('angles', hidden)]
		if opts.show_dihedrals:
			dihedrals = [place(nodes, opts.dihedral_color) for nodes in self._interactions('dihedrals', hidden)]
		return angles, dihedrals

	# labels
	def _label_anchors(self, drawn, positions, hidden):
		'''
		Anchor (left end of the baseline of the last line) of the label of each drawn bead.
		Labels of dummy beads are placed right above the bead; the others are placed by
		opts.bead_label_position.
		'''
		opts = self.opts
		if opts.bead_label_position == 'optimize':
			dummies = [i for i, bead in enumerate(drawn) if bead.bead_type.is_dummy]
			return self._optimized_label_anchors(drawn, positions, hidden, fixed=dummies)
		return [label_above_bead(bead.position, bead.bead_type, bead.name, opts.scale_factor) if bead.bead_type.is_dummy
				else tuple(opts.bead_label_position(bead.position, bead.radius(opts.scale_factor), bead.name))
				for bead in drawn]

	def _optimized_label_anchors(self, drawn, positions, hidden, fixed=()):
		'''
		Label anchors that minimize the overlap of the bead labels with each other, with the beads,
		with the bonds and atom labels of the molecule and with the drawn bonds, angles and dihedrals.
		The labels of the beads with the indices in fixed are placed right above the bead instead.
		'''
		opts, mol = self.opts, self.mol
		atoms = np.array([list(self.drawer.GetDrawCoords(i)) for i in range(mol.GetNumAtoms())]).reshape(-1, 2)
		bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
		bond_length = np.median([np.linalg.norm(atoms[i] - atoms[j]) for i, j in bonds]) if bonds else REFERENCE_SCALE * opts.scale_factor
		# atoms drawn with a label (heteroatoms, or every atom when indices are shown); wider if hydrogens are attached
		circles = [(*atoms[a.GetIdx()], (0.4 if a.GetTotalNumHs() else 0.3) * bond_length) for a in mol.GetAtoms()
				   if a.GetSymbol() != 'C' or opts.show_atom_indices]
		segments = [(*atoms[i], *atoms[j], opts.bond_width * opts.scale_factor) for i, j in bonds]

		# drawn bonds, angles and dihedrals as segments between the beads, as wide as the lines run next to them
		cg_width = opts.cg_line_width * opts.scale_factor
		for option, interaction_types, width in (
				('show_bonds', ('bonds', 'constraints'), cg_width / 2),
				('show_angles', ('angles',), opts.cg_line_offset * opts.scale_factor + cg_width),
				('show_dihedrals', ('dihedrals',), opts.cg_line_offset * opts.scale_factor + cg_width)):
			if not getattr(opts, option):
				continue
			for interaction_type in interaction_types:
				for nodes in self._interactions(interaction_type, hidden):
					segments += [(*positions[a], *positions[b], abs(width)) for a, b in zip(nodes[:-1], nodes[1:])]

		extents = [label_extent(bead.bead_type, bead.name, opts.scale_factor) for bead in drawn]
		fixed_centers = {}
		for i in fixed:
			x, y = label_above_bead(drawn[i].position, drawn[i].bead_type, drawn[i].name, opts.scale_factor)
			width, above, below = extents[i]
			fixed_centers[i] = (x + width / 2, y - (above - below) / 2)
		centers = optimize_label_positions(
			bead_positions=[bead.position for bead in drawn],
			bead_radii=[bead.radius(opts.scale_factor) for bead in drawn],
			label_sizes=[(width, above + below) for width, above, below in extents],
			circles=circles,
			segments=segments,
			canvas_size=opts.canvas_size,
			gap=2 * opts.scale_factor,
			fixed_centers=fixed_centers)
		# convert label centers to text anchors (left end of the baseline of the last line)
		return [(center[0] - width / 2, center[1] + (above - below) / 2) for center, (width, above, below) in zip(centers, extents)]


def svg_to_png(svg_text, dpi=300):
	'''
	Convert an SVG string to PNG bytes with cairosvg. cairosvg runs in a separate process because its
	cairo library crashes (segmentation fault) when RDKit's drawing code was loaded first in the same process.
	'''
	code = ('import sys, cairosvg; '
			'sys.stdout.buffer.write(cairosvg.svg2png(bytestring=sys.stdin.buffer.read(), dpi=int(sys.argv[1])))')
	result = subprocess.run([sys.executable, '-c', code, str(dpi)], input=svg_text.encode(), capture_output=True)
	if result.returncode != 0:
		raise RuntimeError(f"Converting the SVG to PNG with cairosvg failed: {result.stderr.decode().strip()}")
	return result.stdout


def show_svg(svg_text, ax=None):
	'''Show an SVG string on a matplotlib axis (created if None).'''
	import matplotlib.pyplot as plt
	from PIL import Image
	png = svg_to_png(svg_text, dpi=300)
	if ax is None:
		_, ax = plt.subplots()
	ax.imshow(Image.open(io.BytesIO(png)))
	ax.axis("off")
	return ax


def show_cgsmiles_drawing(aa_graph, opts, ax=None):
	'''Draw the all-atom molecule with cgsmiles' own drawing function (only show_hydrogens and scale_factor are used).'''
	import matplotlib.pyplot as plt
	import pysmiles
	from cgsmiles.drawing import draw_molecule
	if ax is None:
		_, ax = plt.subplots(1, 1)
	if opts.show_hydrogens:
		draw_molecule(aa_graph, ax=ax, scale=opts.scale_factor)
		return ax
	aa_graph = aa_graph.copy() # remove_explicit_hydrogens works in place
	pysmiles.remove_explicit_hydrogens(aa_graph)
	labels = {}
	for node, data in aa_graph.nodes(data=True):
		hcount = data.get('hcount', 0)
		labels[node] = data.get('element', '*') + (f"H{hcount}" if hcount > 0 else '')
	draw_molecule(aa_graph, ax=ax, labels=labels, scale=opts.scale_factor, align_with='x')
	return ax
