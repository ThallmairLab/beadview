import cgsmiles
from cgsmiles.drawing import draw_molecule
from fast_forward.itp_parser_sub import read_itp
import pysmiles

import warnings, io, re, string, os, logging, copy
from pathlib import Path
from collections.abc import Callable
from dataclasses import dataclass, replace
import numpy as np
import matplotlib.pyplot as plt
import MDAnalysis as mda
from vermouth.forcefield import ForceField
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Geometry.rdGeometry import Point2D
from rdkit.Chem import rdDepictor, MolToSmiles, SanitizeMol, AllChem, MolToPDBBlock
from rdkit.Chem.Draw import IPythonConsole
from IPython.display import display
from pkasolver.query import draw_pka_map, draw_pka_reactions, calculate_microstate_pka_values
from PIL import Image
import cairosvg

from .utils import cgsmiles_to_rdkit
from .drawing_utils import _bead_radius, _flip_molecule, _rotate_molecule_around_center, _most_vertical_bonds, _get_bead_coordinate, _vs_atom_weights, _get_vs_coordinate, detect_virtual_nodes, _create_bead_circle, _create_bead_outline, _create_bead_label, _bead_label_extent, _label_above_bead, _optimize_label_positions, _create_bond_line, _create_polyline, _offset_path, _offset_side, _strip_background
from .itp_utils import itp_writer, add_comments_itp

REFERENCE_SCALE = 25
BOX_SIZE = 100.0 # Angstrom, box written to the generated structure files
MARTINI_MASSES = {'T': 36.0, 'S': 54.0, 'R': 72.0} # default Martini 3 masses per bead size

def _fragname_to_type(fragname):
	'''
	Bead type of a fragment name: a trailing uppercase letter (used to make fragment names unique,
	e.g. TC5A) is removed, except for single letter names like U.
	'''
	if len(fragname) > 1 and fragname[-1].isupper():
		return fragname[:-1]
	return fragname

def _bead_size(bead_type):
	'''Size letter (T, S or R) of a Martini bead type, None for dummy beads.'''
	if bead_type.startswith('U'):
		return None
	return bead_type[0] if bead_type[0] in ('T', 'S') else 'R'

def _split_top_level(text, sep=','):
	'''Split text at sep, ignoring separators inside brackets.'''
	parts, depth, start = [], 0, 0
	for i, char in enumerate(text):
		if char in '[({':
			depth += 1
		elif char in '])}':
			depth -= 1
		elif char == sep and depth == 0:
			parts.append(text[start:i])
			start = i + 1
	parts.append(text[start:])
	return parts

_BEAD_PATTERN = re.compile(r'\[#([^\];]+)([^\]]*)\]') # [#fragname;annotations]

class suppress_output:
    def __enter__(self):
        # open devnull
        self._null_fd = os.open(os.devnull, os.O_RDWR)
        # save real stdout/stderr file descriptors
        self._save_stdout = os.dup(1)
        self._save_stderr = os.dup(2)
        # replace them with devnull
        os.dup2(self._null_fd, 1)
        os.dup2(self._null_fd, 2)

    def __exit__(self, exc_type, exc_value, traceback):
        # restore stdout/stderr
        os.dup2(self._save_stdout, 1)
        os.dup2(self._save_stderr, 2)
        os.close(self._null_fd)
        os.close(self._save_stdout)
        os.close(self._save_stderr)

@dataclass
class DrawOptions:
	"""
	Options for drawing a CGMapping. Lengths are in pixels at scale_factor=1 and are multiplied by scale_factor.
	"""
	# canvas
	canvas_size: tuple = (300, 300)
	scale_factor: float = 1
	rotate_by: float = 0 # rotation of the molecule in degrees
	flip: bool = False # mirror the molecule horizontally
	remove_background: bool = True # transparent instead of white background

	# atomistic molecule
	show_hydrogens: bool = False
	color_atoms: bool = False # color heteroatoms by element instead of black and white
	show_atom_indices: bool = False
	bond_width: float = 1.5

	# beads
	show_mapping: bool = True # draw the beads; all following options need it
	style: str = 'beads' # 'beads': circles at the bead positions, 'outline': rounded outlines along the bonds of the atoms of each bead
	outline_width: float = 12.0 # style='outline' only
	border_width: float = 1.5 # border of the bead circles or outlines
	border_color: str = '#000000'
	border_alpha: float = 0.5 # opacity of the bead borders
	fill_alpha: float = 0.5 # opacity of the bead filling
	include_hydrogens_in_bead_position: bool | None = None # None follows show_hydrogens
	show_vs: bool = True # draw virtual sites
	indicate_vs: bool = True # dashed outline for beads that are virtual sites in the ITP loaded with read_itp()

	# bead labels (labels of dummy beads, type U, are always placed right above the bead)
	show_bead_labels: bool = False
	bead_type_label_color: str = '#666666'
	bead_name_label_color: str = '#000000'
	# function (bead_position, bead_radius, bead_name) -> label position, or 'optimize' to place the labels around
	# their beads with minimal overlap with other labels, beads, the molecule and the drawn bonds/angles/dihedrals
	bead_label_position: Callable | str = 'optimize'

	# interactions from the ITP loaded with read_itp()
	show_bonds: bool = False # bonds and constraints as straight lines between the beads
	show_angles: bool = False # angles as bent lines along the bonds
	show_dihedrals: bool = False # dihedrals as lines along the bonds of the four beads
	bond_color: str = '#000000'
	constraint_color: str = '#020070FF'
	angle_color: str = '#286E8D'
	dihedral_color: str = '#286E8D'
	cg_line_width: float = 2.0 # width of the bond, constraint, angle and dihedral lines
	cg_line_offset: float = 8.0 # distance between angle/dihedral lines and the bonds they follow, positive = away from the centroid of all beads
	line_spacing: float = 4.0 # distance between angle/dihedral lines that would otherwise overlap

	# debugging
	check_orientation: bool = False # report (and highlight in red) the most vertical bond of the molecule

	def __post_init__(self):
		if self.style not in ('beads', 'outline'):
			raise ValueError(f"style must be 'beads' or 'outline', got '{self.style}'")
		for option in ('fill_alpha', 'border_alpha'):
			if not 0 <= getattr(self, option) <= 1:
				raise ValueError(f"{option} must be between 0 and 1, got {getattr(self, option)}")
		if isinstance(self.bead_label_position, str) and self.bead_label_position != 'optimize':
			raise ValueError(f"bead_label_position must be a function or 'optimize', got '{self.bead_label_position}'")
		if self.show_bead_labels and not self.show_mapping:
			warnings.warn("show_bead_labels has no effect when show_mapping=False")
		for option in ('show_bonds', 'show_angles', 'show_dihedrals'):
			if getattr(self, option) and not self.show_mapping:
				warnings.warn(f"{option} has no effect when show_mapping=False")

	@property
	def hydrogens_in_bead_position(self) -> bool:
		# resolved lazily so that dataclasses.replace(opts, show_hydrogens=...) is respected
		if self.include_hydrogens_in_bead_position is None:
			return self.show_hydrogens
		return self.include_hydrogens_in_bead_position


class CGMapping:
	def __init__(self, cgs_string: str):
		self.cgs_string = cgs_string
		self.cg_graph, self.aa_graph = (
			cgsmiles.MoleculeResolver.from_string(cgs_string).resolve()
		)
		self._clean_cg_graph()
		self.ff = None
		self.molname = None

	# Getters and Setters
	def get_smiles(self, add_hydrogens=False):
		opts = DrawOptions(show_hydrogens=add_hydrogens)
		mol, _ = self._build_mol(opts)
		return MolToSmiles(mol, isomericSmiles=True)
	
	# general utility methods
	def _clean_cg_graph(self):
		for n, data in self.cg_graph.nodes(data=True):
			data["fragname"] = _fragname_to_type(data["fragname"])
			data["atype"] = data["fragname"]
			data["atomname"] = data.get("n", data["fragname"])

	def _build_mol(self, opts):
		mol, idx_map = cgsmiles_to_rdkit(self.aa_graph, add_hydrogens=opts.show_hydrogens)
		rdDepictor.Compute2DCoords(mol) # ensure 2D coordinates are present for the full molecule
		if opts.flip:
			_flip_molecule(mol)
		_rotate_molecule_around_center(mol, opts.rotate_by)
		return mol, idx_map

	# drawing methods
	@staticmethod
	def _make_drawer(opts):
		drawer = rdMolDraw2D.MolDraw2DSVG(*opts.canvas_size)
		# SetScale only takes integer sizes, so scale factors are effectively rounded to multiples of 1/REFERENCE_SCALE
		size = max(int(round(opts.scale_factor * REFERENCE_SCALE)), 1)
		drawer.SetScale(size, size, Point2D(1000, 1000), Point2D(-1000, -1000))
		return drawer

	def _draw_beads(self, mol, svg, idx_map, opts):
		drawer_coords = self._make_drawer(opts)
		drawer_coords.DrawMolecule(mol)
		drawer_coords.FinishDrawing()

		virtual_edges = detect_virtual_nodes(self.cg_graph)
		bead_positions = {}
		beads = [] 
		labeled = [] # (position, bead_type, bead_name) of the drawn beads
		labeled_dummies = [] # indices into labeled of the dummy beads
		itp_vs = self._itp_virtual_sites() if opts.indicate_vs and self.ff is not None else set()
		def add_bead(bead, position, bead_type, bead_name):
			# outline the drawn atoms of the bead; beads without drawn atoms (e.g. virtual sites) are drawn as circles
			atoms = [atom for atom in self.cg_graph.nodes[bead]['graph'].nodes if atom in idx_map] if opts.style == 'outline' else []
			if atoms:
				atom_positions = {atom: list(drawer_coords.GetDrawCoords(idx_map[atom])) for atom in atoms}
				bonds = [(i, j) for i, j in self.aa_graph.subgraph(atoms).edges]
				beads.append(_create_bead_outline(atom_positions, bonds, bead_type, bead_name,
					opts.outline_width * opts.scale_factor, opts.border_width * opts.scale_factor,
					opts.border_color, fill_alpha=opts.fill_alpha, border_alpha=opts.border_alpha))
			else:
				beads.append(_create_bead_circle(position, bead_type, bead_name, opts.scale_factor, dashed=bead in itp_vs,
					border_width=opts.border_width, border_color=opts.border_color,
					fill_alpha=opts.fill_alpha, border_alpha=opts.border_alpha))
			if bead_type.startswith('U'):
				labeled_dummies.append(len(labeled))
			labeled.append((position, bead_type, bead_name))

		for bead, data in self.cg_graph.nodes(data=True):
			if bead in virtual_edges:   # virtual nodes are placed afterwards
				continue
			bead_type = data['fragname']
			pos = _get_bead_coordinate(data['graph'], idx_map, drawer_coords, opts, bead)
			bead_positions[bead] = pos
			add_bead(bead, pos, bead_type, data.get('n'))

		for vnode in virtual_edges:
			data = self.cg_graph.nodes[vnode]
			atom_weights = _vs_atom_weights(self.cg_graph, vnode, virtual_edges)
			pos = _get_vs_coordinate(atom_weights, lambda atom: drawer_coords.GetDrawCoords(idx_map[atom]),
									 self.aa_graph, vnode, include_hydrogens=opts.hydrogens_in_bead_position)
			bead_positions[vnode] = pos
			if opts.show_vs:
				add_bead(vnode, pos, data['fragname'], data.get('n'))

		hidden = set() if opts.show_vs else set(virtual_edges)
		below, above = [], [] # bonds and angles are drawn below the beads, dihedrals on top
		if opts.show_bonds:
			below += self._draw_cg_bonds(bead_positions, opts, hidden)
		angles, dihedrals = self._draw_cg_angles_dihedrals(bead_positions, opts, hidden)
		below += angles
		above += dihedrals

		labels = []
		if opts.show_bead_labels:
			optimize = opts.bead_label_position == 'optimize'
			label_positions = [None if optimize else opts.bead_label_position] * len(labeled)
			for i in labeled_dummies:
				_, bead_type, bead_name = labeled[i]
				label_positions[i] = _label_above_bead(bead_type, bead_name, opts.scale_factor)
			if optimize:
				label_positions = self._optimize_bead_labels(mol, drawer_coords, labeled, bead_positions, opts, hidden, fixed=labeled_dummies)
			for (position, bead_type, bead_name), label_position in zip(labeled, label_positions):
				labels.append(_create_bead_label(
					position, bead_type, bead_name=bead_name,
					label_position=label_position,
					scale_factor=opts.scale_factor,
					type_color=opts.bead_type_label_color,
					name_color=opts.bead_name_label_color))
		return svg.replace('</svg>', ''.join(below) + ''.join(beads) + ''.join(above) + ''.join(labels) + '</svg>')

	def _optimize_bead_labels(self, mol, drawer_coords, labeled, bead_positions, opts, hidden=(), fixed=()):
		'''
		Find label positions that minimize the overlap of the bead labels with each other, with the beads,
		with the bonds and atom labels of the molecule and with the drawn bonds, angles and dihedrals.

		Parameters
		----------
		mol : rdkit.Chem.rdchem.Mol
			The drawn molecule.
		drawer_coords : rdMolDraw2D.MolDraw2DSVG
			Drawer the molecule was drawn with, to get the drawing coordinates of the atoms.
		labeled : list of tuple
			(position, bead_type, bead_name) of each drawn bead.
		bead_positions : dict
			Drawing coordinates of the beads, keyed by cg_graph node.
		opts : DrawOptions
		hidden : set
			cg_graph nodes that are not drawn.
		fixed : list of int
			Indices into labeled of the labels that are placed right above their bead instead.

		Returns
		-------
		list of callable
			One label_position function per entry of labeled, see _create_bead_label.
		'''
		atoms = np.array([list(drawer_coords.GetDrawCoords(i)) for i in range(mol.GetNumAtoms())]).reshape(-1, 2)
		bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
		bond_length = np.median([np.linalg.norm(atoms[i] - atoms[j]) for i, j in bonds]) if bonds else REFERENCE_SCALE * opts.scale_factor
		# atoms drawn with a label (heteroatoms, or every atom when indices are shown); wider if hydrogens are attached
		circles = [(*atoms[a.GetIdx()], (0.4 if a.GetTotalNumHs() else 0.3) * bond_length) for a in mol.GetAtoms()
				   if a.GetSymbol() != 'C' or opts.show_atom_indices]
		segments = [(*atoms[i], *atoms[j], opts.bond_width * opts.scale_factor) for i, j in bonds]

		# drawn bonds, angles and dihedrals as segments between the beads, as wide as the lines run next to them
		cg_width = opts.cg_line_width * opts.scale_factor
		if self.ff is not None:
			for option, interaction_types, width in (
					('show_bonds', ('bonds', 'constraints'), cg_width / 2),
					('show_angles', ('angles',), opts.cg_line_offset * opts.scale_factor + cg_width),
					('show_dihedrals', ('dihedrals',), opts.cg_line_offset * opts.scale_factor + cg_width)):
				if not getattr(opts, option):
					continue
				for interaction_type in interaction_types:
					for nodes in self._itp_cg_interactions(interaction_type, hidden):
						segments += [(*bead_positions[a], *bead_positions[b], abs(width)) for a, b in zip(nodes[:-1], nodes[1:])]

		extents = [_bead_label_extent(bead_type, bead_name, opts.scale_factor) for _, bead_type, bead_name in labeled]
		fixed_centers = {}
		for i in fixed:
			position, bead_type, bead_name = labeled[i]
			x, y = _label_above_bead(bead_type, bead_name, opts.scale_factor)(position, _bead_radius(bead_type) * opts.scale_factor, bead_name)
			width, above, below = extents[i]
			fixed_centers[i] = (x + width / 2, y - (above - below) / 2)
		centers = _optimize_label_positions(
			bead_positions=[position for position, _, _ in labeled],
			bead_radii=[_bead_radius(bead_type) * opts.scale_factor for _, bead_type, _ in labeled],
			label_sizes=[(width, above + below) for width, above, below in extents],
			circles=circles,
			segments=segments,
			canvas_size=opts.canvas_size,
			gap=2 * opts.scale_factor,
			fixed_centers=fixed_centers)
		# convert label centers to text anchors (left end of the baseline of the last line)
		anchors = [(center[0] - width / 2, center[1] + (above - below) / 2) for center, (width, above, below) in zip(centers, extents)]
		return [lambda pos, r, bead_name, anchor=anchor: anchor for anchor in anchors]

	def _itp_cg_interactions(self, interaction_type, hidden=()):
		'''
		Yield the cg_graph nodes of each interaction of the given type in the loaded ITP,
		skipping interactions that involve hidden nodes. Interactions between the same beads
		(in the same or reversed order, e.g. multiple dihedral terms) are only yielded once.
		'''
		if self.ff is None:
			raise RuntimeError("No ITP loaded, call read_itp() first.")
		itp_to_cg = self._itp_to_cg_nodes()
		seen = set()
		for interaction in self.ff.blocks[self.molname].interactions.get(interaction_type, []):
			nodes = [itp_to_cg[atom] for atom in interaction.atoms]
			key = min(tuple(nodes), tuple(reversed(nodes)))
			if key in seen or any(node in hidden for node in nodes):
				continue
			seen.add(key)
			yield nodes

	def _draw_cg_bonds(self, bead_positions, opts, hidden=()):
		'''
		Create SVG lines for the bonds and constraints of the loaded ITP, colored by opts.bond_color and opts.constraint_color.

		Parameters
		----------
		bead_positions : dict
			Drawing coordinates of the beads, keyed by cg_graph node.
		opts : DrawOptions
		hidden : set
			cg_graph nodes that are not drawn; interactions involving them are skipped.

		Returns
		-------
		list of str
			The SVG line elements.
		'''
		lines = []
		for interaction_type, color in (('bonds', opts.bond_color), ('constraints', opts.constraint_color)):
			for i, j in self._itp_cg_interactions(interaction_type, hidden):
				lines.append(_create_bond_line(bead_positions[i], bead_positions[j],
					color=color, width=opts.cg_line_width * opts.scale_factor))
		return lines

	def _draw_cg_angles_dihedrals(self, bead_positions, opts, hidden=()):
		'''
		Create SVG polylines for the angles and dihedrals (including impropers) of the loaded ITP, depending on
		opts.show_angles and opts.show_dihedrals. Each one is drawn as a line along its bonds, shifted by
		opts.cg_line_offset away from the centroid of all beads. Where lines would overlap
		on a bond, the offset of the later line is increased in steps of opts.line_spacing; angles are placed first.

		Parameters
		----------
		bead_positions : dict
			Drawing coordinates of the beads, keyed by cg_graph node.
		opts : DrawOptions
		hidden : set
			cg_graph nodes that are not drawn; interactions involving them are skipped.

		Returns
		-------
		tuple of list of str
			The SVG polyline elements for the angles and for the dihedrals.
		'''
		centroid = np.mean(list(bead_positions.values()), axis=0)
		spacing = opts.line_spacing * opts.scale_factor
		occupied = {} # (bond, side) -> distances of the lines already placed along that bond on that side
		def place(nodes, offset, color):
			points = [bead_positions[n] for n in nodes]
			side = _offset_side(points, offset, centroid)
			# side relative to each bond in a fixed direction (lower node first), so lines on different paths can be compared
			lanes = [((a, b), side) if a < b else ((b, a), -side) for a, b in zip(nodes[:-1], nodes[1:])]
			taken = [d for lane in lanes for d in occupied.get(lane, [])]
			distance = abs(offset)
			while any(abs(distance - d) < spacing - 1e-9 for d in taken):
				distance += spacing
			for lane in lanes:
				occupied.setdefault(lane, []).append(distance)
			return _create_polyline(_offset_path(points, distance, side), color=color, width=opts.cg_line_width * opts.scale_factor)

		angles, dihedrals = [], []
		if opts.show_angles:
			angles = [place(nodes, opts.cg_line_offset * opts.scale_factor, opts.angle_color)
					  for nodes in self._itp_cg_interactions('angles', hidden)]
		if opts.show_dihedrals:
			dihedrals = [place(nodes, opts.cg_line_offset * opts.scale_factor, opts.dihedral_color)
						 for nodes in self._itp_cg_interactions('dihedrals', hidden)]
		return angles, dihedrals

	def to_svg(self, **kwargs) -> str:
		opts = replace(kwargs.pop("opts", None) or DrawOptions(), **kwargs)

		# draw molecule
		mol, idx_map = self._build_mol(opts)
		drawer = self._make_drawer(opts)
		o = drawer.drawOptions()
		o.addAtomIndices = opts.show_atom_indices
		o.bondLineWidth = opts.bond_width * opts.scale_factor
		if not opts.color_atoms:
			o.useBWAtomPalette()
		if opts.check_orientation:
			angle, vertical_bonds = _most_vertical_bonds(mol, tolerance=0.02)
			if abs(angle) < 0.5:
				print(f"Found vertical lines with angle {angle:+.2f} degrees")
			else:
				print(f"No vertical lines found, min angle: {angle:+.2f} degrees")
			o.continuousHighlight = False # draw the highlighted bonds in red instead of underlaying them
			drawer.DrawMolecule(mol, highlightAtoms=[], highlightBonds=vertical_bonds,
								highlightBondColors={bond: (1.0, 0.0, 0.0) for bond in vertical_bonds})
		else:
			drawer.DrawMolecule(mol)
		drawer.FinishDrawing()
		svg = drawer.GetDrawingText()
		if opts.show_mapping:
			svg = self._draw_beads(mol, svg, idx_map, opts)
		if opts.remove_background:
			svg = _strip_background(svg)
		return svg

	def save(self, path, **kwargs):
		path = path if path.endswith(".svg") else path + ".svg"
		with open(path, "w") as f:
			f.write(self.to_svg(**kwargs))

	def show(self, ax=None, **kwargs):
		svg = self.to_svg(**kwargs)
		png = cairosvg.svg2png(bytestring=svg.encode(), dpi=300)
		if ax is None:
			_, ax = plt.subplots()
		ax.imshow(Image.open(io.BytesIO(png)))
		ax.axis("off")
		return ax

	def show_default(self, ax=None, **kwargs):
		opts = kwargs.pop("opts", None) or DrawOptions(**kwargs)
		if ax is None:
			_, ax = plt.subplots(1, 1)
		if not opts.show_hydrogens:
			aa_graph = self.aa_graph.copy() # remove_explicit_hydrogens works in place
			pysmiles.remove_explicit_hydrogens(aa_graph)
			labels = {}
			for node in aa_graph.nodes:
				hcount =  aa_graph.nodes[node].get('hcount', 0)
				label = aa_graph.nodes[node].get('element', '*')
				if hcount > 0:
					label = label + f"H{hcount}"
				labels[node] = label
			draw_molecule(aa_graph, ax=ax, labels=labels, scale=1, align_with='x')
		else:
			draw_molecule(self.aa_graph, ax=ax, scale=opts.scale_factor)
		return ax

	# Structure generation methods
	@staticmethod
	def _prepare_output(out_file, resname):
		"""
		Add a .pdb extension if none is given, check that MDAnalysis can write the format
		and derive the residue name from the file name if none is given.
		"""
		from MDAnalysis.coordinates.core import get_writer_for
		path = Path(out_file)
		if not path.suffix:
			path = path.with_suffix('.pdb')
		try:
			get_writer_for(str(path), multiframe=False)
		except TypeError:
			raise ValueError(f"Unsupported file type: {path.suffix[1:]}. Use a single frame format "
							 "MDAnalysis can write, e.g. pdb or gro.") from None
		resname = path.stem[:5] if resname is None else resname[:5]
		return path, resname

	def _embed_aa(self, random_seed=42):
		"""
		Generate 3D coordinates for the all-atom molecule and optimize the geometry.
		Returns the RDKit molecule and the mapping from aa_graph nodes to RDKit atom indices.
		random_seed makes the conformer reproducible; None gives a different one on every call.
		"""
		mol, idx_map = cgsmiles_to_rdkit(self.aa_graph, add_hydrogens=True)
		SanitizeMol(mol)
		params = AllChem.ETKDG()
		params.randomSeed = -1 if random_seed is None else random_seed
		if AllChem.EmbedMolecule(mol, params) == -1:
			params.useRandomCoords = True # more robust for large or strained molecules
			if AllChem.EmbedMolecule(mol, params) == -1:
				raise RuntimeError("RDKit could not generate 3D coordinates for the molecule.")
		if AllChem.UFFOptimizeMolecule(mol, maxIters=2000) != 0:
			warnings.warn("The UFF geometry optimization of the all-atom structure did not converge.")
		return mol, idx_map

	@staticmethod
	def _write_universe(u, path):
		u.dimensions = [BOX_SIZE, BOX_SIZE, BOX_SIZE, 90, 90, 90]
		with warnings.catch_warnings():
			warnings.filterwarnings("ignore", category=UserWarning, module='MDAnalysis') # e.g. missing elements or occupancies
			u.atoms.write(str(path))

	@classmethod
	def _write_aa(cls, mol, path, resname):
		with warnings.catch_warnings():
			warnings.filterwarnings("ignore", category=UserWarning, module='MDAnalysis')
			u = mda.Universe(mol)
		u.add_TopologyAttr('resnames', [resname])
		cls._write_universe(u, path)

	def aa_structure(self, out_file="molecule_aa.pdb", resname=None, random_seed=42):
		"""
		Generate and save an all-atom structure file from a CGSmiles string.
		random_seed makes the conformer reproducible; None gives a different one on every call.
		"""
		path, resname = self._prepare_output(out_file, resname)
		mol, _ = self._embed_aa(random_seed)
		self._write_aa(mol, path, resname)

	def cg_structure(self, out_file="molecule_cg.pdb", resname=None, write_aa=True, random_seed=42):
		"""
		Generate and save a coarse-grained structure file from a CGSmiles string.
		Bead positions are the centers of the atoms they map to; virtual sites are placed
		at the weighted center of the atoms of the beads they are connected to (weights from wVS, else w). The matching all-atom structure
		is written next to it (``_cg`` in the file name replaced by ``_aa``, or ``_aa`` appended).
		random_seed makes the conformer reproducible; None gives a different one on every call.
		"""
		path, resname = self._prepare_output(out_file, resname)
		# get the suffix of the output file to determine the format (e.g., .pdb, .gro)
		suffix = path.suffix[1:]  # remove the dot
		aa_stem = path.stem.replace('_cg', '_aa') if '_cg' in path.stem else f'{path.stem}_aa'
		mol, idx_map = self._embed_aa(random_seed)
		if write_aa:
			self._write_aa(mol, path.with_name(f'{aa_stem}.{suffix}'), resname)
		aa_positions = mol.GetConformer().GetPositions() # in Angstrom

		virtual_nodes = detect_virtual_nodes(self.cg_graph)
		positions = {}
		for node, data in self.cg_graph.nodes(data=True):
			if node in virtual_nodes:
				continue
			fragment = data['graph']
			if len(fragment) == 0:
				raise ValueError(f"Bead {data.get('n', node)} has no atoms and is not a virtual site.")
			positions[node] = np.mean([aa_positions[idx_map[atom]] for atom in fragment], axis=0)
		for node in virtual_nodes: # weighted by the wVS annotations of the atoms of the parent beads
			atom_weights = _vs_atom_weights(self.cg_graph, node, virtual_nodes)
			positions[node] = _get_vs_coordinate(atom_weights, lambda atom: aa_positions[idx_map[atom]],
												 self.aa_graph, node)

		# order beads by their 'index' property, then by the number in their name, fall back to the node id
		def bead_index(node):
			data = self.cg_graph.nodes[node]
			if 'index' in data:
				return int(data['index'])
			match = re.search(r'\d+$', data.get('n', ''))
			return int(match.group()) if match else node
		nodes = sorted(self.cg_graph.nodes, key=bead_index)

		u = mda.Universe.empty(len(nodes), n_residues=1, atom_resindex=np.zeros(len(nodes), dtype=int), trajectory=True)
		u.add_TopologyAttr('names', [self.cg_graph.nodes[n]['atomname'] for n in nodes])
		u.add_TopologyAttr('resnames', [resname])
		u.add_TopologyAttr('resids', [1])
		u.add_TopologyAttr('record_types', ['HETATM'] * len(nodes))
		u.atoms.positions = np.array([positions[n] for n in nodes])
		self._write_universe(u, path)

	# ITP manipulation methods
	def change_bead_types(self, bead_change):
		"""
		Change the types of beads, selected by their name (n= annotation).

		Each changed bead gets its own copy of its fragment, named after the new type. If that
		name is already taken, an uppercase letter is appended (e.g. TC5Z), which is removed
		again when the type is read. All other annotations of the beads are kept.

		Parameters
		----------
		bead_change : dict
			{bead name: new bead type}

		Returns
		-------
		str
			The new CGSmiles string, also stored in self.cgs_string.
		"""
		resolutions = re.split(r'\}\s*\.\s*\{', self.cgs_string.strip())
		if len(resolutions) != 2 or not resolutions[0].startswith('{') or not resolutions[1].endswith('}'):
			raise ValueError("change_bead_types only supports CGSmiles strings with two resolutions, {cg}.{fragments}.")
		cg, fragments = resolutions[0][1:], resolutions[1][:-1]
		definitions = [part.strip() for part in _split_top_level(fragments)]
		fragment_bodies = {} # fragment name -> definition without '#name'
		for definition in definitions:
			name, _, _ = definition.lstrip('#').partition('=')
			fragment_bodies[name] = definition[len(name) + 1:]

		beads = {} # bead name -> match of the bead in the cg part
		for match in _BEAD_PATTERN.finditer(cg):
			name = re.search(r';n=([^;\]]+)', match.group(2))
			if name is None:
				continue
			if name.group(1) in beads:
				raise ValueError(f"Bead name {name.group(1)} is not unique in the CGSmiles string.")
			beads[name.group(1)] = match
		unknown = [name for name in bead_change if name not in beads]
		if unknown:
			raise ValueError(f"No bead named {', '.join(unknown)} in the CGSmiles string. Named beads: {', '.join(beads)}.")

		used = set(fragment_bodies) | {match.group(1) for match in _BEAD_PATTERN.finditer(cg)}
		def unique_fragname(bead_type):
			if bead_type not in used and _fragname_to_type(bead_type) == bead_type:
				return bead_type
			for letter in reversed(string.ascii_uppercase):
				if bead_type + letter not in used:
					return bead_type + letter
			raise ValueError(f"No unique fragment name left for bead type {bead_type}.")

		copies = {} # (old fragment, new type) -> new fragment, so beads sharing a fragment share the copy
		new_definitions = []
		replacements = [] # (start, end, new fragment) of the fragment names in the cg part
		for name, bead_type in bead_change.items():
			match = beads[name]
			fragname = match.group(1)
			if _fragname_to_type(fragname) == bead_type:
				continue
			if (fragname, bead_type) not in copies:
				new_fragname = unique_fragname(bead_type)
				used.add(new_fragname)
				copies[(fragname, bead_type)] = new_fragname
				if fragname in fragment_bodies: # fragments without atoms (e.g. virtual sites) have no definition
					new_definitions.append(f'#{new_fragname}{fragment_bodies[fragname]}')
			replacements.append((match.start(1), match.end(1), copies[(fragname, bead_type)]))

		for start, end, new_fragname in sorted(replacements, reverse=True):
			cg = cg[:start] + new_fragname + cg[end:]
		self.cgs_string = f'{{{cg}}}.{{{",".join(new_definitions + definitions)}}}'

		self.cg_graph, self.aa_graph = (
			cgsmiles.MoleculeResolver.from_string(self.cgs_string).resolve()
		)
		self._clean_cg_graph()
		return self.cgs_string

	def read_itp(self, path_in, molname=None):
		with open(f'{path_in}') as f:
			lines = f.readlines()
	
		self.ff = ForceField("dummy")
		read_itp(lines, self.ff)
		if molname is not None:
			if molname not in self.ff.blocks:
				raise ValueError(f"Molname {molname} not found in the ITP file.")
			self.molname = molname
		else:
			self.molname = list(self.ff.blocks.keys())[0]
		# types and masses as read, so changes are always reported relative to the ITP file
		self._itp_original = {node: (d['atype'], d.get('mass'))
							  for node, d in self.ff.blocks[self.molname].nodes(data=True)}

	def _itp_to_cg_nodes(self):
		'''
		Map the nodes of the loaded ITP block to cg_graph nodes via their atom names.
		'''
		cg_by_name = {}
		for n, attrs in self.cg_graph.nodes(data=True):
			if attrs["atomname"] in cg_by_name:
				raise ValueError(f"Bead name {attrs['atomname']} is not unique in the CGSmiles string, "
								 "so the beads cannot be matched to the ITP by name.")
			cg_by_name[attrs["atomname"]] = n
		mapping = {}
		for node, d in self.ff.blocks[self.molname].nodes(data=True):
			if d["atomname"] not in cg_by_name:
				raise ValueError("No matching node found in cg_graph for node {} in ff.blocks".format(node))
			mapping[node] = cg_by_name[d["atomname"]]
		return mapping

	def _itp_virtual_sites(self):
		'''
		Return the cg_graph nodes that are defined as virtual sites in the loaded ITP.
		'''
		itp_to_cg = self._itp_to_cg_nodes()
		block = self.ff.blocks[self.molname]
		return {itp_to_cg[interaction.atoms[0]] # the virtual site is the first atom of each entry
				for name, interactions in block.interactions.items() if name.startswith('virtual_sites')
				for interaction in interactions}

	def _apply_cg_types(self, block):
		'''
		Set the atom types of block to the bead types of cg_graph. If the bead size changes and the
		atom has the default Martini mass of its original size, the mass is changed to the default
		of the new size. Changes are relative to the ITP as read.

		Returns
		-------
		dict
			{block node: comment describing the change}
		'''
		comments = {}
		itp_to_cg = self._itp_to_cg_nodes()
		for node, d in block.nodes(data=True):
			old_type, old_mass = self._itp_original[node]
			new_type = self.cg_graph.nodes[itp_to_cg[node]]["atype"]
			d['atype'] = new_type
			d['mass'] = old_mass
			if new_type == old_type:
				continue
			changes = [f"changed from {old_type} to {new_type}"]
			old_size, new_size = _bead_size(old_type), _bead_size(new_type)
			if (old_size and new_size and old_size != new_size and old_mass is not None
					and np.isclose(old_mass, MARTINI_MASSES[old_size])):
				d['mass'] = MARTINI_MASSES[new_size]
				changes.append(f"mass {old_mass:g} -> {d['mass']:g}")
			comments[node] = ', '.join(changes)
		return comments

	def change_itp(self):
		'''
		Update the loaded ITP to the bead types of the CGSmiles string (see _apply_cg_types).

		Returns
		-------
		dict
			{ITP block node: comment describing the change}
		'''
		if self.ff is None:
			raise RuntimeError("No ITP loaded, call read_itp() first.")
		return self._apply_cg_types(self.ff.blocks[self.molname])

	def write_itp(self, path_out, outname=None):
		'''
		Write the loaded ITP with the bead types of the CGSmiles string to <path_out>/<outname>.itp.
		Changed atoms are commented. The loaded ITP itself is not modified.
		'''
		if self.ff is None:
			raise RuntimeError("No ITP loaded, call read_itp() first.")
		block = copy.deepcopy(self.ff.blocks[self.molname])
		comments = self._apply_cg_types(block)
		outname = outname if outname else self.molname
		path = itp_writer(outname, block, self.cgs_string, path_out)
		# the atoms are numbered in the ITP in the order of the block, starting at 1
		itp_index = {node: i for i, node in enumerate(block.nodes, start=1)}
		add_comments_itp({itp_index[node]: comment for node, comment in comments.items()}, path)

	### PKa prediction methods
	def predict_pka(self, show_protonation_reactions=False, show_protonation_map=True, print_pka_values=False):
		"""
		Predict the pKa values of the molecule with pkasolver.

		Parameters
		----------
		show_protonation_reactions : bool
			Display the network of protonation reactions.
		show_protonation_map : bool
			Display the pKa map.
		print_pka_values : bool
			Print the pKa values.

		Returns
		-------
		list
			The protonation states with their predicted pKa values.
		"""
		mol, _ =cgsmiles_to_rdkit(self.aa_graph, add_hydrogens=False)
		SanitizeMol(mol)
		rdDepictor.Compute2DCoords(mol)

		# silence pkasolver (warnings, logging and output of the underlying libraries) only during the prediction
		previous_logging_level = logging.root.manager.disable
		try:
			with warnings.catch_warnings(), suppress_output():
				warnings.simplefilter("ignore", category=FutureWarning)
				logging.disable(logging.CRITICAL)
				pka_results = calculate_microstate_pka_values(mol)
		finally:
			logging.disable(previous_logging_level)

		previous_size = IPythonConsole.molSize
		IPythonConsole.molSize = 400, 400
		try:
			if show_protonation_reactions:
				print('Drawing pKa reaction network...')
				display(draw_pka_reactions(pka_results))
			if show_protonation_map:
				print('Drawing pKa map...')
				display(draw_pka_map(pka_results))
		finally:
			IPythonConsole.molSize = previous_size

		if print_pka_values:
			for state in pka_results:
				print(f'pKa = {state.pka}')
		return pka_results