import cgsmiles
from cgsmiles.drawing import draw_molecule
from fast_forward.itp_parser_sub import read_itp
import pysmiles

import warnings, io, re, string, os, logging
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
from .drawing_utils import _bead_radius, _flip_molecule, _rotate_molecule_around_center, _check_vertical_lines, _get_bead_coordinate, _vs_atom_weights, _get_vs_coordinate, detect_virtual_nodes, _create_bead_circle, _create_bead_label, _bead_label_extent, _label_above_bead, _optimize_label_positions, _create_bond_line, _create_polyline, _offset_path, _offset_side, _strip_background
from .itp_utils import itp_writer, add_comments_itp
warnings.filterwarnings("ignore", category=UserWarning, module='MDAnalysis')  # ignore warnings from MDAnalysis about missing box dimensions

REFERENCE_SCALE = 25

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
	# molecule
	show_hydrogens: bool = False
	color_atoms: bool = False
	show_atom_indices: bool = False
	bond_width: float = 1.5
	# geometry
	canvas_size: tuple = (300, 300)
	scale_factor: float = 1
	rotate_by: float = 0
	flip: bool = False
	remove_background: bool = True
	# mapping
	show_mapping: bool = True
	show_bead_labels: bool = False
	show_vs: bool = True
	indicate_vs: bool = True # dashed outline for beads that are virtual sites in the ITP loaded with read_itp()
	show_node_indicators: bool = False
	show_bonds: bool = False # draw bonds and constraints from the ITP loaded with read_itp()
	bond_color: str = '#000000'
	constraint_color: str = "#020070FF"
	show_angles: bool = False # draw angles from the ITP as bent lines offset from the beads
	angle_color: str = "#286E8D"
	angle_offset: float = 8.0 # distance between an angle line and the bonds it follows, positive = away from the centroid of all beads
	show_dihedrals: bool = False # draw dihedrals from the ITP as lines through the four beads
	dihedral_color: str = '#286E8D'
	dihedral_offset: float = 8.0 # distance between a dihedral line and the bonds it follows, positive = away from the centroid of all beads
	cg_line_width: float = 2.0 # line width for bonds, constraints, angles and dihedrals
	line_spacing: float = 4.0 # distance between angle/dihedral lines that would otherwise overlap
	include_hydrogens_in_bead_position: bool | None = None # None follows show_hydrogens
	bead_type_label_color : str = '#666666'
	bead_name_label_color : str = '#000000'
	bead_label_position: Callable = lambda pos, r, bead_name: (pos[0] + r * 1 / np.sqrt(2), pos[1] - r * 1 / np.sqrt(2)) # function to determine the position of the bead label relative to the bead position
	optimize_bead_labels: bool = False # place bead labels around their beads to minimize overlap with other labels, beads, the molecule and the drawn bonds/angles/dihedrals; overrides bead_label_position
	# labels of dummy beads (type U) are always placed right above the bead
	# debugging
	check_orientation: bool = False

	def __post_init__(self):
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
			if data["fragname"][-1].isupper() and len(data["fragname"]) > 1: # remove trailing letters from fragname except for virtual sites
				data["fragname"] = data["fragname"][:-1]
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
		drawer.SetScale(opts.scale_factor*REFERENCE_SCALE, opts.scale_factor*REFERENCE_SCALE,
						Point2D(1000, 1000), Point2D(-1000, -1000))
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
			beads.append(_create_bead_circle(position, bead_type, bead_name, opts.scale_factor, dashed=bead in itp_vs))
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
			label_positions = [opts.bead_label_position] * len(labeled)
			for i in labeled_dummies:
				_, bead_type, bead_name = labeled[i]
				label_positions[i] = _label_above_bead(bead_type, bead_name, opts.scale_factor)
			if opts.optimize_bead_labels:
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
					('show_angles', ('angles',), opts.angle_offset * opts.scale_factor + cg_width),
					('show_dihedrals', ('dihedrals',), opts.dihedral_offset * opts.scale_factor + cg_width)):
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
		opts.angle_offset or opts.dihedral_offset away from the centroid of all beads. Where lines would overlap
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
			angles = [place(nodes, opts.angle_offset * opts.scale_factor, opts.angle_color)
					  for nodes in self._itp_cg_interactions('angles', hidden)]
		if opts.show_dihedrals:
			dihedrals = [place(nodes, opts.dihedral_offset * opts.scale_factor, opts.dihedral_color)
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
		drawer.DrawMolecule(mol)
		drawer.FinishDrawing()
		svg = drawer.GetDrawingText()
		if opts.show_mapping:
			svg = self._draw_beads(mol, svg, idx_map, opts)
		if opts.check_orientation:
			svg = _check_vertical_lines(svg, highlight=True, threshold=0.5, tolerance=0.02)
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
		path = Path(out_file)
		if not path.suffix:
			path = path.with_suffix('.pdb')
		file_type = path.suffix[1:]
		supported = list(mda._SINGLEFRAME_WRITERS.keys())
		if file_type.casefold() not in [fmt.casefold() for fmt in supported]:
			raise ValueError(f"Unsupported file type: {file_type}. Supported types are: {supported}")
		resname = path.stem[:5] if resname is None else resname[:5]
		return path, resname

	def _embed_aa(self):
		"""
		Generate 3D coordinates for the all-atom molecule and optimize the geometry.
		Returns the RDKit molecule and the mapping from aa_graph nodes to RDKit atom indices.
		"""
		mol, idx_map = cgsmiles_to_rdkit(self.aa_graph, add_hydrogens=True)
		SanitizeMol(mol)
		AllChem.EmbedMolecule(mol, AllChem.ETKDG())
		AllChem.UFFOptimizeMolecule(mol)
		return mol, idx_map

	@staticmethod
	def _write_aa(mol, path, resname):
		u = mda.Universe(mol)
		u.add_TopologyAttr('resnames', [resname])
		u.dimensions = [100, 100, 100, 90, 90, 90]  # set box dimensions to avoid warnings
		u.atoms.write(str(path))

	def aa_structure(self, out_file="molecule_aa.pdb", resname=None):
		"""
		Generate and save an all-atom structure file from a CGSmiles string.
		"""
		path, resname = self._prepare_output(out_file, resname)
		mol, _ = self._embed_aa()
		self._write_aa(mol, path, resname)

	def cg_structure(self, out_file="molecule_cg.pdb", resname=None, write_aa=True):
		"""
		Generate and save a coarse-grained structure file from a CGSmiles string.
		Bead positions are the centers of the atoms they map to; virtual sites are placed
		at the weighted center of the atoms of the beads they are connected to (weights from wVS, else w). The matching all-atom structure
		is written next to it (``_cg`` in the file name replaced by ``_aa``, or ``_aa`` appended).
		"""
		path, resname = self._prepare_output(out_file, resname)
		# get the suffix of the output file to determine the format (e.g., .pdb, .gro)
		suffix = path.suffix[1:]  # remove the dot
		aa_stem = path.stem.replace('_cg', '_aa') if '_cg' in path.stem else f'{path.stem}_aa'
		mol, idx_map = self._embed_aa()
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
		u.add_TopologyAttr('names', [self.cg_graph.nodes[n]['n'] for n in nodes])
		u.add_TopologyAttr('resnames', [resname])
		u.add_TopologyAttr('resids', [0])
		u.add_TopologyAttr('record_types', ['HETATM'] * len(nodes))
		u.atoms.positions = np.array([positions[n] for n in nodes])
		u.dimensions = np.array([10.0, 10.0, 10.0, 90.0, 90.0, 90.0]) # set box size to avoid warnings
		u.atoms.write(str(path))

	# ITP manipulation methods
	def change_bead_types(self, bead_change):
		"""
		bead_change is a dictionary with the format {name: new_type}
		"""
		cg = self.cgs_string.split('}.{')[0][1:]
		aa = self.cgs_string.split('}.{')[1][:-1]

		# split the cg part into its components
		cg = [part.split('[') for part in cg.split(']')]
		cg = [item for sublist in cg for item in sublist if item]
		indices = [i for i, part in enumerate(cg) if part.startswith('#')] # get the indices of the beads in the cg part
		types = []
		names = []
		for i in indices:
			tokens = cg[i].split(';')
			for token in tokens:
				if token.startswith('#'):
					types.append(token[1:])
				if token.startswith('n='):
					names.append(token[2:])
		types_initial = types.copy()
		# check if names are unique
		if len(names) != len(set(names)):
			raise ValueError("Names in the CGSmiles string are not unique")

		non_unique_count = 0
		letters = list(reversed(string.ascii_uppercase))
		# change types according to the bead_change
		new_aa_parts = []
		for name, type_ in bead_change.items():
			idx = names.index(name)
			if type_ != types[idx]: # change type
				new_type = f'{type_}{letters[non_unique_count]}' if type_ in types_initial else type_ # add letter if the type is already present in the cg part
				for part in aa.split(','):
					if part.startswith(f'#{types[idx]}'):
						new_aa_parts.append(part.replace(f'#{types[idx]}', f'#{new_type}')) # copy the part with the new type 
				types[idx] = new_type
				non_unique_count += 1 if type_ in types else 0

		# assembling the aa string again
		aa = ','.join(new_aa_parts) + ',' + aa if new_aa_parts else aa

		# assembling the cg string again
		for i, idx in enumerate(indices):
			cg[idx] = f'[#{types[i]};n={names[i]}]'
		cg_string = ''.join(cg)
		self.cgs_string = f'{{{cg_string}}}.{{{aa}}}'

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

	def _itp_to_cg_nodes(self):
		'''
		Map the nodes of the loaded ITP block to cg_graph nodes via their atom names.
		'''
		cg_by_name = {attrs["atomname"]: n for n, attrs in self.cg_graph.nodes(data=True)}
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

	def change_itp(self):
		if self.ff is None:
			raise RuntimeError("No ITP loaded, call read_itp() first.")
		comments = {}
		itp_to_cg = self._itp_to_cg_nodes()
		for node, d in self.ff.blocks[self.molname].nodes(data=True):
			cg_node = itp_to_cg[node]
			if d['atype'] != self.cg_graph.nodes[cg_node]["atype"]:
				comments[node] = f"changed from {d['atype']} to {self.cg_graph.nodes[cg_node]['atype']}"
				d['atype'] = self.cg_graph.nodes[cg_node]["atype"]
		return comments

	def write_itp(self, path_out, outname=None):
		comments = self.change_itp()
		outname = outname if outname else self.molname
	
		itp_writer(outname, self.ff.blocks[self.molname], self.cgs_string, path_out)	
		add_comments_itp(comments, f'{path_out}/{outname}.itp')

	### PKa prediction methods
	def predict_pka(self, show_protontation_reactions=False, show_protonation_map=True, print_pka_values=False):
		"""
		Predict the pKa values of the molecule.

		Returns
		-------
		dict
			A dictionary with predicted pKa values and associated information.
		"""
		mol, _ = self._build_mol(opts=DrawOptions())

		warnings.filterwarnings("ignore", category=FutureWarning)
		logging.disable(logging.CRITICAL)

		with suppress_output():
			pka_results = calculate_microstate_pka_values(mol)
		
		IPythonConsole.molSize = 400,400
		if show_protontation_reactions:
			print('Drawing pKa reaction network...')
			display(draw_pka_reactions(pka_results))

		if show_protonation_map:
			print('Drawing pKa map...')
			display(draw_pka_map(pka_results))

		if print_pka_values:
			for state in pka_results:
				print(f'pKa = {state.pka}')
		return pka_results