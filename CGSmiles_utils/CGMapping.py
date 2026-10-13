import cgsmiles
from cgsmiles.drawing import draw_molecule
from fast_forward.itp_parser_sub import read_itp
import pysmiles

import warnings, io, re, string
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
from PIL import Image
import cairosvg

from .utils import cgsmiles_to_rdkit
from .drawing_utils import _flip_molecule, _rotate_molecule_around_center, _check_vertical_lines, _get_bead_coordinate, detect_virtual_nodes, _create_bead_circle, _create_bead_label, _create_bond_line, _strip_background
from .itp_utils import itp_writer, add_comments_itp
warnings.filterwarnings("ignore", category=UserWarning, module='MDAnalysis')  # ignore warnings from MDAnalysis about missing box dimensions

REFERENCE_SCALE = 25

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
	constraint_color: str = "#0400FF"
	cg_bond_width: float = 2.0
	include_hydrogens_in_bead_position: bool | None = None # None follows show_hydrogens
	bead_type_label_color : str = '#666666'
	bead_name_label_color : str = '#000000'
	bead_label_position: Callable = lambda pos, r, bead_name: (pos[0] + r * 1 / np.sqrt(2), pos[1] - r * 1 / np.sqrt(2)) # function to determine the position of the bead label relative to the bead position
	# debugging
	check_orientation: bool = False

	def __post_init__(self):
		if self.show_bead_labels and not self.show_mapping:
			warnings.warn("show_bead_labels has no effect when show_mapping=False")
		if self.show_bonds and not self.show_mapping:
			warnings.warn("show_bonds has no effect when show_mapping=False")

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
			data["atomname"] = data["n"]

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
		labels = [] 
		itp_vs = self._itp_virtual_sites() if opts.indicate_vs and self.ff is not None else set()
		def add_bead(bead, position, bead_type, bead_name):
			beads.append(_create_bead_circle(position, bead_type, bead_name, opts.scale_factor, dashed=bead in itp_vs))
			if opts.show_bead_labels:
				labels.append(_create_bead_label(
					position, bead_type, bead_name=bead_name,
					label_position=opts.bead_label_position,
					scale_factor=opts.scale_factor,
					type_color=opts.bead_type_label_color,
					name_color=opts.bead_name_label_color))

		for bead, data in self.cg_graph.nodes(data=True):
			if bead in virtual_edges:   # virtual nodes are placed afterwards
				continue
			bead_type = data['fragname']
			pos = _get_bead_coordinate(data['graph'], idx_map, drawer_coords, opts, bead)
			bead_positions[bead] = pos
			add_bead(bead, pos, bead_type, data.get('n'))

		for vnode, connections in virtual_edges.items():
			data = self.cg_graph.nodes[vnode]
			pos = np.mean([bead_positions[b] for b in connections], axis=0)
			bead_positions[vnode] = pos
			if opts.show_vs:
				add_bead(vnode, pos, data['fragname'], data.get('n'))

		hidden = set() if opts.show_vs else set(virtual_edges)
		bonds = self._draw_cg_bonds(bead_positions, opts, hidden) if opts.show_bonds else []
		return svg.replace('</svg>', ''.join(bonds) + ''.join(beads) + ''.join(labels) + '</svg>') # bonds below beads

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
		if self.ff is None:
			raise RuntimeError("No ITP loaded, call read_itp() first.")
		block = self.ff.blocks[self.molname]
		itp_to_cg = self._itp_to_cg_nodes()
		lines = []
		for interaction_type, color in (('bonds', opts.bond_color), ('constraints', opts.constraint_color)):
			for interaction in block.interactions.get(interaction_type, []):
				i, j = (itp_to_cg[atom] for atom in interaction.atoms)
				if i in hidden or j in hidden:
					continue
				lines.append(_create_bond_line(bead_positions[i], bead_positions[j],
					color=color, width=opts.cg_bond_width * opts.scale_factor))
		return lines

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
		at the center of the beads they are connected to. The matching all-atom structure
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
		for node, connections in virtual_nodes.items():
			positions[node] = np.mean([positions[c] for c in connections], axis=0)

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