import cgsmiles
from cgsmiles.drawing import draw_molecule
import pysmiles

import warnings, io, re, os
from collections.abc import Callable
from dataclasses import dataclass, replace
import numpy as np
import matplotlib.pyplot as plt
import MDAnalysis as mda
import networkx as nx
import vermouth
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Geometry.rdGeometry import Point2D
from rdkit.Chem import rdDepictor, MolToSmiles, SanitizeMol, AllChem, MolToPDBBlock
from PIL import Image
import cairosvg

from .utils import cgsmiles_to_rdkit, write_file
from .drawing_utils import _flip_molecule, _rotate_molecule_around_center, _check_vertical_lines, _get_bead_coordinate, detect_virtual_nodes, _create_bead_circle, _create_bead_label, _strip_background
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
	show_node_indicators: bool = False
	include_hydrogens_in_bead_position: bool | None = None
	bead_type_label_color : str = '#666666'
	bead_name_label_color : str = '#000000'
	bead_label_position: Callable = lambda pos, r, bead_name: (pos[0] + r * 1 / np.sqrt(2), pos[1] - r * 1 / np.sqrt(2)) # function to determine the position of the bead label relative to the bead position
	# debugging
	check_orientation: bool = False
	

	def __post_init__(self):
		if self.include_hydrogens_in_bead_position is None:
			self.include_hydrogens_in_bead_position = self.show_hydrogens
		if self.show_bead_labels and not self.show_mapping:
			warnings.warn("show_bead_labels has no effect when show_mapping=False")


class CGMapping:
	def __init__(self, cgs_string: str):
		self.cgs_string = cgs_string
		self.cg_graph, self.aa_graph = (
			cgsmiles.MoleculeResolver.from_string(cgs_string).resolve()
		)
		self._clean_cg_graph()

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
		def add_bead(position, bead_type, bead_name):
			beads.append(_create_bead_circle(position, bead_type, bead_name, opts.scale_factor))
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
			add_bead(pos, bead_type, data.get('n'))

		if opts.show_vs:
			for vnode, connections in virtual_edges.items():
				data = self.cg_graph.nodes[vnode]
				pos = np.mean([bead_positions[b] for b in connections], axis=0)
				add_bead(pos, data['fragname'], data.get('n'))

		return svg.replace('</svg>', ''.join(beads) + ''.join(labels) + '</svg>')

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
			pysmiles.remove_explicit_hydrogens(self.aa_graph)
			labels = {}
			for node in self.aa_graph.nodes:
				hcount =  self.aa_graph.nodes[node].get('hcount', 0)
				label = self.aa_graph.nodes[node].get('element', '*')
				if hcount > 0:
					label = label + f"H{hcount}"
				labels[node] = label
			draw_molecule(self.aa_graph, ax=ax, labels=labels, scale=1, align_with='x')
		else:
			draw_molecule(self.aa_graph, ax=ax, scale=opts.scale_factor)
		return ax

	def smiles(self, add_hydrogens=False):
		opts = DrawOptions(show_hydrogens=add_hydrogens)
		mol, _ = self._build_mol(opts)
		return MolToSmiles(mol, isomericSmiles=True)

	def generate_AA_structure(self, out_file="molecule_aa.pdb", resname=None):
		"""
		Generate an all-atom structure file from a CGSmiles string.

		Parameters
		----------
		out_file : str, optional
			The name of the output structure file (default is "molecule_aa.pdb").

		Returns
		-------
		None
			The function saves the structure file to the specified filename.
		"""

		mol, _ =  cgsmiles_to_rdkit(self.aa_graph, add_hydrogens=True)

		# generate 3D coordinates and optimize the geometry
		tokens = out_file.split('.')
		if len(tokens) == 2:
			name = tokens[0].split('/')[-1]
			file_type = tokens[1]
		if len(tokens) > 2:
			raise ValueError("Output file name should have only one dot (.) to separate the name and the file type.")
		if len(tokens) == 1:
			name = out_file.split('/')[-1]
			file_type = 'pdb'
			out_file = f"{out_file}.pdb"

		resname = name[:5] if resname is None else resname[:5]
		SanitizeMol(mol)
		AllChem.EmbedMolecule(mol, AllChem.ETKDG())
		AllChem.UFFOptimizeMolecule(mol)
		u = mda.Universe(mol)
		u.add_TopologyAttr('resnames', [resname])
		u.dimensions = [100, 100, 100, 90, 90, 90]  # set box dimensions to avoid warnings

		if file_type.lower() in [fmt.casefold() for fmt in mda._SINGLEFRAME_WRITERS.keys()]:
			u.atoms.write(f"{out_file}")
		else:
			raise ValueError(f"Unsupported file type: {file_type}. Supported types are: {list(mda._SINGLEFRAME_WRITERS.keys())}")

	def generate_CG_structure(self, out_file="molecule_cg.pdb", resname=None):

		if '_cg' in out_file:
			out_file_aa = out_file.replace('_cg', '_aa')
		else:
			out_file_aa = f'{out_file.split(".")[0]}_aa'
		out_file_aa = f'{out_file_aa.split(".")[0]}.pdb'
		self.generate_AA_structure(out_file_aa, resname)

		mol = vermouth.pdb.read_pdb(out_file_aa)[0]

		mapping = next(nx.isomorphism.GraphMatcher(mol, self.aa_graph).match())

		# check if index property is present in res_graph, if not add it
		for node, data in self.cg_graph.nodes(data=True):
			if 'index' not in data.keys():
				if 'n' in data.keys() and re.search(r'\d+$', data['n']):
					self.cg_graph.nodes[node]['index'] = int(re.search(r'\d+$', data['n']).group())  # extract number from n property
				else:
					self.cg_graph.nodes[node]['index'] = node  # fallback to using the node id if 'n' property is not present

		for node in self.cg_graph.nodes:
			pos = np.zeros((3))
			fragment = self.cg_graph.nodes[node]['graph']
			for all_atom_node in fragment:
				pos += mol.nodes[mapping[all_atom_node]]['position']
			pos = np.array(pos)
			final_pos = pos / len(fragment) if len(fragment) > 0 else pos
			self.cg_graph.nodes[node]['position'] = final_pos

		position_dict = {node_id: data['position'] for node_id, data in self.cg_graph.nodes(data=True)}
		for n, d in self.cg_graph.nodes(data=True):
			if d['fragname'] == 'U':
				self.cg_graph.nodes[n]['position'] = np.array([0.0, 0.0, 0.0])
			else:
				self.cg_graph.nodes[n]['position'] = position_dict[n]

		for n in self.cg_graph.nodes:
			d = self.cg_graph.nodes[n]
			if d['fragname'] == 'U':
				connected_nodes = [edge[1] if edge[0] == n else edge[0] for edge in self.cg_graph.edges(n)]
				if len(connected_nodes) == 0:
					raise ValueError(f'Warning: Unconnected U node at {n}, skipping')
				coords = [self.cg_graph.nodes[c]["position"] for c in connected_nodes if self.cg_graph.nodes[c]["fragname"] != 'U']
				d['position'] = np.mean(coords, axis=0)

		pdb_lines = []
		for n, d in sorted(self.cg_graph.nodes(data=True), key=lambda x: int(x[1]['index'])):
			coords = d["position"]*10
			line = f'HETATM{d["index"]:5d}  {d["n"][:3]:<3s} {resname[:3]:<3s}    1    {coords[0]:8.3f}{coords[1]:8.3f}{coords[2]:8.3f}  1.00  0.00            \n'
			pdb_lines.append(line)
		pdb_lines.append('END\n')

		write_file(f"temp.pdb", pdb_lines)
		u = mda.Universe("temp.pdb", in_memory=True)
		os.remove("temp.pdb")
		u.add_TopologyAttr('resnames', [resname])
		u.dimensions = np.array([10.0, 10.0, 10.0, 90.0, 90.0, 90.0]) # set box size to avoid warnings
		if  out_file.split('.')[-1].lower() in [fmt.casefold() for fmt in mda._SINGLEFRAME_WRITERS.keys()]:
			u.atoms.write(f"{out_file}")
		else:
			raise ValueError(f"Unsupported file type: {out_file.split('.')[-1]}. Supported types are: {list(mda._SINGLEFRAME_WRITERS.keys())}")
	
	