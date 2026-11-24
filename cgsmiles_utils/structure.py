'''
Generation of all-atom and coarse-grained 3D structures. MDAnalysis is only imported when a structure is written.
'''
import re
import warnings
from pathlib import Path

import numpy as np
from rdkit.Chem import AllChem, SanitizeMol

from .beads import bead_positions
from .rdkit_conv import cgsmiles_to_rdkit

BOX_SIZE = 100.0 # Angstrom, box written to the generated structure files


def _prepare_output(out_file, resname, default_resname=None):
	'''
	Add a .pdb extension if none is given, check that MDAnalysis can write the format
	and use default_resname, else the file name, as residue name if none is given.
	'''
	from MDAnalysis.coordinates.core import get_writer_for
	path = Path(out_file)
	if not path.suffix:
		path = path.with_suffix('.pdb')
	try:
		get_writer_for(str(path), multiframe=False)
	except TypeError:
		raise ValueError(f"Unsupported file type: {path.suffix[1:]}. Use a single frame format "
						 "MDAnalysis can write, e.g. pdb or gro.") from None
	if resname is None:
		resname = default_resname or path.stem
	return path, resname[:5]


def embed_aa(aa_graph, random_seed=42):
	'''
	Generate 3D coordinates for the all-atom molecule and optimize the geometry with UFF.
	random_seed makes the conformer reproducible; None gives a different one on every call.

	Returns
	-------
	mol : rdkit.Chem.rdchem.Mol
		The molecule with hydrogens and a 3D conformer (Angstrom).
	idx_map : dict
		{aa_graph node: RDKit atom index}
	'''
	mol, idx_map = cgsmiles_to_rdkit(aa_graph, add_hydrogens=True)
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


def _write_universe(u, path):
	u.dimensions = [BOX_SIZE, BOX_SIZE, BOX_SIZE, 90, 90, 90]
	with warnings.catch_warnings():
		warnings.filterwarnings("ignore", category=UserWarning, module='MDAnalysis') # e.g. missing elements or occupancies
		u.atoms.write(str(path))


def _write_aa(mol, path, resname):
	import MDAnalysis as mda
	with warnings.catch_warnings():
		warnings.filterwarnings("ignore", category=UserWarning, module='MDAnalysis')
		u = mda.Universe(mol)
	u.add_TopologyAttr('resnames', [resname])
	_write_universe(u, path)


def aa_structure(mapping, out_file="molecule_aa.pdb", resname=None, random_seed=42):
	'''
	Generate and save an all-atom structure file of a CGMapping. See CGMapping.aa_structure.
	'''
	path, resname = _prepare_output(out_file, resname)
	mol, _ = embed_aa(mapping.aa_graph, random_seed)
	_write_aa(mol, path, resname)
	return path


def _bead_order(cg_graph):
	'''
	Order of the beads in a CG structure written without ITP: by their index annotation if all beads have one,
	else by the number at the end of their name if all names end with one, else in the order of the CGSmiles string.
	'''
	nodes = sorted(cg_graph.nodes)
	def name(node):
		return cg_graph.nodes[node]['atomname']
	with_index = [node for node in nodes if 'index' in cg_graph.nodes[node]]
	if with_index:
		missing = [name(node) for node in nodes if node not in with_index]
		if missing:
			raise ValueError(f"Only some beads have an index annotation, it is missing for {', '.join(missing)}.")
		indices = {}
		for node in nodes:
			value = cg_graph.nodes[node]['index']
			try:
				indices[node] = int(value)
			except (TypeError, ValueError):
				raise ValueError(f"index={value} of bead {name(node)} is not an integer.") from None
		return sorted(nodes, key=indices.get)
	numbers = {node: re.search(r'\d+$', cg_graph.nodes[node].get('n', '')) for node in nodes}
	if all(numbers.values()):
		return sorted(nodes, key=lambda node: int(numbers[node].group()))
	return nodes


def _structure_beads(mapping):
	'''
	(beads in the order they are written, default residue name) of a CG structure: the order and residue name of
	the loaded ITP if there is one, else see _bead_order.
	'''
	topology = mapping.topology
	if topology is None:
		return _bead_order(mapping.cg_graph), None
	itp_to_cg = topology.cg_node_map(mapping.cg_graph)
	nodes = [itp_to_cg[node] for node in topology.block.nodes]
	missing = sorted(set(mapping.cg_graph.nodes) - set(nodes))
	if missing:
		names = ', '.join(mapping.cg_graph.nodes[node]['atomname'] for node in missing)
		raise ValueError(f"The beads {names} are not in the loaded ITP, so the structure cannot follow its atom order.")
	resnames = {d.get('resname') for _, d in topology.block.nodes(data=True)}
	return nodes, resnames.pop() if len(resnames) == 1 else topology.molname


def cg_structure(mapping, out_file="molecule_cg.pdb", resname=None, write_aa=True, random_seed=42):
	'''
	Generate and save a coarse-grained structure file of a CGMapping. See CGMapping.cg_structure.
	'''
	import MDAnalysis as mda
	nodes, default_resname = _structure_beads(mapping)
	path, resname = _prepare_output(out_file, resname, default_resname)
	mol, idx_map = embed_aa(mapping.aa_graph, random_seed)
	if write_aa:
		aa_stem = path.stem.replace('_cg', '_aa') if '_cg' in path.stem else f'{path.stem}_aa'
		_write_aa(mol, path.with_name(aa_stem + path.suffix), resname)
	aa_positions = mol.GetConformer().GetPositions() # in Angstrom
	positions = bead_positions(mapping.cg_graph, mapping.aa_graph, lambda atom: aa_positions[idx_map[atom]],
							   include_hydrogens=True, weighted=False)

	u = mda.Universe.empty(len(nodes), n_residues=1, atom_resindex=np.zeros(len(nodes), dtype=int), trajectory=True)
	u.add_TopologyAttr('names', [mapping.cg_graph.nodes[n]['atomname'] for n in nodes])
	u.add_TopologyAttr('resnames', [resname])
	u.add_TopologyAttr('resids', [1])
	u.add_TopologyAttr('record_types', ['HETATM'] * len(nodes))
	u.atoms.positions = np.array([positions[n] for n in nodes])
	_write_universe(u, path)
	return path
