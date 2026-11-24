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


def _prepare_output(out_file, resname):
	'''
	Add a .pdb extension if none is given, check that MDAnalysis can write the format
	and derive the residue name from the file name if none is given.
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
	resname = path.stem[:5] if resname is None else resname[:5]
	return path, resname


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
	'''Beads ordered by their 'index' property, then by the number in their name, falling back to the node id.'''
	def bead_index(node):
		data = cg_graph.nodes[node]
		if 'index' in data:
			return int(data['index'])
		match = re.search(r'\d+$', data.get('n', ''))
		return int(match.group()) if match else node
	return sorted(cg_graph.nodes, key=bead_index)


def cg_structure(mapping, out_file="molecule_cg.pdb", resname=None, write_aa=True, random_seed=42):
	'''
	Generate and save a coarse-grained structure file of a CGMapping. See CGMapping.cg_structure.
	'''
	import MDAnalysis as mda
	path, resname = _prepare_output(out_file, resname)
	mol, idx_map = embed_aa(mapping.aa_graph, random_seed)
	if write_aa:
		aa_stem = path.stem.replace('_cg', '_aa') if '_cg' in path.stem else f'{path.stem}_aa'
		_write_aa(mol, path.with_name(aa_stem + path.suffix), resname)
	aa_positions = mol.GetConformer().GetPositions() # in Angstrom
	positions = bead_positions(mapping.cg_graph, mapping.aa_graph, lambda atom: aa_positions[idx_map[atom]],
							   include_hydrogens=True, weighted=False)

	nodes = _bead_order(mapping.cg_graph)
	u = mda.Universe.empty(len(nodes), n_residues=1, atom_resindex=np.zeros(len(nodes), dtype=int), trajectory=True)
	u.add_TopologyAttr('names', [mapping.cg_graph.nodes[n]['atomname'] for n in nodes])
	u.add_TopologyAttr('resnames', [resname])
	u.add_TopologyAttr('resids', [1])
	u.add_TopologyAttr('record_types', ['HETATM'] * len(nodes))
	u.atoms.positions = np.array([positions[n] for n in nodes])
	_write_universe(u, path)
	return path
