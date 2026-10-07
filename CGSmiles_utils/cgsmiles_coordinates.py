import networkx as nx
import vermouth
from rdkit import Chem
from rdkit.Chem import AllChem
import cgsmiles
import numpy as np
import re
import MDAnalysis as mda
import warnings
from .utils import cgsmiles_to_rdkit
warnings.filterwarnings("ignore", category=UserWarning, module='MDAnalysis')  # ignore warnings from MDAnalysis about missing box dimensions

def generate_AA_pdb(cgs_string, resname="molecule", path='.'):
	"""
	Generate an all-atom PDB file from a CGSmiles string.

	Parameters
	----------
	cgs_string : str
		The CGSmiles string representing the molecule.
	pdb_filename : str, optional
		The name of the output PDB file (default is "molecule.pdb").

	Returns
	-------
	None
		The function saves the PDB file to the specified filename.
	"""
	_, mol_graph = cgsmiles.MoleculeResolver.from_string(cgs_string).resolve()

	mol, _ =  cgsmiles_to_rdkit(mol_graph, add_hydrogens=True)

	Chem.SanitizeMol(mol)
	AllChem.EmbedMolecule(mol, AllChem.ETKDG())
	AllChem.UFFOptimizeMolecule(mol)
	pdb_block = Chem.MolToPDBBlock(mol)
	with open(f"{path}/{resname}.pdb", "w") as f:
		f.write(pdb_block)

def generate_CG_pdb(cgs_string, resname, path='.'):
	res_graph, mol_graph = cgsmiles.MoleculeResolver.from_string(cgs_string).resolve()

	generate_AA_pdb(cgs_string, f"{resname}_AA", path)
	mol = vermouth.pdb.read_pdb(f"{path}/{resname}_AA.pdb")[0]

	mapping = next(nx.isomorphism.GraphMatcher(mol, mol_graph).match())

	# check if index property is present in res_graph, if not add it
	for node, data in res_graph.nodes(data=True):
		if 'index' not in data.keys():
			res_graph.nodes[node]['index'] = int(re.search(r'\d+$', data['n']).group())  # extract number from n property

	for node in res_graph.nodes:
		pos = np.zeros((3))
		fragment = res_graph.nodes[node]['graph']
		for all_atom_node in fragment:
			pos += mol.nodes[mapping[all_atom_node]]['position']
		pos = np.array(pos)
		final_pos = pos / len(fragment) if len(fragment) > 0 else pos
		res_graph.nodes[node]['position'] = final_pos

	position_dict = {node_id: data['position'] for node_id, data in res_graph.nodes(data=True)}
	for n, d in res_graph.nodes(data=True):
		if d['fragname'] == 'U':
			res_graph.nodes[n]['position'] = np.array([0.0, 0.0, 0.0])
		else:
			res_graph.nodes[n]['position'] = position_dict[n]

	for n in res_graph.nodes:
		d = res_graph.nodes[n]
		if d['fragname'] == 'U':
			connected_nodes = [edge[1] if edge[0] == n else edge[0] for edge in res_graph.edges(n)]
			if len(connected_nodes) == 0:
				raise ValueError(f'Warning: Unconnected U node at {n}, skipping')
			coords = [res_graph.nodes[c]["position"] for c in connected_nodes if res_graph.nodes[c]["fragname"] != 'U']
			d['position'] = np.mean(coords, axis=0)

	pdb_lines = []
	for n, d in sorted(res_graph.nodes(data=True), key=lambda x: int(x[1]['index'])):
		coords = d["position"]*10
		line = f'HETATM{d["index"]:5d}  {d["n"][:3]:<3s} {resname[:3]:<3s}    1    {coords[0]:8.3f}{coords[1]:8.3f}{coords[2]:8.3f}  1.00  0.00            \n'
		pdb_lines.append(line)
	pdb_lines.append('END\n')
	write_file(f"{path}/{resname}_CG.pdb", pdb_lines)

def generate_CG_gro(cgs_string, resname, path='.'):
	generate_CG_pdb(cgs_string, resname, path)
	
	u = mda.Universe(f"{path}/{resname}_CG.pdb", in_memory=True)
	u.dimensions = np.array([10.0, 10.0, 10.0, 90.0, 90.0, 90.0]) # set box size to avoid warnings
	with mda.Writer(f"{path}/{resname}.gro", n_atoms=u.atoms.n_atoms) as W:
		W.write(u.atoms)

