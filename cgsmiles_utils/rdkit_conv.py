import warnings

import networkx as nx
from rdkit import Chem


def cgsmiles_to_rdkit(mol_graph, add_hydrogens=False):
	'''
	Convert a CGSmiles molecule graph to an RDKit molecule. 
	Automatically detects bond orders and aromaticity.

	Parameters
	----------
	mol_graph : networkx.Graph
		The molecular graph to convert. Needs to be atomistic resolution.
	add_hydrogens : bool, optional
		If True, hydrogens will be added to the RDKit molecule. Default is False.

	Returns
	-------
	rdkit.Chem.rdchem.Mol
		The corresponding RDKit molecule.
	idx_mapping : dict
		A mapping from the node indices in the input graph to the atom indices in the RDKit molecule.
	'''
	mol = Chem.RWMol() # create an empty RDKit molecule

	# add atoms
	idx_mapping = {} # dictionary to map node indices in the graph to atom indices in the RDKit molecule
	for node_idx, data in sorted(mol_graph.nodes(data=True)):
		if not add_hydrogens and data['element'] == 'H' :
			continue 
		atom = Chem.Atom(data['element'])
		atom.SetFormalCharge(data['charge'])
		rdkit_idx = mol.AddAtom(atom)
		idx_mapping[node_idx] = rdkit_idx

	# add bonds
	for i, j, data in mol_graph.edges(data=True):
		if not add_hydrogens and (mol_graph.nodes[i]['element'] == 'H' or mol_graph.nodes[j]['element'] == 'H'):
			continue  
		i = idx_mapping[i]
		j = idx_mapping[j]
		match data['order']:
			case 1:
				bond_order = Chem.rdchem.BondType.SINGLE
			case 2:
				bond_order = Chem.rdchem.BondType.DOUBLE
			case 3:
				bond_order = Chem.rdchem.BondType.TRIPLE
			case 1.5:
				bond_order = Chem.rdchem.BondType.AROMATIC
			case _: 
				raise ValueError(f"Unsupported bond order: {data['order']}") 
		mol.AddBond(i,j, bond_order)

	# add isomerism information if present
	isomer_data = [d[0] for d in nx.get_node_attributes(mol_graph, 'ez_isomer').values() if d is not None]
	for isomer in isomer_data:  
		isomer_type = isomer[4]
		isomer = [idx_mapping[idx] for idx in isomer[:4]]  # convert to rdkit indices
		if isomer[1] > isomer[2]:
			continue
		bond = mol.GetBondBetweenAtoms(isomer[1], isomer[2])
		if not bond.GetBondType() == Chem.rdchem.BondType.DOUBLE:
			warnings.warn(f'Isomerisation defined on non-double bond (atoms {isomer[1]}-{isomer[2]}), it is ignored. '
						  'Check your CGSmiles string.')
			continue
		bond.SetStereoAtoms(isomer[0], isomer[3])
		if isomer_type == 'cis':
			bond.SetStereo(Chem.rdchem.BondStereo.STEREOZ)
		elif isomer_type == 'trans':
			bond.SetStereo(Chem.rdchem.BondStereo.STEREOE)
		else:
			raise ValueError(f"Unsupported isomer type: {isomer[4]}")
	return mol, idx_mapping