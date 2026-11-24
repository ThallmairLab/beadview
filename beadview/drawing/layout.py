'''
Fixed 2D coordinates of atoms for DrawOptions(fixed_coords=...), e.g. straight and parallel lipid tails.
'''
import numpy as np
from rdkit import Chem

from ..rdkit_conv import cgsmiles_to_rdkit

BOND_LENGTH = 1.5 # bond length of RDKit 2D depictions
TURN = np.pi / 3 # change of direction between successive bonds of a zigzag chain (120 degree bond angles)


def find_tails(mol, min_length=6):
	'''
	Tails of a molecule without hydrogens: unbranched, acyclic carbon chains that end in a terminal carbon.
	A tail runs from the terminal carbon through the carbons bonded to two heavy atoms and includes the first
	carbon outside a ring where the chain branches (e.g. the carbonyl carbon of an acyl chain).

	Parameters
	----------
	mol : rdkit.Chem.Mol
		Molecule without explicit hydrogens.
	min_length : int
		Minimum number of carbons of a tail.

	Returns
	-------
	list of list of int
		Atom indices of each tail, from the end where it is attached to the terminal carbon,
		ordered by the index of their first atom.
	'''
	Chem.FastFindRings(mol)
	tails, used = [], set()
	for atom in mol.GetAtoms():
		if atom.GetSymbol() != 'C' or atom.GetDegree() != 1:
			continue
		chain, previous = [atom], None
		while chain[-1].GetDegree() == 2 or previous is None:
			following = [n for n in chain[-1].GetNeighbors() if previous is None or n.GetIdx() != previous.GetIdx()]
			if len(following) != 1:
				break
			nxt = following[0]
			if nxt.GetSymbol() != 'C' or nxt.IsInRing():
				break
			previous = chain[-1]
			chain.append(nxt)
		indices = [a.GetIdx() for a in reversed(chain)]
		if len(indices) >= min_length and used.isdisjoint(indices):
			tails.append(indices)
			used.update(indices)
	return sorted(tails)


def _is_cis(bond):
	return bond.GetStereo() in (Chem.BondStereo.STEREOCIS, Chem.BondStereo.STEREOZ)


def _is_trans(bond):
	return bond.GetStereo() in (Chem.BondStereo.STEREOTRANS, Chem.BondStereo.STEREOE)


def zigzag(mol, tail):
	'''
	2D coordinates of a tail (see find_tails) as zigzag chain, starting at the origin and running along -y.
	Cis double bonds keep their geometry: the chain steps sideways and continues parallel to its original
	direction.

	Returns
	-------
	numpy.ndarray
		(len(tail), 2) coordinates.
	'''
	directions = [TURN / 2] # direction of each bond, as angle to -y
	turns = [] # turns[k] is the change of direction at tail[k + 1]
	for k in range(1, len(tail) - 1):
		incoming = mol.GetBondBetweenAtoms(tail[k - 1], tail[k])
		if turns and _is_cis(incoming): # both substituents of the double bond on the same side
			turn = turns[-1]
		elif turns and _is_trans(incoming):
			turn = -turns[-1]
		else: # keep the chain along its axis
			turn = min((TURN, -TURN), key=lambda t: abs(directions[-1] + t))
		turns.append(turn)
		directions.append(directions[-1] + turn)
	steps = BOND_LENGTH * np.column_stack([np.sin(directions), -np.cos(directions)])
	return np.vstack([np.zeros(2), np.cumsum(steps, axis=0)])


def parallel_tails(spacing=4.0, min_length=6):
	'''
	Layout for DrawOptions(fixed_coords=...): the tails of the molecule (see find_tails) drawn as straight,
	parallel zigzag chains pointing down, side by side in the order of the atoms in the CGSmiles string (use
	flip=True to reverse it). RDKit places the rest of the molecule (e.g. a lipid headgroup) around them.

	Parameters
	----------
	spacing : float
		Distance between neighboring tails, in RDKit units (bond length 1.5).
	min_length : int
		Minimum number of carbons of a tail, which keeps short alkyl groups (e.g. methyl groups) free.

	Returns
	-------
	callable
		Function aa_graph -> {aa_graph node: (x, y)}.
	'''
	def layout(aa_graph):
		mol, idx_map = cgsmiles_to_rdkit(aa_graph, add_hydrogens=False)
		node_of = {idx: node for node, idx in idx_map.items()}
		tails = find_tails(mol, min_length)
		coords = {}
		for i, tail in enumerate(tails):
			positions = zigzag(mol, tail)
			# mirror the tail if that moves the sideways steps at cis double bonds away from the other tails
			mirrored = positions * [-1, 1]
			if len(tails) > 1 and i == 0:
				positions = min(positions, mirrored, key=lambda p: p[:, 0].max())
			elif len(tails) > 1 and i == len(tails) - 1:
				positions = max(positions, mirrored, key=lambda p: p[:, 0].min())
			else:
				positions = min(positions, mirrored, key=lambda p: np.abs(p[:, 0]).max())
			for idx, (x, y) in zip(tail, positions):
				coords[node_of[idx]] = (x + i * spacing, y)
		return coords
	return layout
