import numpy as np
import pytest
from rdkit import Chem

from cgsmiles_utils import CGMapping, DrawOptions, parallel_tails
from cgsmiles_utils.drawing import Renderer
from cgsmiles_utils.drawing.layout import BOND_LENGTH, find_tails, zigzag
from cgsmiles_utils.rdkit_conv import cgsmiles_to_rdkit

# POPC: oleoyl (cis double bond) at sn-2, palmitoyl at sn-1
POPC = ("{[#Q1;n=NC3][#Q5;n=PO4][#SN4a;n=GL1]([#C1;n=C1A][#C4h;n=D2A][#C1;n=C3A][#SC1;n=C4A])"
		"[#N4a;n=GL2][#C1;n=C1B][#C1;n=C2B][#C1;n=C3B][#SC1;n=C4B]}."
		"{#Q1=[$]CC[N+](C)(C)C,#Q5=[$]OP(=O)([O-])O[$1],#SN4a=[$1]CC(C[$2])OC(=O)[$],"
		"#N4a=[$2]OC(=O)[$],#C4h=[$]CCC/C=C\\C[$],#C1=[$]CCCC[$],#SC1=[$]CCC}")


@pytest.fixture
def popc():
	return CGMapping(POPC)


def heavy_mol(mapping):
	return cgsmiles_to_rdkit(mapping.aa_graph)[0]


def drawn_positions(mapping, **kwargs):
	'''{aa_graph node: (x, y)} of the 2D layout, before rotate_by and flip.'''
	renderer = Renderer(mapping, DrawOptions(**kwargs))
	conf = renderer.mol.GetConformer()
	return {node: np.array(conf.GetAtomPosition(idx))[:2] for node, idx in renderer.idx_map.items()}


def side(p, q, r):
	'''Side of point r relative to the line p -> q.'''
	(ux, uy), (vx, vy) = q - p, r - p
	return np.sign(ux * vy - uy * vx)


def test_find_tails(popc):
	mol = heavy_mol(popc)
	tails = find_tails(mol)
	assert sorted(len(t) for t in tails) == [16, 18] # palmitoyl and oleoyl, with the carbonyl carbon
	for tail in tails:
		assert mol.GetAtomWithIdx(tail[-1]).GetDegree() == 1
		assert mol.GetAtomWithIdx(tail[0]).GetDegree() == 3 # carbonyl carbon
		assert all(mol.GetAtomWithIdx(i).GetSymbol() == 'C' for i in tail)


def test_find_tails_ignores_rings_and_short_chains():
	assert find_tails(Chem.MolFromSmiles('CCCCCCc1ccccc1')) == [[5, 4, 3, 2, 1, 0]]
	assert find_tails(Chem.MolFromSmiles('C[N+](C)(C)CCO')) == []
	assert len(find_tails(Chem.MolFromSmiles('CCCCCCCCCC'))) == 1 # both ends are terminal, found once


@pytest.mark.parametrize('smiles', ['CCCC/C=C\\CCCC', 'CCCC/C=C/CCCC', 'CC/C=C\\C\\C=C/C/C=C\\CCC'])
def test_zigzag_keeps_double_bond_geometry(smiles):
	mol = Chem.MolFromSmiles(smiles)
	tail = find_tails(mol)[0]
	positions = dict(zip(tail, zigzag(mol, tail)))
	steps = np.diff([positions[i] for i in tail], axis=0)
	assert np.allclose(np.linalg.norm(steps, axis=1), BOND_LENGTH)
	assert np.all(steps[:, 1] <= 1e-9) # never back up
	for bond in mol.GetBonds():
		if bond.GetStereo() == Chem.BondStereo.STEREONONE:
			continue
		a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
		s, t = bond.GetStereoAtoms()
		same_side = side(positions[a], positions[b], positions[s]) == side(positions[a], positions[b], positions[t])
		assert same_side == (bond.GetStereo() in (Chem.BondStereo.STEREOCIS, Chem.BondStereo.STEREOZ))


def test_fixed_coords_dict(simple):
	fixed = {0: (0.0, 0.0), 1: (1.5, 0.0)}
	positions = drawn_positions(simple, fixed_coords=fixed)
	for node, xy in fixed.items():
		assert positions[node] == pytest.approx(xy)


def test_fixed_coords_unknown_node(simple):
	with pytest.raises(ValueError, match='not atoms'):
		simple.to_svg(fixed_coords={999: (0, 0)})


@pytest.mark.parametrize('show_hydrogens', [False, True])
def test_parallel_tails(popc, show_hydrogens):
	positions = drawn_positions(popc, fixed_coords=parallel_tails(spacing=5.0), show_hydrogens=show_hydrogens)
	mol = heavy_mol(popc)
	node_of = {idx: node for node, idx in cgsmiles_to_rdkit(popc.aa_graph)[1].items()}
	tails = [[positions[node_of[i]] for i in tail] for tail in find_tails(mol)]
	assert [tail[0] for tail in tails] == [pytest.approx((0, 0)), pytest.approx((5.0, 0))]
	left, right = (np.array(tail) for tail in tails)
	assert left[:, 0].max() < right[:, 0].min() # side by side, without crossing
	for tail in (left, right):
		assert tail[-1][1] < -15 # pointing down


def test_parallel_tails_without_tails(simple):
	assert simple.to_svg(fixed_coords=parallel_tails()) == simple.to_svg()


def test_parallel_tails_drawing(popc):
	svg = popc.to_svg(fixed_coords=parallel_tails(), show_bead_labels=True, rotate_by=90, flip=True)
	assert svg.rstrip().endswith('</svg>')
