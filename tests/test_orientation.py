import numpy as np
import pytest
from rdkit.Chem import rdDepictor

from cgsmiles_utils import CGMapping, DrawOptions, parallel_tails
from cgsmiles_utils.drawing import Renderer
from cgsmiles_utils.drawing.orientation import bead_centers, parse_rotation
from test_layout import POPC

# phenyl - ethyl - hydroxymethyl, and variants of it
PHENYL = "{[#TC5;n=R][#SC3;n=E][#SP1;n=OH]}.{#TC5=c1ccc(cc1)[$],#SC3=[$]CC[$],#SP1=[$]CO}"
REVERSED = "{[#SP1;n=OH][#SC3;n=E][#TC5;n=R]}.{#TC5=[$]c1ccc(cc1),#SC3=[$]CC[$],#SP1=OC[$]}"
TOLYL = "{[#TC5;n=R][#SC3;n=E][#SP1;n=OH]}.{#TC5=Cc1ccc(cc1)[$],#SC3=[$]CC[$],#SP1=[$]CO}"


def renderer(cgs, **kwargs):
	return Renderer(CGMapping(cgs), DrawOptions(**kwargs))


def heavy_xy(r):
	'''{aa_graph node: (x, y)} of the heavy atoms in RDKit coordinates (y up).'''
	xy = r.mol.GetConformer().GetPositions()[:, :2]
	return {node: xy[i] for node, i in r.idx_map.items() if r.mapping.aa_graph.nodes[node]['element'] != 'H'}


def centers(r):
	return bead_centers(r.mapping.cg_graph, r.idx_map, r.mol.GetConformer().GetPositions()[:, :2])


def bond_angles(r):
	xy = r.mol.GetConformer().GetPositions()[:, :2]
	return np.array([np.degrees(np.arctan2(*(xy[b.GetEndAtomIdx()] - xy[b.GetBeginAtomIdx()])[::-1]))
					 for b in r.mol.GetBonds()])


def assert_same_shape(a, b, angle=0.0, mirror=False):
	'''Coordinates b are a rotated (clockwise by angle, as drawn) and optionally mirrored copy of a.'''
	nodes = sorted(a)
	a = np.array([a[n] for n in nodes]); b = np.array([b[n] for n in nodes])
	a, b = a - a.mean(axis=0), b - b.mean(axis=0)
	if mirror:
		a = a * [-1, 1]
	theta = np.deg2rad(-angle)
	rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
	np.testing.assert_allclose(a @ rotation.T, b, atol=1e-6)


@pytest.mark.parametrize('rotate_by, expected', [
	(0, (False, 0.0)), (90, (False, 90.0)), (-12.5, (False, -12.5)), ('auto', (True, 0.0)),
	('auto+90', (True, 90.0)), ('auto - 30', (True, -30.0)), ('auto+.5', (True, 0.5))])
def test_parse_rotation(rotate_by, expected):
	assert parse_rotation(rotate_by) == expected


@pytest.mark.parametrize('rotate_by', ['automatic', '90', 'auto*2', True, None])
def test_invalid_rotate_by(rotate_by):
	with pytest.raises(ValueError, match='rotate_by'):
		DrawOptions(rotate_by=rotate_by)


@pytest.mark.parametrize('cgs', [PHENYL, TOLYL, POPC])
def test_auto_orientation(cgs):
	r = renderer(cgs)
	c = centers(r)
	# beads along the x-axis
	assert np.var(c[:, 0]) > np.var(c[:, 1])
	# most bonds at multiples of 30 degrees
	deviation = np.abs((bond_angles(r) + 15) % 30 - 15)
	assert np.mean(deviation < 0.1) > 0.5
	# the first bead on the left
	assert c[0, 0] < c[-1, 0]


def test_auto_orientation_follows_bead_order():
	forward, backward = centers(renderer(PHENYL)), centers(renderer(REVERSED))
	assert forward[0, 0] < forward[-1, 0] # ring left
	assert backward[0, 0] < backward[-1, 0] # hydroxyl left


def test_first_beads_on_top():
	# beads that do not lie on a line: the earlier ones go up (y up in RDKit coordinates)
	c = centers(renderer(POPC))
	order = np.arange(len(c)) - (len(c) - 1) / 2
	assert order @ (c[:, 1] - c[:, 1].mean()) <= 0


@pytest.mark.parametrize('kwargs', [{}, {'rotate_by': 0}, {'rotate_by': 'auto+45'}, {'flip': True},
									{'fixed_coords': parallel_tails()}])
def test_hydrogens_keep_the_heavy_atom_layout(kwargs):
	cgs = POPC if 'fixed_coords' in kwargs else PHENYL
	without, with_h = heavy_xy(renderer(cgs, **kwargs)), heavy_xy(renderer(cgs, show_hydrogens=True, **kwargs))
	assert_same_shape(without, with_h)


def test_extra_rotation_after_auto():
	auto, rotated = heavy_xy(renderer(PHENYL)), heavy_xy(renderer(PHENYL, rotate_by='auto+90'))
	assert_same_shape(auto, rotated, angle=90)


def test_numeric_rotation_without_auto():
	plain, rotated = heavy_xy(renderer(PHENYL, rotate_by=0)), heavy_xy(renderer(PHENYL, rotate_by=30))
	assert_same_shape(plain, rotated, angle=30)
	assert 'no auto orientation' in renderer(PHENYL, rotate_by=0).orientation.describe()


def test_flip_mirrors_before_rotation():
	plain = heavy_xy(renderer(PHENYL, rotate_by=0))
	assert_same_shape(plain, heavy_xy(renderer(PHENYL, rotate_by=30, flip=True)), angle=30, mirror=True)


def test_fixed_coords_keep_their_orientation():
	r = renderer(POPC, fixed_coords=parallel_tails())
	assert r.orientation.mode == 'fixed_coords'
	fixed = parallel_tails()(r.mapping.aa_graph)
	xy = heavy_xy(r)
	for node, position in fixed.items():
		np.testing.assert_allclose(xy[node], position, atol=1e-6)


@pytest.mark.parametrize('reference', [PHENYL, CGMapping(PHENYL)])
def test_align_to(reference):
	ref = renderer(PHENYL)
	r = renderer(TOLYL, align_to=reference)
	assert r.orientation.mode == 'align_to' and r.orientation.aligned_atoms == 10
	ref_xy, xy = heavy_xy(ref), heavy_xy(r)
	# the atoms the two molecules have in common are where they are in the reference, compared bead by bead
	for name in ('E', 'OH'):
		ref_atoms = sorted(ref_xy[a].round(6).tolist() for a in _bead_atoms(ref, name))
		atoms = sorted(xy[a].round(6).tolist() for a in _bead_atoms(r, name))
		assert ref_atoms == atoms


def _bead_atoms(r, name):
	cg = r.mapping.cg_graph
	node = next(n for n, d in cg.nodes(data=True) if d.get('n') == name)
	return [a for a in cg.nodes[node]['graph'].nodes if r.mapping.aa_graph.nodes[a]['element'] != 'H']


def test_align_to_applies_rotation_once():
	aligned = heavy_xy(renderer(PHENYL, align_to=PHENYL, rotate_by='auto+90'))
	assert_same_shape(heavy_xy(renderer(PHENYL, rotate_by='auto+90')), aligned)


def test_align_to_without_common_atoms_warns():
	with pytest.warns(UserWarning, match='align_to'):
		r = renderer("{[#SN6;n=A]}.{#SN6=OO}", align_to=PHENYL)
	assert r.orientation.mode == 'auto'


def test_check_orientation_reports_auto_rotation(capsys):
	CGMapping(PHENYL).to_svg(check_orientation=True)
	assert 'auto orientation: rotated by' in capsys.readouterr().out


def test_coordgen_restores_global_setting():
	before = rdDepictor.GetPreferCoordGen()
	assert '</svg>' in CGMapping(PHENYL).to_svg(coordgen=not before)
	assert rdDepictor.GetPreferCoordGen() == before
