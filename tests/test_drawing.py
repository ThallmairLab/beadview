import importlib
import logging

import pytest
from rdkit import Chem

from CGSmiles_utils.drawing_utils import _most_vertical_bonds, _get_bead_coordinate

# the package attribute CGMapping is the class, so get the module itself
cgmapping_module = importlib.import_module('CGSmiles_utils.CGMapping')


@pytest.mark.parametrize('style', ['beads', 'outline'])
def test_to_svg_smoke(azobenzene, style):
	svg = azobenzene.to_svg(style=style, show_bonds=True, show_angles=True, show_dihedrals=True, show_bead_labels=True)
	assert svg.startswith('<?xml') and svg.rstrip().endswith('</svg>')


@pytest.mark.parametrize('scale_factor', [0.5, 1.5, 2.3])
def test_non_integer_scale_factor(simple, scale_factor):
	assert '</svg>' in simple.to_svg(scale_factor=scale_factor)


def test_most_vertical_bonds():
	mol = Chem.MolFromSmiles('CCO')
	conf = Chem.Conformer(3)
	for i, (x, y) in enumerate([(0, 0), (0.1, 1), (1.1, 1)]): # first bond almost vertical, second horizontal
		conf.SetAtomPosition(i, (x, y, 0))
	mol.AddConformer(conf)
	angle, bonds = _most_vertical_bonds(mol)
	assert bonds == [0]
	assert abs(angle) == pytest.approx(5.71, abs=0.01)


def test_check_orientation_highlights(simple, capsys):
	svg = simple.to_svg(check_orientation=True)
	assert 'vertical lines' in capsys.readouterr().out
	assert '#FF0000' in svg


def test_bead_coordinate_without_weight():
	import networkx as nx
	from types import SimpleNamespace
	bead = nx.Graph()
	bead.add_node(0, element='C') # no weight annotation
	drawer = SimpleNamespace(GetDrawCoords=lambda i: (1.0, 2.0))
	opts = SimpleNamespace(hydrogens_in_bead_position=False)
	assert list(_get_bead_coordinate(bead, {0: 0}, drawer, opts, 'A')) == [1.0, 2.0]


def test_predict_pka_restores_global_state(simple, monkeypatch):
	monkeypatch.setattr(cgmapping_module, 'calculate_microstate_pka_values', lambda mol: [])
	monkeypatch.setattr(cgmapping_module, 'display', lambda obj: None)
	monkeypatch.setattr(cgmapping_module, 'draw_pka_map', lambda results: None)
	size = cgmapping_module.IPythonConsole.molSize
	simple.predict_pka()
	assert logging.root.manager.disable == logging.NOTSET
	assert cgmapping_module.IPythonConsole.molSize == size
