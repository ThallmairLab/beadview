import logging
import sys
from types import SimpleNamespace

import pytest
from rdkit import Chem

from cgsmiles_utils.beads import bead_position
from cgsmiles_utils.drawing.geometry import most_vertical_bonds


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
	angle, bonds = most_vertical_bonds(mol)
	assert bonds == [0]
	assert abs(angle) == pytest.approx(5.71, abs=0.01)


def test_check_orientation_highlights(simple, capsys):
	svg = simple.to_svg(check_orientation=True)
	assert 'vertical lines' in capsys.readouterr().out
	assert '#FF0000' in svg


def test_bead_coordinate_without_weight():
	import networkx as nx
	bead = nx.Graph()
	bead.add_node(0, element='C') # no weight annotation
	assert list(bead_position(bead, lambda atom: (1.0, 2.0), 'A', include_hydrogens=False)) == [1.0, 2.0]


def test_element_ids_are_unique():
	import re
	from cgsmiles_utils import CGMapping
	svg = CGMapping("{[#SC3][#SC3]}.{#SC3=[$]CC[$]}").to_svg(show_bead_labels=True)
	ids = re.findall(r'id="([^"]+)"', svg)
	assert len(ids) == len(set(ids)) == 4 # two circles, two labels


def test_label_text_is_escaped():
	from cgsmiles_utils import CGMapping
	svg = CGMapping("{[#SC3;n=A&B][#SN4]}.{#SC3=CC[$],#SN4=[$]CO}").to_svg(show_bead_labels=True)
	assert '>A&amp;B</tspan>' in svg


def test_background(simple):
	assert '<rect' not in simple.to_svg()
	assert '<rect' in simple.to_svg(remove_background=False)


def test_custom_label_position(simple):
	svg = simple.to_svg(show_bead_labels=True, bead_label_position=lambda pos, r, name: (pos[0] + r, pos[1]))
	assert svg.count('<text') == 3


def test_hidden_virtual_sites(azobenzene):
	svg = azobenzene.to_svg(show_vs=False, show_angles=True, show_bead_labels=True)
	assert 'circle_bead_U4' not in svg and 'circle_bead_R1' in svg


def test_save_accepts_path(simple, tmp_path):
	path = simple.save(tmp_path / 'drawing')
	assert path == tmp_path / 'drawing.svg' and path.read_text().startswith('<?xml')


def test_get_smiles(simple):
	assert simple.get_smiles() == 'CCCCCO'
	assert simple.get_smiles(add_hydrogens=True).count('[H]') == 12


def test_predict_pka_restores_global_state(simple, monkeypatch):
	from rdkit.Chem.Draw import IPythonConsole # imported before IPython.display is replaced below
	# stand-ins for pkasolver and IPython, so the test neither needs them nor runs the model
	fake_pkasolver = SimpleNamespace(calculate_microstate_pka_values=lambda mol: [],
									 draw_pka_map=lambda results: None, draw_pka_reactions=lambda results: None)
	monkeypatch.setitem(sys.modules, 'pkasolver.query', fake_pkasolver)
	monkeypatch.setitem(sys.modules, 'IPython.display', SimpleNamespace(display=lambda obj: None))
	size = IPythonConsole.molSize
	simple.predict_pka()
	assert logging.root.manager.disable == logging.NOTSET
	assert IPythonConsole.molSize == size
