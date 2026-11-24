import subprocess
import sys

import pytest

import cgsmiles_utils
from cgsmiles_utils import CGMapping, Topology
from cgsmiles_utils.cgsmiles_string import change_bead_types

from conftest import DATA, SIMPLE

OPTIONAL = ['pkasolver', 'torch', 'IPython', 'marimo', 'MDAnalysis', 'vermouth', 'resvg_py', 'PIL', 'matplotlib']


def run_python(code):
	return subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, check=True).stdout


def test_public_api():
	assert set(cgsmiles_utils.__all__) >= {'CGMapping', 'DrawOptions', 'BeadType', 'Topology'}
	assert cgsmiles_utils.__version__


def test_import_does_not_load_optional_dependencies():
	loaded = run_python(f"import sys, cgsmiles_utils; print([m for m in {OPTIONAL!r} if m in sys.modules])")
	assert loaded.strip() == '[]'


def test_drawing_does_not_load_optional_dependencies():
	loaded = run_python(f"import sys; from cgsmiles_utils import CGMapping; CGMapping({SIMPLE!r}).to_svg(); "
						f"print([m for m in {OPTIONAL!r} if m in sys.modules])")
	assert loaded.strip() == '[]'


def test_show_outside_notebook_uses_matplotlib():
	# RDKit's drawing code is imported first: with cairosvg this order crashed show()
	shape = run_python("import matplotlib; matplotlib.use('Agg'); from rdkit.Chem import Draw; import IPython; "
					   f"from cgsmiles_utils import CGMapping; ax = CGMapping({SIMPLE!r}).show(); "
					   "print(ax.images[0].get_array().shape)")
	assert shape.strip() == '(900, 900, 4)' # PNG at 3x the canvas size


def test_show_on_axis(simple):
	import matplotlib
	matplotlib.use('Agg')
	import matplotlib.pyplot as plt
	fig, axs = plt.subplots(1, 2)
	assert simple.show(ax=axs[1], canvas_size=(200, 100)) is axs[1]
	assert axs[1].images[0].get_array().shape == (300, 600, 4)
	plt.close(fig)


def test_show_in_notebook_displays_svg(simple, monkeypatch):
	from cgsmiles_utils.drawing import render
	shown = []
	monkeypatch.setattr(render, '_notebook_display', lambda: shown.append)
	assert simple.show(show_bead_labels=True) is None
	assert shown[0]._repr_svg_() == simple.to_svg(show_bead_labels=True)


def test_notebook_detection_outside_notebook():
	from cgsmiles_utils.drawing.render import _notebook_display
	assert _notebook_display() is None # pytest is neither a Jupyter kernel nor marimo


def test_repr_svg(simple):
	assert simple._repr_svg_() == simple.to_svg()


def test_topology_from_itp():
	topology = Topology.from_itp(DATA / 'azobenzene.itp')
	assert topology.molname == 'AZB'
	assert len(topology.block) == 9
	with pytest.raises(ValueError, match='not found'):
		Topology.from_itp(DATA / 'azobenzene.itp', molname='XYZ')


def test_without_itp(simple):
	assert simple.topology is None
	assert '</svg>' in simple.to_svg(show_bonds=True, show_angles=True) # nothing to draw, but no error
	with pytest.raises(RuntimeError, match='read_itp'):
		simple.write_itp('.')


def test_change_bead_types_string_function():
	new = change_bead_types(SIMPLE, {'A': 'SP1'})
	assert '[#SP1Z;n=A]' in new and '#SP1Z=CC[$]' in new
