import subprocess
import sys

import pytest

import cgsmiles_utils
from cgsmiles_utils import CGMapping, Topology
from cgsmiles_utils.cgsmiles_string import change_bead_types

from conftest import DATA, SIMPLE

OPTIONAL = ['pkasolver', 'torch', 'IPython', 'MDAnalysis', 'vermouth', 'fast_forward', 'cairosvg', 'PIL', 'matplotlib']


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


def test_show_after_rdkit_drawing_was_imported():
	# cairosvg segfaults in a process that loaded RDKit's drawing code first; show() must not crash
	shape = run_python("import matplotlib; matplotlib.use('Agg'); from rdkit.Chem import Draw; "
					   f"from cgsmiles_utils import CGMapping; ax = CGMapping({SIMPLE!r}).show(); "
					   "print(ax.images[0].get_array().shape)")
	assert shape.strip() == '(300, 300, 4)'


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
