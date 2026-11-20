import MDAnalysis as mda
import numpy as np
import pytest

from CGSmiles_utils import CGMapping


def test_cg_structure(azobenzene, tmp_path):
	azobenzene.cg_structure(tmp_path / 'azb_cg.pdb', resname='AZB')
	cg = mda.Universe(tmp_path / 'azb_cg.pdb')
	aa = mda.Universe(tmp_path / 'azb_aa.pdb')
	assert len(cg.atoms) == 9
	assert set(cg.atoms.names) == {'R1', 'R2', 'R3', 'U4', 'N5', 'R6', 'R7', 'R8', 'U9'}
	assert list(cg.residues.resids) == [1]
	assert len(aa.atoms) == 24 # C12 H10 N2


def test_unnamed_beads_get_fragment_names(tmp_path):
	CGMapping("{[#SC3][#SN4]}.{#SC3=CC[$],#SN4=[$]CO}").cg_structure(tmp_path / 'x_cg.pdb', write_aa=False)
	assert list(mda.Universe(tmp_path / 'x_cg.pdb').atoms.names) == ['SC3', 'SN4']


def test_structure_is_reproducible(simple, tmp_path):
	simple.cg_structure(tmp_path / 'a_cg.pdb', write_aa=False)
	simple.cg_structure(tmp_path / 'b_cg.pdb', write_aa=False)
	a = mda.Universe(tmp_path / 'a_cg.pdb').atoms.positions
	b = mda.Universe(tmp_path / 'b_cg.pdb').atoms.positions
	assert np.allclose(a, b)


def test_unsupported_format(simple, tmp_path):
	with pytest.raises(ValueError, match='Unsupported file type'):
		simple.aa_structure(tmp_path / 'x.foo')
