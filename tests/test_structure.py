import MDAnalysis as mda
import numpy as np
import pytest

from cgsmiles_utils import CGMapping


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


ITP_ORDER = ['R1', 'R2', 'R3', 'N5', 'R6', 'R7', 'R8', 'U4', 'U9'] # atom order in tests/data/azobenzene.itp


def test_cg_structure_follows_itp(azobenzene, tmp_path):
	path = azobenzene.cg_structure(tmp_path / 'azb_cg.gro', write_aa=False)
	cg = mda.Universe(path)
	assert list(cg.atoms.names) == ITP_ORDER
	assert list(cg.residues.resnames) == ['AZB'] # residue name of the ITP


def test_cg_structure_of_variant_matches_written_itp(azobenzene, tmp_path):
	variant = azobenzene.with_bead_types({'N5': 'SN6a'})
	itp = variant.write_itp(tmp_path / 'V1.itp')
	cg = mda.Universe(variant.cg_structure(tmp_path / 'V1.gro', resname='V1', write_aa=False))
	section, itp_atoms = None, []
	for line in open(itp):
		fields = line.split()
		if fields[:1] == ['[']:
			section = fields[1]
		elif section == 'atoms' and fields[:1] and fields[0].isdigit():
			itp_atoms.append(fields)
	assert list(cg.atoms.names) == [atom[4] for atom in itp_atoms]
	assert {atom[3] for atom in itp_atoms} == set(cg.residues.resnames) == {'V1'}


def test_bead_order_without_itp(azobenzene):
	from cgsmiles_utils.structure import _bead_order
	names = lambda mapping: [mapping.cg_graph.nodes[n]['atomname'] for n in _bead_order(mapping.cg_graph)]
	azobenzene.topology = None
	assert names(azobenzene) == ['R1', 'R2', 'R3', 'U4', 'N5', 'R6', 'R7', 'R8', 'U9'] # by name number
	assert names(CGMapping("{[#SC3;n=X][#SN4;n=B2][#SP1;n=C1]}.{#SC3=CC[$],#SN4=[$]CC[$],#SP1=[$]CO}")) == ['X', 'B2', 'C1']
	assert names(CGMapping("{[#SC3;n=A;index=3][#SN4;n=B;index=1][#SP1;n=C;index=2]}."
						   "{#SC3=CC[$],#SN4=[$]CC[$],#SP1=[$]CO}")) == ['B', 'C', 'A']


@pytest.mark.parametrize('string, message', [
	("{[#SC3;n=A;index=2][#SN4;n=B]}.{#SC3=CC[$],#SN4=[$]CO}", 'missing for B'),
	("{[#SC3;n=A;index=x][#SN4;n=B;index=1]}.{#SC3=CC[$],#SN4=[$]CO}", 'index=x of bead A'),
])
def test_invalid_index_annotations(string, message, tmp_path):
	with pytest.raises(ValueError, match=message):
		CGMapping(string).cg_structure(tmp_path / 'x_cg.gro', write_aa=False)


def test_cg_structure_needs_all_beads_in_itp(tmp_path):
	mapping = CGMapping("{[#SC3;n=A][#SN4;n=B]}.{#SC3=CC[$],#SN4=[$]CO}")
	itp = tmp_path / 'one.itp'
	itp.write_text("[ moleculetype ]\nONE 1\n[ atoms ]\n1 SC3 1 ONE A 1 0 54\n")
	with pytest.warns(UserWarning, match='Beads B of the CGSmiles string are not in the ITP'):
		mapping.read_itp(itp)
	with pytest.raises(ValueError, match='B are not in the loaded ITP'):
		mapping.cg_structure(tmp_path / 'x_cg.gro', write_aa=False)
