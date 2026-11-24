import pytest

from cgsmiles_utils.topology import itp_writer, add_comments_itp


def atom_lines(path):
	'''{atom name: line} of the [ atoms ] section of an ITP file.'''
	lines, in_atoms = {}, False
	for line in open(path):
		stripped = line.strip()
		if stripped.startswith('['):
			in_atoms = stripped.strip('[] ') == 'atoms'
		elif in_atoms and stripped[:1].isdigit():
			lines[stripped.split()[4]] = stripped
	return lines


def test_comment_lands_on_changed_atom(azobenzene, tmp_path):
	# R2 is TN1 in the ITP and TC5 (fragment TC5A) in the CGSmiles string
	azobenzene.write_itp(tmp_path, 'out')
	lines = atom_lines(tmp_path / 'out.itp')
	assert 'changed from TN1 to TC5' in lines['R2']
	assert lines['R2'].split()[1] == 'TC5'
	assert all(';' not in line for name, line in lines.items() if name != 'R2')


def test_write_itp_twice_is_identical(azobenzene, tmp_path):
	azobenzene.write_itp(tmp_path, 'first')
	azobenzene.write_itp(tmp_path, 'second')
	first = (tmp_path / 'first.itp').read_text().replace('first', 'NAME')
	second = (tmp_path / 'second.itp').read_text().replace('second', 'NAME')
	assert first == second


def test_write_itp_does_not_modify_loaded_itp(azobenzene, tmp_path):
	azobenzene.write_itp(tmp_path, 'out')
	assert azobenzene.topology.block.nodes[1]['atype'] == 'TN1'


def test_change_itp_reports_relative_to_file(azobenzene):
	assert azobenzene.change_itp() == {1: 'changed from TN1 to TC5'}
	assert azobenzene.change_itp() == {1: 'changed from TN1 to TC5'} # in place, but still reported


def test_mass_follows_size_change(azobenzene, tmp_path):
	azobenzene.change_bead_types({'N5': 'SN6a', 'R1': 'TC4'})
	azobenzene.write_itp(tmp_path, 'out')
	lines = atom_lines(tmp_path / 'out.itp')
	assert lines['N5'].split()[7] == '54.0'
	assert 'mass 36 -> 54' in lines['N5']
	assert lines['R1'].split()[7] == '36.0' # same size, mass unchanged
	assert lines['U4'].split()[7] == '0.0'


def test_itp_writer_without_cgsmiles_string(azobenzene, tmp_path):
	path = itp_writer('plain', azobenzene.topology.block, None, tmp_path)
	assert path == tmp_path / 'plain.itp'
	assert 'CGSmiles' not in path.read_text()


def test_add_comments_without_bonds_section(tmp_path):
	itp = tmp_path / 'single.itp'
	itp.write_text("[ moleculetype ]\nONE 1\n\n[ atoms ]\n; comment line\n1 SC1 1 ONE A 1 0 54\n")
	add_comments_itp({1: 'note'}, itp)
	assert '1 SC1 1 ONE A 1 0 54 ; note' in itp.read_text()


def test_interaction_comments_are_kept(tmp_path):
	from cgsmiles_utils import CGMapping
	mapping = CGMapping("{[#SC3;n=A][#SN4;n=B]}.{#SC3=CC[$],#SN4=[$]CO}")
	itp = tmp_path / 'two.itp'
	itp.write_text("[ moleculetype ]\nTWO 1\n[ atoms ]\n1 SC3 1 TWO A 1 0 54\n2 SN4 1 TWO B 2 0 54\n"
				   "[ bonds ]\n1 2 1 0.4 5000 ; backbone ; second comment\n")
	mapping.read_itp(itp)
	assert mapping.topology.block.interactions['bonds'][0].meta['comment'].strip() == 'backbone'
	assert 'backbone' in mapping.write_itp(tmp_path, 'out').read_text()


def test_duplicate_bead_names_are_rejected(tmp_path):
	from cgsmiles_utils import CGMapping
	mapping = CGMapping("{[#SC3;n=A][#SN4;n=A]}.{#SC3=CC[$],#SN4=[$]CO}")
	itp = tmp_path / 'dup.itp'
	itp.write_text("[ moleculetype ]\nDUP 1\n[ atoms ]\n1 SC3 1 DUP A 1 0 54\n2 SN4 1 DUP A 2 0 54\n")
	mapping.read_itp(itp)
	with pytest.raises(ValueError, match='not unique'):
		mapping.change_itp()
