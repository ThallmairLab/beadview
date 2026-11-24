import pytest

from conftest import DATA

from beadview.topology import ITPConsistencyWarning, itp_writer, add_comments_itp


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
	from beadview import CGMapping
	mapping = CGMapping("{[#SC3;n=A][#SN4;n=B]}.{#SC3=CC[$],#SN4=[$]CO}")
	itp = tmp_path / 'two.itp'
	itp.write_text("[ moleculetype ]\nTWO 1\n[ atoms ]\n1 SC3 1 TWO A 1 0 54\n2 SN4 1 TWO B 2 0 54\n"
				   "[ bonds ]\n1 2 1 0.4 5000 ; backbone ; second comment\n")
	mapping.read_itp(itp)
	assert mapping.topology.block.interactions['bonds'][0].meta['comment'].strip() == 'backbone'
	assert 'backbone' in mapping.write_itp(tmp_path, 'out').read_text()


def test_duplicate_bead_names_are_rejected(tmp_path):
	from beadview import CGMapping
	mapping = CGMapping("{[#SC3;n=A][#SN4;n=A]}.{#SC3=CC[$],#SN4=[$]CO}")
	itp = tmp_path / 'dup.itp'
	itp.write_text("[ moleculetype ]\nDUP 1\n[ atoms ]\n1 SC3 1 DUP A 1 0 54\n2 SN4 1 DUP A 2 0 54\n")
	with pytest.warns(ITPConsistencyWarning, match='not unique'):
		mapping.read_itp(itp)
	with pytest.raises(ValueError, match='not unique'):
		mapping.change_itp()


def test_change_itp_does_not_modify_loaded_itp(azobenzene):
	assert azobenzene.change_itp() == {1: 'changed from TN1 to TC5'}
	assert azobenzene.topology.block.nodes[1]['atype'] == 'TN1'


def test_with_bead_types_leaves_base_unchanged(azobenzene, tmp_path):
	before = azobenzene.cgs_string
	variant = azobenzene.with_bead_types({'N5': 'SN6a'})
	assert azobenzene.cgs_string == before
	def bead_type(mapping, name):
		return next(d['atype'] for _, d in mapping.cg_graph.nodes(data=True) if d['atomname'] == name)
	assert bead_type(azobenzene, 'N5') == 'TN1a' and bead_type(variant, 'N5') == 'SN6a'
	assert variant.topology is azobenzene.topology # read once, shared
	assert 'changed from TN1a to SN6a' in atom_lines(variant.write_itp(tmp_path, 'V1'))['N5']
	assert 'SN6a' not in (azobenzene.write_itp(tmp_path, 'BASE')).read_text()
	n5 = next(node for node, d in azobenzene.topology.block.nodes(data=True) if d['atomname'] == 'N5')
	assert azobenzene.topology.block.nodes[n5]['atype'] == 'TN1a'


def moleculetype(path):
	'''Name of the first molecule type in an ITP file.'''
	lines = [line.split() for line in open(path) if line.strip() and not line.lstrip().startswith(';')]
	return next(lines[i + 1][0] for i, line in enumerate(lines) if line[:3] == ['[', 'moleculetype', ']'])


def test_write_itp_to_file_renames_molecule_and_residue(azobenzene, tmp_path):
	path = azobenzene.write_itp(tmp_path / 'V1.itp')
	assert path == tmp_path / 'V1.itp'
	assert moleculetype(path) == 'V1'
	assert {line.split()[3] for line in atom_lines(path).values()} == {'V1'}


def test_write_itp_keeps_name_without_rename(azobenzene, tmp_path):
	path = azobenzene.write_itp(tmp_path)
	assert path == tmp_path / 'AZB.itp' and moleculetype(path) == 'AZB'
	assert {line.split()[3] for line in atom_lines(path).values()} == {'AZB'}


def test_write_itp_keeps_other_residue_names(tmp_path):
	from beadview import CGMapping
	mapping = CGMapping("{[#SC3;n=A][#SN4;n=B]}.{#SC3=CC[$],#SN4=[$]CO}")
	itp = tmp_path / 'two.itp'
	itp.write_text("[ moleculetype ]\nTWO 1\n[ atoms ]\n1 SC3 1 RA A 1 0 54\n2 SN4 2 RB B 2 0 54\n")
	mapping.read_itp(itp)
	path = mapping.write_itp(tmp_path / 'NEW.itp')
	assert moleculetype(path) == 'NEW'
	assert [line.split()[3] for line in atom_lines(path).values()] == ['RA', 'RB']


def check_problems(string, itp_text, tmp_path):
	from beadview import CGMapping
	itp = tmp_path / 'check.itp'
	itp.write_text(itp_text)
	with pytest.warns(ITPConsistencyWarning) as record:
		CGMapping(string).read_itp(itp)
	return [str(warning.message) for warning in record]


ATOMS = "[ moleculetype ]\nMOL 1\n[ atoms ]\n1 SC3 1 MOL A 1 0 54\n2 SN4 1 MOL B 2 0 54\n"


def test_check_missing_beads(tmp_path):
	problems = check_problems("{[#SC3;n=A][#SN4;n=C]}.{#SC3=CC[$],#SN4=[$]CO}", ATOMS, tmp_path)
	assert any('Atoms B of the ITP' in p for p in problems)
	assert any('Beads C of the CGSmiles string' in p for p in problems)


def test_check_ignores_virtual_sites(tmp_path):
	import warnings
	from beadview import CGMapping
	itp = tmp_path / 'vs.itp'
	itp.write_text(ATOMS + "[ virtual_sitesn ]\n2 1 1\n") # B is a virtual site in the ITP but bonded in the string
	with warnings.catch_warnings():
		warnings.simplefilter('error', ITPConsistencyWarning)
		CGMapping("{[#SC3;n=A][#SN4;n=B]}.{#SC3=CC[$],#SN4=[$]CO}").read_itp(itp)


def test_check_charges(tmp_path):
	problems = check_problems("{[#SC3;n=A][#SQ5;n=B;q=-1]}.{#SC3=CC[$],#SQ5=[$]C[O-]}", ATOMS, tmp_path)
	assert problems == ['B has charge -1 in the CGSmiles string but 0 in the ITP (the ITP charge is kept).']


def test_check_ignores_bonds(azobenzene, tmp_path):
	import warnings
	from beadview import CGMapping
	itp = tmp_path / 'bond.itp'
	# A and C are not neighbors in the string, the ITP bonds them anyway
	itp.write_text("[ moleculetype ]\nMOL 1\n[ atoms ]\n1 SC3 1 MOL A 1 0 54\n2 SN4 1 MOL B 2 0 54\n"
				   "3 SP1 1 MOL C 3 0 54\n[ bonds ]\n1 2 1 0.4 5000\n2 3 1 0.4 5000\n1 3 1 0.6 5000\n")
	with warnings.catch_warnings():
		warnings.simplefilter('error', ITPConsistencyWarning)
		CGMapping("{[#SC3;n=A][#SN4;n=B][#SP1;n=C]}.{#SC3=CC[$],#SN4=[$]CC[$],#SP1=[$]CO}").read_itp(itp)
		azobenzene.read_itp(DATA / 'azobenzene.itp') # consistent test case


def test_vs_constructions_follow_the_itp(azobenzene):
	cg_graph = azobenzene.cg_graph
	name = lambda node: cg_graph.nodes[node]['atomname']
	constructions = azobenzene.topology.vs_constructions(cg_graph)
	assert {name(vs): [name(parent) for parent in parents] for vs, parents in constructions.items()} == \
		{'U4': ['R1', 'R2', 'R3'], 'U9': ['R6', 'R7', 'R8']}
	assert azobenzene.topology.virtual_sites(cg_graph) == set(constructions)


def test_vs_construction_drawn_from_itp_only(tmp_path):
	# B is a virtual site of A in the ITP, but an ordinary bonded bead in the CGSmiles string
	from beadview import CGMapping
	itp = tmp_path / 'vs.itp'
	itp.write_text(ATOMS + "[ virtual_sitesn ]\n2 1 1\n")
	mapping = CGMapping("{[#SC3;n=A][#SN4;n=B]}.{#SC3=CC[$],#SN4=[$]CO}")
	mapping.read_itp(itp)
	svg = mapping.to_svg(show_vs_construction=True)
	assert 'id="vs_B_A"' in svg and svg.count('id="vs_') == 1
