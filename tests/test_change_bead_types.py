import pytest

from cgsmiles_utils import CGMapping


def types(mapping):
	return {d['atomname']: d['atype'] for _, d in mapping.cg_graph.nodes(data=True)}


def test_swap_types(simple):
	simple.change_bead_types({'A': 'SN4', 'B': 'SP1', 'C': 'SC3'})
	assert types(simple) == {'A': 'SN4', 'B': 'SP1', 'C': 'SC3'}
	assert len(simple.aa_graph) == len(CGMapping(simple.cgs_string).aa_graph)


def test_unchanged_type_is_noop(simple):
	before = simple.cgs_string
	assert simple.change_bead_types({'A': 'SC3'}) == before


def test_unnamed_bead_does_not_break_parsing():
	mapping = CGMapping("{[#SC3;n=A][#SN4]}.{#SC3=CC[$],#SN4=[$]CO}")
	mapping.change_bead_types({'A': 'SP1'})
	assert sorted(types(mapping).values()) == ['SN4', 'SP1']


def test_unknown_name_raises(simple):
	with pytest.raises(ValueError, match='No bead named X'):
		simple.change_bead_types({'X': 'SP1'})


def test_other_annotations_are_kept():
	mapping = CGMapping("{[#SC3;n=A;q=0][#SN4;n=B]}.{#SC3=CC[$],#SN4=[$]CO}")
	mapping.change_bead_types({'A': 'SP1'})
	assert '[#SP1;n=A;q=0]' in mapping.cgs_string


def test_fragment_names_with_common_prefix():
	# changing the TC5 bead must not copy the TC5A fragment as well
	mapping = CGMapping("{[#TC5;n=A][#TC5A;n=B]}.{#TC5=CC[$],#TC5A=[$]CO}")
	mapping.change_bead_types({'A': 'TN1'})
	assert types(mapping) == {'A': 'TN1', 'B': 'TC5'}
	assert mapping.cgs_string.count('=CC[$]') == 2 # original TC5 and the TN1 copy


def test_suffixes_stay_unique(simple):
	# all three beads become types that already exist, each needs its own suffix
	simple.change_bead_types({'A': 'SN4', 'B': 'SC3', 'C': 'SN4'})
	assert types(simple) == {'A': 'SN4', 'B': 'SC3', 'C': 'SN4'}
	elements = sorted(d['element'] for _, d in simple.aa_graph.nodes(data=True) if d['element'] != 'H')
	assert elements == ['C', 'C', 'C', 'C', 'C', 'O'] # atoms unchanged


def test_beads_sharing_a_fragment_share_the_copy():
	mapping = CGMapping("{[#SC3;n=A][#SC3;n=B]}.{#SC3=[$]CC[$]}")
	mapping.change_bead_types({'A': 'SN4', 'B': 'SN4'})
	assert types(mapping) == {'A': 'SN4', 'B': 'SN4'}
	assert mapping.cgs_string.count('#SN4=') == 1


@pytest.mark.parametrize('broken, message', [
	("{[#SN4;n=A][#SN4;n=B][#SP1;n=C]}.{#SN4=[$]CC[$],#SP1=[$]CN}", 'beads C differ'), # O -> N
	("{[#SN4;n=A][#SP1;n=C]}.{#SN4=CCCC[$],#SP1=[$]CO}", 'beads'), # bead B lost
	("{[#SC3;n=A][#SN4;n=B][#SP1;n=C]}.{#SC3=CC[$],#SN4=[$]CC[$],#SP1=[$]CO}", 'not changed as requested'),
])
def test_generated_string_is_checked(simple, monkeypatch, broken, message):
	from cgsmiles_utils import cgsmiles_string
	before = simple.cgs_string
	monkeypatch.setattr(cgsmiles_string, 'change_bead_types', lambda string, change: broken)
	with pytest.raises(ValueError, match=message):
		simple.change_bead_types({'A': 'SN4'})
	assert simple.cgs_string == before # mapping unchanged after a failed check


def test_check_passes_for_real_changes(azobenzene):
	azobenzene.change_bead_types({'R2': 'TN1', 'N5': 'SN6a', 'U4': 'U'}) # includes a no-op on a virtual site
	assert len(azobenzene.cg_graph) == 9


def test_check_detects_changed_bonds(simple):
	from cgsmiles_utils.cgsmiles_string import check_bead_type_change
	aa = simple.aa_graph.copy()
	i, j = next((i, j) for i, j in aa.edges if {aa.nodes[i]['element'], aa.nodes[j]['element']} == {'C'})
	aa.edges[i, j]['order'] = 2
	with pytest.raises(ValueError, match='different all-atom molecule'):
		check_bead_type_change((simple.cg_graph, simple.aa_graph), (simple.cg_graph, aa), {})
