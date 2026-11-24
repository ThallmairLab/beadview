import pytest

from cgsmiles_utils.beads import BeadType, fragname_to_type


@pytest.mark.parametrize('name, size, chemical_class, radius, color', [
	('TC5', 'T', 'C', 15.0, '#cccccc'),
	('SN6a', 'S', 'N', 20.0, '#80b3ff'),
	('P1', 'R', 'P', 23.5, '#ff0000'),
	('RQ5', 'R', 'Q', 23.5, '#ffcc00'),
	('U', 'U', 'U', 7.5, '#8e008e'),
	('T', 'T', '', 15.0, '#808080'), # size only, unknown class
])
def test_bead_type(name, size, chemical_class, radius, color):
	bead_type = BeadType(name)
	assert (bead_type.size, bead_type.chemical_class, bead_type.radius, bead_type.color) == (size, chemical_class, radius, color)


def test_default_mass():
	assert BeadType('TC5').default_mass == 36.0
	assert BeadType('SC3').default_mass == 54.0
	assert BeadType('C1').default_mass == 72.0
	assert BeadType('U').default_mass is None


@pytest.mark.parametrize('fragname, bead_type', [('TC5A', 'TC5'), ('TC5', 'TC5'), ('U', 'U'), ('SN6a', 'SN6a')])
def test_fragname_to_type(fragname, bead_type):
	assert fragname_to_type(fragname) == bead_type
