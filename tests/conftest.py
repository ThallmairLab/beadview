from pathlib import Path

import pytest

from beadview import CGMapping

DATA = Path(__file__).parent / 'data'

# azobenzene with two virtual sites; matches tests/data/azobenzene.itp
AZOBENZENE = ("{[#TC5;n=R1]1.2[#TC5A;n=R2].3[#TC5;n=R3]1.([#U;n=U4]23)[#TN1a;n=N5]"
			  "[#TC5;n=R6]1.2[#TC5A;n=R8].3[#TC5;n=R7]1.([#U;n=U9]23)}."
			  "{#TC5=[!]ccc[!]/[>],#TC5A=[!]ccc[!],#TN1a=[<]nn[<]}")

# propanol-like molecule with three beads
SIMPLE = "{[#SC3;n=A][#SN4;n=B][#SP1;n=C]}.{#SC3=CC[$],#SN4=[$]CC[$],#SP1=[$]CO}"


@pytest.fixture
def azobenzene():
	mapping = CGMapping(AZOBENZENE)
	mapping.read_itp(DATA / 'azobenzene.itp')
	return mapping


@pytest.fixture
def simple():
	return CGMapping(SIMPLE)
