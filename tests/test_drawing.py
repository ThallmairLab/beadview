
import pytest
from rdkit import Chem

from beadview.beads import bead_position
from beadview.drawing.geometry import most_vertical_bonds


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
	from beadview import CGMapping
	svg = CGMapping("{[#SC3][#SC3]}.{#SC3=[$]CC[$]}").to_svg(show_bead_labels=True)
	ids = re.findall(r'id="([^"]+)"', svg)
	assert len(ids) == len(set(ids)) == 4 # two circles, two labels


def test_label_text_is_escaped():
	from beadview import CGMapping
	svg = CGMapping("{[#SC3;n=A&B][#SN4]}.{#SC3=CC[$],#SN4=[$]CO}").to_svg(show_bead_labels=True)
	assert '>A&amp;B</tspan>' in svg


def test_background(simple):
	assert '<rect' not in simple.to_svg()
	assert '<rect' in simple.to_svg(remove_background=False)


def test_custom_label_position(simple):
	svg = simple.to_svg(show_bead_labels=True, bead_label_position=lambda pos, r, name: (pos[0] + r, pos[1]))
	assert svg.count('<text') == 3


def test_virtual_sites_are_always_drawn(azobenzene):
	svg = azobenzene.to_svg(show_angles=True, show_bead_labels=True)
	assert 'circle_bead_U4' in svg and 'circle_bead_U9' in svg
	with pytest.raises(TypeError):
		azobenzene.to_svg(show_vs=False)


def element_colors(svg, prefix):
	'''{id: stroke color} of the SVG elements whose id starts with prefix.'''
	import re
	return dict(re.findall(rf'id="({prefix}[^"]*)"\s*style="[^"]*stroke:([^;"]+)', svg))


def test_vs_construction(azobenzene):
	from beadview.drawing.options import VS_PALETTE
	svg = azobenzene.to_svg(show_vs_construction=True, show_bead_labels=True)
	colors = element_colors(svg, 'vs_')
	assert set(colors) == {f'vs_U4_{b}' for b in ('R1', 'R2', 'R3')} | {f'vs_U9_{b}' for b in ('R6', 'R7', 'R8')}
	assert {colors['vs_U4_R1'], colors['vs_U9_R6']} == set(VS_PALETTE[:2])
	assert svg.count('stroke-dasharray:6,4') == 6
	u4 = svg.split('id="circle_bead_U4"')[1].split('</g>')[0]
	assert f'stroke:{VS_PALETTE[0]};stroke-opacity:1.0' in u4 # opaque border of the virtual site
	single = element_colors(azobenzene.to_svg(show_vs_construction=True, vs_construction_colors='#123456'), 'vs_')
	assert set(single.values()) == {'#123456'}
	assert 'vs_' not in azobenzene.to_svg() # off by default


def test_angle_and_dihedral_palettes(azobenzene):
	from beadview.drawing.options import ANGLE_PALETTE, DIHEDRAL_PALETTE
	svg = azobenzene.to_svg(show_angles=True, show_dihedrals=True)
	angles = element_colors(svg, 'angle_')
	assert len(angles) == 3 and set(angles.values()) == set(ANGLE_PALETTE[:3])
	assert list(element_colors(svg, 'dihedral_').values()) == [DIHEDRAL_PALETTE[0]]
	assert set(element_colors(azobenzene.to_svg(show_angles=True, angle_colors='#000000'), 'angle_').values()) == {'#000000'}
	assert set(element_colors(azobenzene.to_svg(show_angles=True, angle_colors=['red']), 'angle_').values()) == {'red'}


def test_highlight_beads(azobenzene):
	svg = azobenzene.to_svg(show_bonds=True, show_angles=True, show_dihedrals=True, show_vs_construction=True,
							highlight_beads=['R1'])
	assert set(element_colors(svg, 'bond_')) == {'bond_R1_N5'}
	assert set(element_colors(svg, 'constraint_')) == {'constraint_R1_R2', 'constraint_R1_R3'}
	assert len(element_colors(svg, 'angle_')) == 2 and len(element_colors(svg, 'dihedral_')) == 1
	assert set(element_colors(svg, 'vs_')) == {'vs_U4_R1', 'vs_U4_R2', 'vs_U4_R3'}
	with pytest.raises(ValueError, match='XX'):
		azobenzene.to_svg(show_bonds=True, highlight_beads=['XX'])


def test_save_accepts_path(simple, tmp_path):
	path = simple.save(tmp_path / 'drawing')
	assert path == tmp_path / 'drawing.svg' and path.read_text().startswith('<?xml')


def test_save_png(simple, tmp_path):
	from PIL import Image
	path = simple.save(tmp_path / 'drawing.png', canvas_size=(300, 200), dpi=192)
	assert path == tmp_path / 'drawing.png'
	with Image.open(path) as image:
		assert image.format == 'PNG' and image.size == (600, 400) # 2x at 192 dpi
	with Image.open(simple.save(tmp_path / 'default.PNG', canvas_size=(96, 96))) as image:
		assert image.size == (300, 300) # 300 dpi by default, extension case-insensitive


def test_save_rejects_other_image_formats(simple, tmp_path):
	with pytest.raises(ValueError, match='svg or .png'):
		simple.save(tmp_path / 'drawing.pdf')
	assert not list(tmp_path.iterdir())


def test_save_appends_svg_to_other_extensions(simple, tmp_path):
	assert simple.save(tmp_path / 'mol_v1.2').name == 'mol_v1.2.svg'


def test_get_smiles(simple):
	assert simple.get_smiles() == 'CCCCCO'
	assert simple.get_smiles(add_hydrogens=True).count('[H]') == 12


@pytest.mark.parametrize('smiles', ['CC/C=C\\CC', 'CC/C=C/CC', 'CC/C=C(\\Cl)CC', 'CC/C=C(/Cl)CC',
									'C/C=C\\CC/C=C\\C', 'C/C=C\\C\\C=C/C', 'CC/C=C/C=C/C'])
def test_get_smiles_keeps_double_bond_stereo(smiles):
	from beadview import CGMapping
	assert Chem.CanonSmiles(CGMapping(f'{{[#A]}}.{{#A={smiles}}}').get_smiles()) == Chem.CanonSmiles(smiles)


CHAIN = '{[#C1;n=A1][#C1;n=A2][#C1;n=A3][#C1;n=A4][#C1;n=A5][#C1;n=A6]}.{#C1=[$]CCCC[$]}'


def view_box(svg):
	import re
	x, y, w, h = map(float, re.search(r"viewBox='([^']*)'", svg).group(1).split())
	return x, y, x + w, y + h


def test_auto_canvas_contains_all_beads_and_labels():
	import re
	from beadview import CGMapping
	svg = CGMapping(CHAIN).to_svg(show_bead_labels=True, padding=0)
	x0, y0, x1, y1 = view_box(svg)
	circles = [(float(cx), float(cy), float(r)) for cx, cy, r in re.findall(r'cx="([^"]+)" cy="([^"]+)" r="([^"]+)"', svg)]
	assert len(circles) == 12 # filling and border of 6 beads
	assert all(x0 <= cx - r and cx + r <= x1 and y0 <= cy - r and cy + r <= y1 for cx, cy, r in circles)
	assert all(x0 <= float(x) <= x1 for x in re.findall(r'<tspan x="([^"]+)"', svg))
	assert x1 - x0 > 300 # wider than the old fixed canvas, which clipped this molecule


def test_fixed_canvas_is_kept(simple):
	svg = simple.to_svg(canvas_size=(300, 200))
	assert "width='300px' height='200px' viewBox='0 0 300 200'" in svg


def test_auto_canvas_padding_and_background(simple):
	import re
	tight, padded = view_box(simple.to_svg(padding=0)), view_box(simple.to_svg(padding=20))
	assert padded[2] - padded[0] == pytest.approx(tight[2] - tight[0] + 40, abs=2)
	svg = simple.to_svg(remove_background=False)
	x0, y0, x1, y1 = view_box(svg)
	rect = re.search(r"<rect[^>]*width='([^']*)' height='([^']*)' x='([^']*)' y='([^']*)'", svg).groups()
	assert tuple(map(float, rect)) == (x1 - x0, y1 - y0, x0, y0)


def test_invalid_canvas_size(simple):
	with pytest.raises(ValueError, match='canvas_size'):
		simple.to_svg(canvas_size='big')


def test_content_bbox_path_commands():
	from beadview.drawing.svg import content_bbox
	svg = ("<path d='M 10,20 H 50 V 70 A 5 5 0 0 1 -5,30 Q 12 13, 14 15' />"
		   '<circle cx="100" cy="100" r="10" /><line x1="0" y1="0" x2="1" y2="200" />'
		   "<rect width='999' height='999' x='-500' y='-500'> </rect>")
	assert content_bbox(svg) == (-5, 0, 110, 200)
	assert content_bbox(svg, extra_boxes=[(-20, 5, 0, 5)]) == (-20, 0, 110, 200)


def test_bead_colors_by_type_and_class(simple):
	svg = simple.to_svg(bead_colors={'C': '#00ff00', 'SP1': '#123456', 'N': '#abcdef'})
	assert 'fill:#00ff00' in svg # SC3 by class C
	assert 'fill:#abcdef' in svg # SN4 by class N
	assert 'fill:#123456' in svg # SP1 by type, before class P
	assert 'fill:#ff0000' not in svg # default color of class P not used


def test_bead_radii_by_type_and_size(simple):
	import re
	svg = simple.to_svg(bead_radii={'S': 10, 'SN4': 30}, canvas_size=(300, 300))
	radii = sorted({float(r) for r in re.findall(r'stroke-dasharray:none" cx="[^"]+" cy="[^"]+" r="([^"]+)"', svg)})
	assert radii == [10.0, 30.0]


def test_custom_radius_moves_dummy_label():
	from beadview.beads import BeadType
	from beadview.drawing.labels import label_above_bead
	dummy = BeadType('U')
	assert label_above_bead((0, 0), dummy, 'U1', radius=20)[1] == pytest.approx(-21)
	assert label_above_bead((0, 0), dummy, 'U1')[1] == pytest.approx(-dummy.radius - 1)


def test_lines_through_the_bead_centers_are_spaced_on_either_side(azobenzene):
	"""With no offset, two angles along one bond may pick opposite sides: they still must not coincide."""
	import re
	svg = azobenzene.to_svg(show_angles=True, cg_line_offset=0, line_spacing=8)
	lines = re.findall(r'id="angle_[^"]*"[^>]*points="([^"]+)"', svg)
	assert len(lines) == 3
	segments = [set(zip(line.split()[:-1], line.split()[1:])) for line in lines]
	assert not any(a & b for i, a in enumerate(segments) for b in segments[i + 1:])  # no segment drawn twice
