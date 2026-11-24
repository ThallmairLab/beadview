import warnings
from collections.abc import Callable
from dataclasses import dataclass

from .orientation import parse_rotation

REFERENCE_SCALE = 25 # drawing scale (pixels per unit of the RDKit 2D coordinates) at scale_factor=1

# default colors of the virtual site constructions, angles and dihedrals, cycled one color per interaction
VS_PALETTE = ('#E69F00', '#56B4E9', '#009E73', '#D55E00', '#CC79A7', '#0072B2') # Okabe-Ito without yellow
ANGLE_PALETTE = ('#286E8D', '#3A9AB2', '#1B4F72', '#5DADE2', '#117A65', '#48C9B0') # blues and teals
DIHEDRAL_PALETTE = ('#8E44AD', '#C0392B', '#D35400', '#B7950B', '#A93226', '#6C3483') # reds and purples


@dataclass
class DrawOptions:
	"""
	Options for drawing a CGMapping. Lengths are in pixels at scale_factor=1 and are multiplied by scale_factor.
	"""
	# canvas
	canvas_size: tuple | str = 'auto' # (width, height) in pixels, or 'auto' to fit the canvas to the drawing
	padding: float = 10.0 # space around the drawing with canvas_size='auto'
	scale_factor: float = 1
	# 'auto': orient the molecule automatically (CG backbone horizontal, bonds at multiples of 30 degrees, first
	# beads of the CGSmiles string left and top); 'auto+90' / 'auto-30': rotate clockwise by that many degrees
	# afterwards; a number: clockwise rotation in degrees of the RDKit layout without automatic orientation. With fixed_coords
	# or align_to, 'auto' keeps their orientation.
	rotate_by: float | str = 'auto'
	flip: bool = False # mirror the molecule horizontally (before the rotation of rotate_by)
	# fixed 2D coordinates of atoms, {aa_graph node: (x, y)} in RDKit units (bond length 1.5, y up), or a function
	# aa_graph -> such a dict, e.g. parallel_tails(); RDKit lays out the other atoms around them. Applied before
	# rotate_by and flip.
	fixed_coords: dict | Callable | None = None
	# a CGMapping or CGSmiles string: the atoms the molecule has in common with it (maximum common substructure)
	# are placed as in its drawing with the same options, for consistent drawings of a series of molecules
	align_to: object | None = None
	coordgen: bool = False # lay out the molecule with RDKit's CoordGen instead of its default depictor
	remove_background: bool = True # transparent instead of white background

	# atomistic molecule
	show_hydrogens: bool = False
	color_atoms: bool = False # color heteroatoms by element instead of black and white
	show_atom_indices: bool = False
	bond_width: float = 1.5

	# beads
	show_mapping: bool = True # draw the beads; all following options need it
	style: str = 'beads' # 'beads': circles at the bead positions, 'outline': rounded outlines along the bonds of the atoms of each bead
	outline_width: float = 12.0 # style='outline' only
	border_width: float = 1.5 # border of the bead circles or outlines
	border_color: str = '#000000'
	border_alpha: float = 0.5 # opacity of the bead borders
	fill_alpha: float = 0.5 # opacity of the bead filling
	# fill colors by bead type (e.g. 'TN1a') or chemical class (e.g. 'N'), overriding the defaults
	bead_colors: dict | None = None
	# circle radii at scale_factor=1 by bead type (e.g. 'TN1a') or size ('T', 'S', 'R', 'U'), overriding the defaults
	bead_radii: dict | None = None
	include_hydrogens_in_bead_position: bool | None = None # None follows show_hydrogens
	indicate_vs: bool = True # dashed outline for beads that are virtual sites in the ITP loaded with read_itp()

	# bead labels (labels of dummy beads, type U, are always placed right above the bead)
	show_bead_labels: bool = False
	bead_type_label_color: str = '#666666'
	bead_name_label_color: str = '#000000'
	# function (bead_position, bead_radius, bead_name) -> label position, or 'optimize' to place the labels around
	# their beads with minimal overlap with other labels, beads, the molecule and the drawn bonds/angles/dihedrals
	bead_label_position: Callable | str = 'optimize'

	# interactions from the ITP loaded with read_itp(); the *_colors options take a list of colors (cycled, one per
	# interaction), a single color for all, or None for the default palette
	show_vs_construction: bool = False # dashed lines from each virtual site to the beads it is constructed from
	show_bonds: bool = False # bonds and constraints as straight lines between the beads
	show_angles: bool = False # angles as bent lines along the bonds
	show_dihedrals: bool = False # dihedrals (proper and improper) as lines along the bonds of the four beads
	# only draw the interactions (and constructions) that include at least one of these bead names
	highlight_beads: list | None = None
	vs_construction_colors: str | list | None = None # default VS_PALETTE
	vs_construction_width: float = 2.0
	bond_color: str = '#000000'
	constraint_color: str = '#020070FF'
	angle_colors: str | list | None = None # default ANGLE_PALETTE
	dihedral_colors: str | list | None = None # default DIHEDRAL_PALETTE
	cg_line_width: float = 2.0 # width of the bond, constraint, angle and dihedral lines
	cg_line_offset: float = 8.0 # distance between angle/dihedral lines and the bonds they follow, positive = away from the centroid of all beads
	line_spacing: float = 4.0 # distance between angle/dihedral lines that would otherwise overlap

	# debugging
	# report how the molecule was oriented and the most vertical bond of the molecule, highlighted in red
	check_orientation: bool = False

	def __post_init__(self):
		if self.canvas_size != 'auto' and (isinstance(self.canvas_size, str) or len(self.canvas_size) != 2):
			raise ValueError(f"canvas_size must be (width, height) or 'auto', got {self.canvas_size!r}")
		parse_rotation(self.rotate_by) # raises ValueError for invalid values
		if self.style not in ('beads', 'outline'):
			raise ValueError(f"style must be 'beads' or 'outline', got '{self.style}'")
		for option in ('fill_alpha', 'border_alpha'):
			if not 0 <= getattr(self, option) <= 1:
				raise ValueError(f"{option} must be between 0 and 1, got {getattr(self, option)}")
		if isinstance(self.bead_label_position, str) and self.bead_label_position != 'optimize':
			raise ValueError(f"bead_label_position must be a function or 'optimize', got '{self.bead_label_position}'")
		for option in ('vs_construction_colors', 'angle_colors', 'dihedral_colors'):
			colors = getattr(self, option)
			if colors is not None and not isinstance(colors, str) and len(colors) == 0:
				raise ValueError(f"{option} must not be empty")
		if isinstance(self.highlight_beads, str):
			self.highlight_beads = [self.highlight_beads]
		if self.show_bead_labels and not self.show_mapping:
			warnings.warn("show_bead_labels has no effect when show_mapping=False")
		for option in ('show_bonds', 'show_angles', 'show_dihedrals', 'show_vs_construction'):
			if getattr(self, option) and not self.show_mapping:
				warnings.warn(f"{option} has no effect when show_mapping=False")

	@property
	def hydrogens_in_bead_position(self) -> bool:
		# resolved lazily so that dataclasses.replace(opts, show_hydrogens=...) is respected
		if self.include_hydrogens_in_bead_position is None:
			return self.show_hydrogens
		return self.include_hydrogens_in_bead_position
