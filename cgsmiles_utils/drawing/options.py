import warnings
from collections.abc import Callable
from dataclasses import dataclass

REFERENCE_SCALE = 25 # drawing scale (pixels per unit of the RDKit 2D coordinates) at scale_factor=1


@dataclass
class DrawOptions:
	"""
	Options for drawing a CGMapping. Lengths are in pixels at scale_factor=1 and are multiplied by scale_factor.
	"""
	# canvas
	canvas_size: tuple | str = 'auto' # (width, height) in pixels, or 'auto' to fit the canvas to the drawing
	padding: float = 10.0 # space around the drawing with canvas_size='auto'
	scale_factor: float = 1
	rotate_by: float = 0 # rotation of the molecule in degrees
	flip: bool = False # mirror the molecule horizontally
	# fixed 2D coordinates of atoms, {aa_graph node: (x, y)} in RDKit units (bond length 1.5, y up), or a function
	# aa_graph -> such a dict, e.g. parallel_tails(); RDKit lays out the other atoms around them. Applied before
	# rotate_by and flip.
	fixed_coords: dict | Callable | None = None
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
	show_vs: bool = True # draw virtual sites
	indicate_vs: bool = True # dashed outline for beads that are virtual sites in the ITP loaded with read_itp()

	# bead labels (labels of dummy beads, type U, are always placed right above the bead)
	show_bead_labels: bool = False
	bead_type_label_color: str = '#666666'
	bead_name_label_color: str = '#000000'
	# function (bead_position, bead_radius, bead_name) -> label position, or 'optimize' to place the labels around
	# their beads with minimal overlap with other labels, beads, the molecule and the drawn bonds/angles/dihedrals
	bead_label_position: Callable | str = 'optimize'

	# interactions from the ITP loaded with read_itp()
	show_bonds: bool = False # bonds and constraints as straight lines between the beads
	show_angles: bool = False # angles as bent lines along the bonds
	show_dihedrals: bool = False # dihedrals as lines along the bonds of the four beads
	bond_color: str = '#000000'
	constraint_color: str = '#020070FF'
	angle_color: str = '#286E8D'
	dihedral_color: str = '#286E8D'
	cg_line_width: float = 2.0 # width of the bond, constraint, angle and dihedral lines
	cg_line_offset: float = 8.0 # distance between angle/dihedral lines and the bonds they follow, positive = away from the centroid of all beads
	line_spacing: float = 4.0 # distance between angle/dihedral lines that would otherwise overlap

	# debugging
	check_orientation: bool = False # report (and highlight in red) the most vertical bond of the molecule

	def __post_init__(self):
		if self.canvas_size != 'auto' and (isinstance(self.canvas_size, str) or len(self.canvas_size) != 2):
			raise ValueError(f"canvas_size must be (width, height) or 'auto', got {self.canvas_size!r}")
		if self.style not in ('beads', 'outline'):
			raise ValueError(f"style must be 'beads' or 'outline', got '{self.style}'")
		for option in ('fill_alpha', 'border_alpha'):
			if not 0 <= getattr(self, option) <= 1:
				raise ValueError(f"{option} must be between 0 and 1, got {getattr(self, option)}")
		if isinstance(self.bead_label_position, str) and self.bead_label_position != 'optimize':
			raise ValueError(f"bead_label_position must be a function or 'optimize', got '{self.bead_label_position}'")
		if self.show_bead_labels and not self.show_mapping:
			warnings.warn("show_bead_labels has no effect when show_mapping=False")
		for option in ('show_bonds', 'show_angles', 'show_dihedrals'):
			if getattr(self, option) and not self.show_mapping:
				warnings.warn(f"{option} has no effect when show_mapping=False")

	@property
	def hydrogens_in_bead_position(self) -> bool:
		# resolved lazily so that dataclasses.replace(opts, show_hydrogens=...) is respected
		if self.include_hydrogens_in_bead_position is None:
			return self.show_hydrogens
		return self.include_hydrogens_in_bead_position
