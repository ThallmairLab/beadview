'''
2D layout and orientation of the atomistic molecule of a CGMapping for drawing.

The heavy atoms are laid out first and oriented, the hydrogens (if drawn) are placed around them afterwards,
so a drawing with and without hydrogens shows the same skeleton in the same orientation.

Automatic orientation (DrawOptions(rotate_by='auto')):
1. the principal axis of the bead positions is turned horizontal (the CG backbone runs left to right),
2. the depiction is straightened with the smallest rotation that puts most bonds at multiples of 30 degrees,
3. the four remaining equivalent orientations (0 and 180 degrees, mirrored or not) are decided by the bead order
   of the CGSmiles string: the first beads go to the left and, second, to the top.
'''
import re
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, replace

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDepictor, rdFMCS
from rdkit.Geometry.rdGeometry import Point2D

from ..rdkit_conv import cgsmiles_to_rdkit

AUTO_PATTERN = re.compile(r'auto\s*(?:([+-])\s*(\d+(?:\.\d*)?|\.\d+))?')
ISOTROPIC = 0.9 # ratio of the variances along the two principal axes above which the beads have no clear axis
MIN_ALIGNED_ATOMS = 3 # fewest atoms in common with the reference of align_to


def parse_rotation(rotate_by):
	'''
	Split DrawOptions.rotate_by into (auto, degrees): whether the molecule is oriented automatically and the
	rotation (in degrees) applied afterwards. Accepts a number, 'auto' or 'auto+<degrees>' / 'auto-<degrees>'.
	'''
	if isinstance(rotate_by, str):
		match = AUTO_PATTERN.fullmatch(rotate_by.strip())
		if match is None:
			raise ValueError(f"rotate_by must be a number, 'auto' or e.g. 'auto+90', got {rotate_by!r}")
		sign, degrees = match.groups()
		return True, (-1 if sign == '-' else 1) * float(degrees or 0)
	if isinstance(rotate_by, bool) or not isinstance(rotate_by, (int, float, np.number)):
		raise ValueError(f"rotate_by must be a number, 'auto' or e.g. 'auto+90', got {rotate_by!r}")
	return False, float(rotate_by)


@dataclass
class Orientation:
	'''How the molecule was oriented, see layout_molecule.'''
	mode: str # 'auto', 'manual', 'fixed_coords' or 'align_to'
	auto_rotation: float = 0.0 # rotation (degrees, counterclockwise) applied by the automatic orientation
	mirrored: bool = False # whether the automatic orientation mirrored the molecule
	rotation: float = 0.0 # rotation from rotate_by applied afterwards (numbers or 'auto+<degrees>')
	aligned_atoms: int = 0 # atoms placed as in the reference (align_to)

	def describe(self):
		if self.mode == 'auto':
			text = f"auto orientation: rotated by {self.auto_rotation:+.1f} degrees" + (", mirrored" if self.mirrored else "")
		elif self.mode == 'align_to':
			text = f"orientation of the reference (align_to), {self.aligned_atoms} atoms in common"
		elif self.mode == 'fixed_coords':
			text = "orientation given by fixed_coords"
		else:
			text = "no auto orientation"
		return text + (f", then rotated by {self.rotation:+.1f} degrees" if self.rotation else "")


# coordinates
def get_xy(mol):
	return np.array(mol.GetConformer().GetPositions())[:, :2]


def set_xy(mol, xy):
	conf = mol.GetConformer()
	for i, (x, y) in enumerate(xy):
		conf.SetAtomPosition(i, (float(x), float(y), 0.0))


def rotation_matrix(degrees):
	theta = np.deg2rad(degrees)
	c, s = np.cos(theta), np.sin(theta)
	return np.array([[c, -s], [s, c]]) # counterclockwise for y up (RDKit coordinates)


def transform(mol, matrix, center):
	'''Apply the 2x2 matrix to the 2D coordinates of mol around center.'''
	set_xy(mol, (get_xy(mol) - center) @ np.asarray(matrix).T + center)


def heavy_centroid(mol):
	heavy = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() != 1]
	return get_xy(mol)[heavy or list(range(mol.GetNumAtoms()))].mean(axis=0)


@contextmanager
def coordgen(enabled):
	'''Temporarily make RDKit use (or not use) CoordGen for 2D layouts; it is a global RDKit setting.'''
	previous = rdDepictor.GetPreferCoordGen()
	rdDepictor.SetPreferCoordGen(enabled)
	try:
		yield
	finally:
		rdDepictor.SetPreferCoordGen(previous)


# automatic orientation
def bead_centers(cg_graph, idx_map, xy):
	'''
	Centers of the heavy atoms of each bead in the order of the CGSmiles string; beads without drawn atoms
	(e.g. virtual sites) are left out.
	'''
	centers = []
	for node in sorted(cg_graph.nodes):
		atoms = [idx_map[atom] for atom in cg_graph.nodes[node]['graph'].nodes if atom in idx_map]
		if atoms:
			centers.append(xy[atoms].mean(axis=0))
	return np.array(centers).reshape(-1, 2)


def principal_angle(points):
	'''
	Angle (degrees) of the principal axis of the points to the x-axis, or None if they have no clear axis.
	'''
	if len(points) < 2:
		return None
	variances, axes = np.linalg.eigh(np.cov(points.T))
	if variances[1] <= 1e-9 or variances[0] / variances[1] > ISOTROPIC:
		return None
	axis = axes[:, 1]
	return np.degrees(np.arctan2(axis[1], axis[0]))


def auto_orient(mol, cg_graph, idx_map):
	'''
	Orient the 2D depiction of the molecule (heavy atoms only) in place, see the module docstring.

	Returns
	-------
	rotation : float
		Total rotation (degrees, counterclockwise) applied, ignoring the mirroring.
	mirrored : bool
	'''
	center = heavy_centroid(mol)
	start = get_xy(mol)
	# 1. principal axis of the beads horizontal, falling back to that of the atoms
	angle = principal_angle(bead_centers(cg_graph, idx_map, start))
	if angle is None:
		angle = principal_angle(start)
	if angle is not None:
		transform(mol, rotation_matrix(-angle), center)
	# 2. bonds at multiples of 30 degrees
	if mol.GetNumBonds():
		rdDepictor.StraightenDepiction(mol, minimizeRotation=True)
	# 3. the first beads to the left and to the top
	xy = get_xy(mol)
	centers = bead_centers(cg_graph, idx_map, xy)
	signs = np.ones(2)
	if len(centers) >= 2:
		order = np.arange(len(centers)) - (len(centers) - 1) / 2
		trend = order @ (centers - centers.mean(axis=0)) / (order @ order) # change of x and y per bead
		scale = max(np.ptp(xy, axis=0).max(), 1e-9)
		tolerance = 1e-3 * scale
		if trend[0] < -tolerance: # later beads further left
			signs[0] = -1
		if trend[1] > tolerance: # later beads further up (y up)
			signs[1] = -1
	if (signs != 1).any():
		transform(mol, np.diag(signs), heavy_centroid(mol))
	mirrored = signs[0] * signs[1] < 0
	# total rotation from the movement of the atoms, the mirroring undone
	end = get_xy(mol)
	if mirrored:
		end = end * [-1, 1]
	a, b = start - start.mean(axis=0), end - end.mean(axis=0)
	rotation = np.degrees(np.arctan2((a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]).sum(), (a * b).sum()))
	return float(rotation), bool(mirrored)


# alignment with a reference molecule
def _with_rings(mol):
	mol.UpdatePropertyCache(strict=False)
	Chem.FastFindRings(mol)
	return mol


def aligned_coords(mol, reference):
	'''
	{atom index of mol: Point2D} for the atoms of the maximum common substructure of mol and reference
	(a molecule with 2D coordinates), at the positions of the matching atoms of the reference.
	'''
	result = rdFMCS.FindMCS([_with_rings(Chem.Mol(reference)), _with_rings(Chem.Mol(mol))], timeout=10,
							ringMatchesRingOnly=True, completeRingsOnly=True)
	if result.numAtoms < MIN_ALIGNED_ATOMS:
		return {}
	query = Chem.MolFromSmarts(result.smartsString)
	ref_match = _with_rings(Chem.Mol(reference)).GetSubstructMatch(query)
	match = _with_rings(Chem.Mol(mol)).GetSubstructMatch(query)
	if not ref_match or not match:
		return {}
	ref_xy = get_xy(reference)
	return {i: Point2D(*map(float, ref_xy[j])) for i, j in zip(match, ref_match)}


def reference_layout(align_to, opts):
	'''
	The heavy atom molecule of the reference of align_to, laid out and oriented as it would be drawn with the
	same options (without the rotation of rotate_by and without flip, which are applied to the aligned molecule).
	'''
	from ..mapping import CGMapping
	reference = align_to if isinstance(align_to, CGMapping) else CGMapping(align_to)
	ref_opts = replace(opts, align_to=None, show_hydrogens=False, flip=False)
	mol, _, _ = layout_molecule(reference.aa_graph, reference.cg_graph, ref_opts, transform_final=False)
	return mol


# layout
def fixed_coord_map(aa_graph, fixed, idx_map):
	'''{RDKit atom index: Point2D} of the atoms of idx_map fixed by DrawOptions.fixed_coords.'''
	if fixed is None:
		return {}
	if callable(fixed):
		fixed = fixed(aa_graph)
	unknown = [node for node in fixed if node not in aa_graph]
	if unknown:
		raise ValueError(f"fixed_coords contains nodes that are not atoms of the aa_graph: {unknown}")
	return {idx_map[node]: Point2D(*map(float, xy)) for node, xy in fixed.items() if node in idx_map}


def layout_molecule(aa_graph, cg_graph, opts, transform_final=True):
	'''
	RDKit molecule of aa_graph with 2D coordinates as drawn with the DrawOptions opts: hydrogens if
	opts.show_hydrogens, positions from opts.fixed_coords and opts.align_to, oriented by opts.rotate_by and opts.flip.

	Parameters
	----------
	transform_final : bool
		Apply the rotation of rotate_by (numbers and 'auto+<degrees>') and flip.

	Returns
	-------
	mol : rdkit.Chem.RWMol
	idx_map : dict
		{aa_graph node: RDKit atom index}
	orientation : Orientation
	'''
	auto, degrees = parse_rotation(opts.rotate_by)
	fixed = opts.fixed_coords(aa_graph) if callable(opts.fixed_coords) else opts.fixed_coords

	heavy, heavy_idx = cgsmiles_to_rdkit(aa_graph, add_hydrogens=False)
	coord_map = {}
	orientation = Orientation('auto' if auto else 'manual')
	if opts.align_to is not None:
		coord_map = aligned_coords(heavy, reference_layout(opts.align_to, opts))
		if coord_map:
			orientation = Orientation('align_to', aligned_atoms=len(coord_map))
		else:
			warnings.warn(f"align_to: fewer than {MIN_ALIGNED_ATOMS} atoms in common with the reference, "
						  "the molecule is laid out on its own")
	fixed_map = fixed_coord_map(aa_graph, fixed, heavy_idx)
	if fixed_map and orientation.mode != 'align_to':
		orientation = Orientation('fixed_coords')
	coord_map.update(fixed_map) # fixed_coords take precedence over the positions from the reference
	with coordgen(opts.coordgen):
		rdDepictor.Compute2DCoords(heavy, canonOrient=not coord_map, coordMap=coord_map)
		if orientation.mode == 'auto':
			orientation.auto_rotation, orientation.mirrored = auto_orient(heavy, cg_graph, heavy_idx)

	mol, idx_map = heavy, heavy_idx
	if opts.show_hydrogens:
		mol, idx_map = cgsmiles_to_rdkit(aa_graph, add_hydrogens=True)
		heavy_xy = get_xy(heavy)
		coord_map = {idx_map[node]: Point2D(*map(float, heavy_xy[i])) for node, i in heavy_idx.items()}
		coord_map.update(fixed_coord_map(aa_graph, fixed, idx_map)) # fixed hydrogens
		with coordgen(opts.coordgen):
			rdDepictor.Compute2DCoords(mol, canonOrient=False, coordMap=coord_map)

	if transform_final:
		center = heavy_centroid(mol)
		if opts.flip:
			transform(mol, np.diag([-1., 1.]), center)
		if degrees: # clockwise as drawn
			transform(mol, rotation_matrix(-degrees), center)
			orientation.rotation = degrees
	return mol, idx_map, orientation
