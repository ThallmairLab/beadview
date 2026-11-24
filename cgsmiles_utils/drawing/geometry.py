'''
Geometry helpers for drawing: transformations of the 2D molecule, offset paths and distance fields.
'''
import numpy as np
from rdkit.Chem import rdMolTransforms


def rotate_molecule(mol, theta):
	'''
	Rotate the 2D conformer of a molecule in place around its centroid by theta degrees.
	'''
	conf = mol.GetConformer(0)
	centroid = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())]).mean(axis=0)
	to_origin, back = np.eye(4), np.eye(4)
	to_origin[:3, 3] = -centroid
	back[:3, 3] = centroid
	theta = np.deg2rad(theta)
	c, s = np.cos(theta), np.sin(theta)
	rotation = np.array([
		[c,  s, 0, 0],
		[-s, c, 0, 0],
		[0,  0, 1, 0],
		[0,  0, 0, 1]
	])
	rdMolTransforms.TransformConformer(conf, back @ rotation @ to_origin)


def flip_molecule(mol):
	'''
	Mirror the 2D conformer of a molecule in place horizontally (x -> -x).
	'''
	rdMolTransforms.TransformConformer(mol.GetConformer(0), np.diag([-1., 1., 1., 1.]))


def most_vertical_bonds(mol, tolerance=0.02):
	'''
	Find the bonds of a molecule that are closest to vertical in the drawing.

	Parameters
	----------
	mol : rdkit.Chem.rdchem.Mol
		Molecule with a 2D conformer.
	tolerance : float, optional, default: 0.02
		Bonds whose angle to the vertical differs by less than this (in degrees) from the
		smallest angle are all returned.

	Returns
	-------
	angle : float
		Signed angle in degrees between the most vertical bond and the y-axis of the drawing
		(90 if the molecule has no bonds).
	bonds : list of int
		Indices of the bonds within tolerance of the most vertical one.
	'''
	conf = mol.GetConformer()
	angles = []
	for bond in mol.GetBonds():
		p = conf.GetAtomPosition(bond.GetBeginAtomIdx())
		q = conf.GetAtomPosition(bond.GetEndAtomIdx())
		dx, dy = p.x - q.x, q.y - p.y # the y-axis of the drawing points down
		angle = np.degrees(np.arctan(dx / dy)) if dy != 0 else 90.0
		angles.append((angle, bond.GetIdx()))
	if not angles:
		return 90.0, []
	angle = min((a for a, _ in angles), key=abs)
	return angle, [idx for a, idx in angles if abs(a) - abs(angle) < tolerance]


def offset_path(points, distance, side):
	'''
	Points of a line that follows the path through the given points, shifted by distance to one side.
	The line ends next to the centers of the first and last bead.

	Parameters
	----------
	points : list of list or np.array
		The (x, y) positions of the beads along the path (3 for angles, 4 for dihedrals).
	distance : float
		Distance between the line and the path.
	side : int
		+1 to shift to the left of the path direction (first to last point), -1 to the right.

	Returns
	-------
	list of np.array
		The points of the shifted line.
	'''
	points = [np.asarray(p, dtype=float) for p in points]
	directions = [b - a for a, b in zip(points[:-1], points[1:])]
	normals = [side * np.array([-d[1], d[0]]) / np.linalg.norm(d) for d in directions]
	path = [points[0] + distance * normals[0]]
	for point, d1, d2, n1, n2 in zip(points[1:-1], directions[:-1], directions[1:], normals[:-1], normals[1:]):
		cos = np.dot(n1, n2)
		outer = side * (d1[0] * d2[1] - d1[1] * d2[0]) < 0 # the line runs around the outside of this corner
		if outer and 1 + cos < 2 / 1.5**2: # miter longer than 1.5 * distance: cut the corner off
			path += [point + distance * n1, point + distance * n2]
		elif 1 + cos > 1e-6:
			# corner where the two shifted segments meet; limited for very sharp turns
			path.append(point + distance * min(1 / (1 + cos), 3 / np.linalg.norm(n1 + n2)) * (n1 + n2))
		else:
			path.append(point + distance * n1)
	path.append(points[-1] + distance * normals[-1])
	return path


def offset_side(points, offset, away_from):
	'''
	Side (+1 left, -1 right of the path direction) to which a path has to be shifted to move away
	from the point away_from, or towards it for a negative offset. See offset_path.
	'''
	away_from = np.asarray(away_from, dtype=float)
	def mean_distance(side):
		return np.mean([np.linalg.norm(p - away_from) for p in offset_path(points, abs(offset), side)])
	further = 1 if mean_distance(1) >= mean_distance(-1) else -1
	return further if offset >= 0 else -further


def distance_to_skeleton(x, y, segments, points):
	'''
	Distance of the grid points (x, y) to the nearest of the line segments and points.
	'''
	distance = np.full(x.shape, np.inf)
	for (ax, ay), (bx, by) in segments:
		dx, dy = bx - ax, by - ay
		length2 = dx * dx + dy * dy
		t = np.clip(((x - ax) * dx + (y - ay) * dy) / length2, 0, 1) if length2 > 0 else 0
		distance = np.minimum(distance, np.hypot(x - ax - t * dx, y - ay - t * dy))
	for px, py in points:
		distance = np.minimum(distance, np.hypot(x - px, y - py))
	return distance


def point_segment_distances(points, segments):
	'''
	Distances between each point and each line segment.

	Parameters
	----------
	points : np.array, shape (N, 2)
	segments : np.array, shape (S, 2, 2)

	Returns
	-------
	np.array, shape (N, S)
	'''
	a, b = segments[:, 0], segments[:, 1]
	ab = b - a
	length2 = np.maximum(np.sum(ab**2, axis=1), 1e-12)
	ap = points[:, None, :] - a[None, :, :]
	t = np.clip(np.sum(ap * ab[None], axis=2) / length2[None], 0, 1)
	closest = a[None] + t[..., None] * ab[None]
	return np.linalg.norm(points[:, None, :] - closest, axis=2)
