'''
Content, size and placement of the bead labels.
'''
import numpy as np

from .geometry import point_segment_distances

LABEL_FONT_SIZE = 13 # at scale_factor=1


def label_lines(bead_type, bead_name=None):
	'''
	Lines of a bead label from top to bottom as (text, is_name) tuples.
	Dummy beads with a name only show the name, other beads show the name above the type.
	'''
	if bead_name is None:
		return [(bead_type.name, False)]
	if bead_type.is_dummy:
		return [(bead_name, True)]
	return [(bead_name, True), (bead_type.name, False)]


def label_extent(bead_type, bead_name=None, scale_factor=1.0):
	'''
	Approximate extent of a bead label relative to its anchor (left end of the baseline of the last line).

	Returns
	-------
	tuple of float
		(width, above, below): the label spans x to x + width and y - above to y + below.
	'''
	fontsize = LABEL_FONT_SIZE * scale_factor
	lines = label_lines(bead_type, bead_name)
	width = 0.62 * fontsize * max(len(text) for text, _ in lines) # average glyph width of sans-serif fonts
	above = (len(lines) - 1) * fontsize + 0.75 * fontsize # cap height of the top line
	below = 0.2 * fontsize # descenders
	return width, above, below


def label_above_bead(bead_position, bead_type, bead_name=None, scale_factor=1.0, gap=1.0):
	'''
	Label anchor that centers the label horizontally on the bead with its baseline just above the top edge of the bead.
	'''
	width, _, _ = label_extent(bead_type, bead_name, scale_factor)
	return (bead_position[0] - width / 2, bead_position[1] - bead_type.radius * scale_factor - gap * scale_factor)


def optimize_label_positions(bead_positions, bead_radii, label_sizes, circles, segments, canvas_size, gap,
							 fixed_centers=None, n_directions=24, n_rings=3, max_sweeps=20):
	'''
	Place one rectangular label next to each bead such that the labels overlap as little as possible
	with each other, with the beads and with the obstacles of the underlying structure.

	Each label can be placed at n_directions directions around its bead and n_rings distances from it.
	The overlap of a candidate with the beads, circles, segments and the area outside the canvas is
	estimated on a grid of points inside the label, the overlap with other labels from the exact
	intersection area. Labels are first placed independently and then repeatedly moved to their best
	position given the other labels until no label moves anymore.

	Parameters
	----------
	bead_positions : np.array, shape (B, 2)
		Centers of the beads.
	bead_radii : np.array, shape (B,)
		Radii of the beads.
	label_sizes : np.array, shape (B, 2)
		Width and height of the label of each bead.
	circles : np.array, shape (C, 3)
		Circular obstacles (x, y, radius), e.g. atom labels.
	segments : np.array, shape (S, 5)
		Line obstacles (x1, y1, x2, y2, half width), e.g. bonds.
	canvas_size : tuple
		Width and height of the drawing; area outside is avoided.
	gap : float
		Clearance kept between labels and their own bead and to all obstacles.
	fixed_centers : dict, optional, default: None
		Centers of labels that are not moved, keyed by bead index. They are still avoided by the other labels.

	Returns
	-------
	np.array, shape (B, 2)
		Centers of the labels.
	'''
	bead_positions = np.asarray(bead_positions, dtype=float).reshape(-1, 2)
	bead_radii = np.asarray(bead_radii, dtype=float)
	label_sizes = np.asarray(label_sizes, dtype=float).reshape(-1, 2)
	circles = np.asarray(circles, dtype=float).reshape(-1, 3)
	segments = np.asarray(segments, dtype=float).reshape(-1, 5)
	n_labels = len(bead_positions)
	if n_labels == 0:
		return bead_positions.copy()

	# candidate directions, starting at the upper right (the default position), y points down in SVG
	angles = -np.pi / 4 + 2 * np.pi * np.arange(n_directions) / n_directions
	directions = np.stack([np.cos(angles), np.sin(angles)], axis=1)
	# shift the label by half its size along the direction so that its edge (or corner) faces the bead
	corner = np.clip(directions * np.sqrt(2), -1, 1)
	ring_steps = np.arange(n_rings)

	# weights of the different kinds of overlap and of small preferences that break ties
	w_label, w_canvas, w_bead, w_circle, w_segment = 4.0, 4.0, 2.0, 1.5, 1.0
	w_ring, w_direction = 0.05, 0.02

	grid = np.stack(np.meshgrid(np.linspace(-0.5, 0.5, 8), np.linspace(-0.5, 0.5, 4)), axis=-1).reshape(-1, 2)
	candidates, static_costs = [], []
	fixed_centers = fixed_centers or {}
	for i in range(n_labels):
		if i in fixed_centers:
			candidates.append(np.asarray(fixed_centers[i], dtype=float).reshape(1, 2))
			static_costs.append(np.zeros(1))
			continue
		size = label_sizes[i]
		ring_distance = bead_radii[i] + gap + ring_steps * 0.5 * size[1]
		centers = (bead_positions[i][None, None] + ring_distance[:, None, None] * directions[None]
				   + corner[None] * size / 2).reshape(-1, 2) # (rings * directions, 2)
		points = (centers[:, None] + grid[None] * size).reshape(-1, 2)
		n_points = len(grid)

		def fraction(inside):
			return inside.reshape(len(centers), n_points).mean(axis=1)

		cost = np.zeros(len(centers))
		# beads, including the own one when a label would be pushed into it
		d = np.linalg.norm(points[:, None] - bead_positions[None], axis=2)
		cost += w_bead * fraction(np.any(d < bead_radii[None] + gap, axis=1))
		if len(circles):
			d = np.linalg.norm(points[:, None] - circles[None, :, :2], axis=2)
			cost += w_circle * fraction(np.any(d < circles[None, :, 2] + gap, axis=1))
		if len(segments):
			d = point_segment_distances(points, segments[:, :4].reshape(-1, 2, 2))
			cost += w_segment * fraction(np.any(d < segments[None, :, 4] + gap, axis=1))
		outside = (points[:, 0] < 0) | (points[:, 1] < 0) | (points[:, 0] > canvas_size[0]) | (points[:, 1] > canvas_size[1])
		cost += w_canvas * fraction(outside)
		cost += w_ring * np.repeat(ring_steps, n_directions)
		cost += w_direction * np.tile((1 - np.cos(angles - angles[0])) / 2, n_rings)
		candidates.append(centers)
		static_costs.append(cost)

	def label_overlap(i, centers, choice):
		'''Intersection area of candidate labels of label i with all other placed labels, relative to the label area.'''
		others = [j for j in range(n_labels) if j != i]
		if not others:
			return np.zeros(len(centers))
		other_centers = np.array([candidates[j][choice[j]] for j in others])
		half = (label_sizes[i] + gap) / 2
		other_half = (label_sizes[others] + gap) / 2
		overlap = np.clip(np.minimum(centers[:, None] + half, other_centers[None] + other_half)
						  - np.maximum(centers[:, None] - half, other_centers[None] - other_half), 0, None)
		return np.prod(overlap, axis=2).sum(axis=1) / np.prod(label_sizes[i] + gap)

	choice = [int(np.argmin(cost)) for cost in static_costs]
	for _ in range(max_sweeps):
		costs = [static_costs[i] + w_label * label_overlap(i, candidates[i], choice) for i in range(n_labels)]
		moved = False
		# the labels in the worst position move first
		for i in sorted(range(n_labels), key=lambda i: -costs[i][choice[i]]):
			cost = static_costs[i] + w_label * label_overlap(i, candidates[i], choice)
			best = int(np.argmin(cost))
			if cost[best] < cost[choice[i]] - 1e-9:
				choice[i] = best
				moved = True
		if not moved:
			break
	return np.array([candidates[i][choice[i]] for i in range(n_labels)])
