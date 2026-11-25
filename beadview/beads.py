'''
Bead types and the positions of beads and virtual sites computed from the positions of their atoms.
'''
from dataclasses import dataclass

import numpy as np

# radius of the drawn bead circles at scale_factor=1, per bead size
BEAD_RADII = {'T': 15.0, 'S': 20.0, 'R': 23.5, 'U': 7.5}
# fill colors of the drawn beads, per chemical class
BEAD_COLORS = {
	'C': '#cccccc', # light gray
	'N': '#80b3ff', # blue
	'P': '#ff0000', # red
	'X': '#ccff00', # lime
	'Q': '#ffcc00', # yellow
	'D': '#ff8000', # orange
	'U': '#8e008e', # purple
}
DEFAULT_BEAD_COLOR = '#808080' # gray
# default Martini 3 masses per bead size
MARTINI_MASSES = {'T': 36.0, 'S': 54.0, 'R': 72.0}


def fragname_to_type(fragname):
	'''
	Bead type of a fragment name: a trailing uppercase letter (used to make fragment names unique,
	e.g. TC5A) is removed, except for single letter names like U.
	'''
	if len(fragname) > 1 and fragname[-1].isupper():
		return fragname[:-1]
	return fragname


@dataclass(frozen=True)
class BeadType:
	'''
	A Martini bead type such as TC5, SN6a, P1 or U.
	'''
	name: str

	@property
	def is_dummy(self) -> bool:
		return self.name.startswith('U')

	@property
	def size(self) -> str:
		'''T, S or R (regular, the default when no size is given), U for dummy beads.'''
		if self.is_dummy:
			return 'U'
		return self.name[0] if self.name[0] in ('T', 'S', 'R') else 'R'

	@property
	def chemical_class(self) -> str:
		'''Letter of the chemical class (C, N, P, X, Q, D), U for dummy beads, '' if unknown.'''
		if self.is_dummy:
			return 'U'
		rest = self.name[1:] if self.name[0] in ('T', 'S', 'R') else self.name
		return rest[:1]

	@property
	def radius(self) -> float:
		'''Radius of the drawn bead circle at scale_factor=1.'''
		return BEAD_RADII[self.size]

	@property
	def color(self) -> str:
		'''Fill color of the drawn bead.'''
		return BEAD_COLORS.get(self.chemical_class, DEFAULT_BEAD_COLOR)

	@property
	def default_mass(self) -> float | None:
		'''Default Martini 3 mass, None for dummy beads.'''
		return MARTINI_MASSES.get(self.size)

	def color_with(self, overrides=None) -> str:
		'''Fill color, taken from overrides by bead type or chemical class if given there.'''
		for key in (self.name, self.chemical_class):
			if overrides and key in overrides:
				return overrides[key]
		return self.color

	def radius_with(self, overrides=None) -> float:
		'''Radius at scale_factor=1, taken from overrides by bead type or size if given there.'''
		for key in (self.name, self.size):
			if overrides and key in overrides:
				return float(overrides[key])
		return self.radius


def detect_virtual_nodes(cg_graph):
	'''
	Detect virtual sites: beads that are only connected by edges with order 0.

	Parameters
	----------
	cg_graph : networkx.Graph
		The coarse-grained graph.

	Returns
	-------
	dict
		{virtual node: sorted list of the nodes it is connected to}
	'''
	virtual_nodes = {}
	for node in cg_graph.nodes:
		edges = list(cg_graph.edges(node, data=True))
		if edges and all(data.get('order') == 0 for _, _, data in edges):
			virtual_nodes[node] = sorted(v if u == node else u for u, v, _ in edges)
	return virtual_nodes


def bead_position(bead_graph, coord_of, bead, include_hydrogens=True, weighted=True):
	'''
	Position of a bead: the average of the positions of its atoms, weighted by their mapping weights
	(w annotation, default 1) if weighted.

	Parameters
	----------
	bead_graph : networkx.Graph
		The atoms of the bead.
	coord_of : callable
		Function atom -> position.
	bead : int
		The bead node, for error messages.
	include_hydrogens : bool
		Include hydrogen atoms.
	weighted : bool
		Use the mapping weights instead of the plain average.
	'''
	coords, weights = [], []
	for atom, data in bead_graph.nodes(data=True):
		if not include_hydrogens and data['element'] == 'H':
			continue
		coords.append(coord_of(atom))
		weights.append(data.get('weight', 1.0) if weighted else 1.0)
	if not weights:
		raise ValueError(f"Bead {bead} has no atoms to compute its position from "
						 "(a bead of only hydrogens needs the hydrogens included in the bead position).")
	if np.all(np.array(weights) == 0):
		raise ValueError(f"All weights are zero for bead {bead}. Cannot compute bead position.")
	return np.average(coords, axis=0, weights=weights)


def _parse_wvs(bead_graph, atom, key='wVS'):
	'''
	Return the ``wVS`` annotation (or the one named `key`, e.g. ``wVS_U8``) of an atom as float,
	or None if it has none.
	'''
	value = bead_graph.nodes[atom].get(key)
	if value is None:
		return None
	try:
		return float(value)
	except (TypeError, ValueError):
		raise ValueError(f"{key}='{value}' of atom {bead_graph.nodes[atom].get('element')} "
						 "is not a number.") from None


def _own_wvs(bead_graph, atom, vs_name):
	'''
	The weight of an atom for the virtual site `vs_name` from its own annotations:
	``wVS_<vs_name>`` if given, else ``wVS``, else None.
	'''
	wvs = _parse_wvs(bead_graph, atom, f'wVS_{vs_name}') if vs_name is not None else None
	return wvs if wvs is not None else _parse_wvs(bead_graph, atom)


def _vs_weight(bead_graph, atom, vs_name=None):
	'''
	Weight of an atom of a parent bead when it is used for the virtual site with fragment name
	`vs_name` (mirrors fast_forward). The first of these that is given: the atom's own
	``wVS_<vs_name>``, its own ``wVS``, for a hydrogen the same two of the heavy atom it is
	bonded to, the atom's normal mapping weight (w, default 1).
	'''
	wvs = _own_wvs(bead_graph, atom, vs_name)
	if wvs is None and bead_graph.nodes[atom].get('element') == 'H':
		for anchor in bead_graph.neighbors(atom):
			if bead_graph.nodes[anchor].get('element') != 'H':
				wvs = _own_wvs(bead_graph, anchor, vs_name)
				break
	if wvs is None:
		return bead_graph.nodes[atom].get('weight', 1.0)
	return wvs


def vs_atom_weights(cg_graph, vnode, virtual_nodes):
	'''
	Atoms that determine the position of a virtual site and their weights.
	The atoms are those of the neighboring beads that are not virtual sites. An atom
	shared between several of these beads gets the sum of its weights.

	Returns
	-------
	dict
		{atom node in the AA graph: weight}
	'''
	parents = [n for n in cg_graph.neighbors(vnode) if n not in virtual_nodes]
	if not parents:
		data = cg_graph.nodes[vnode]
		raise ValueError(f"Virtual site {data.get('n', data.get('fragname'))} (bead {vnode}) "
						 "is not bonded to any bead with atoms, so its position is undefined.")
	vs_name = cg_graph.nodes[vnode].get('fragname')
	weights = {}
	for parent in parents:
		bead_graph = cg_graph.nodes[parent]['graph']
		for atom in bead_graph.nodes:
			weights[atom] = weights.get(atom, 0.0) + _vs_weight(bead_graph, atom, vs_name)
	return weights


def vs_position(atom_weights, coord_of, aa_graph, vnode, include_hydrogens=True):
	'''
	Weighted average of the atom coordinates given by coord_of(atom) for a virtual site.
	'''
	coords, weights = [], []
	for atom, weight in atom_weights.items():
		if not include_hydrogens and aa_graph.nodes[atom]['element'] == 'H':
			continue
		coords.append(coord_of(atom))
		weights.append(weight)
	if not weights or np.all(np.array(weights) == 0):
		raise ValueError(f"All wVS weights are zero for virtual site {vnode}. Cannot compute its position.")
	return np.average(coords, axis=0, weights=weights)


def bead_positions(cg_graph, aa_graph, coord_of, include_hydrogens=True, weighted=True):
	'''
	Positions of all beads, computed from the atom positions given by coord_of(atom).
	Beads are placed at the (weighted, see bead_position) center of their atoms, virtual sites
	at the center of the atoms of the beads they are connected to, weighted by wVS (see vs_atom_weights).

	Returns
	-------
	dict
		{cg_graph node: position}, virtual sites last.
	'''
	virtual_nodes = detect_virtual_nodes(cg_graph)
	positions = {}
	for node, data in cg_graph.nodes(data=True):
		if node not in virtual_nodes:
			positions[node] = bead_position(data['graph'], coord_of, node, include_hydrogens, weighted)
	for node in virtual_nodes:
		positions[node] = vs_position(vs_atom_weights(cg_graph, node, virtual_nodes), coord_of,
									  aa_graph, node, include_hydrogens)
	return positions
