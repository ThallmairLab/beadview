'''
Editing of CGSmiles strings.
'''
import re
import string

from .beads import fragname_to_type

BEAD_PATTERN = re.compile(r'\[#([^\];]+)([^\]]*)\]') # [#fragname;annotations]


def split_top_level(text, sep=','):
	'''Split text at sep, ignoring separators inside brackets.'''
	parts, depth, start = [], 0, 0
	for i, char in enumerate(text):
		if char in '[({':
			depth += 1
		elif char in '])}':
			depth -= 1
		elif char == sep and depth == 0:
			parts.append(text[start:i])
			start = i + 1
	parts.append(text[start:])
	return parts


def change_bead_types(cgs_string, bead_change):
	'''
	Change the types of beads in a CGSmiles string, selected by their name (n= annotation).

	Each changed bead gets its own copy of its fragment, named after the new type. If that
	name is already taken, an uppercase letter is appended (e.g. TC5Z), which is removed
	again when the type is read (see fragname_to_type). All other annotations of the beads are kept.

	Parameters
	----------
	cgs_string : str
		CGSmiles string with two or more resolutions. The beads are in the second-to-last resolution, their
		fragments in the last one. A bead inside a residue fragment (three or more resolutions) changes in
		every copy of that residue.
	bead_change : dict
		{bead name: new bead type}

	Returns
	-------
	str
		The new CGSmiles string.
	'''
	resolutions = re.split(r'\}\s*\.\s*\{', cgs_string.strip())
	if len(resolutions) < 2 or not resolutions[0].startswith('{') or not resolutions[-1].endswith('}'):
		raise ValueError("change_bead_types needs a CGSmiles string with two or more resolutions, {cg}.{fragments}.")
	resolutions[0], resolutions[-1] = resolutions[0][1:], resolutions[-1][:-1]
	upper, cg, fragments = resolutions[:-2], resolutions[-2], resolutions[-1]
	definitions = [part.strip() for part in split_top_level(fragments)]
	fragment_bodies = {} # fragment name -> definition without '#name'
	for definition in definitions:
		name, _, _ = definition.lstrip('#').partition('=')
		fragment_bodies[name] = definition[len(name) + 1:]

	beads = {} # bead name -> match of the bead in the cg part
	for match in BEAD_PATTERN.finditer(cg):
		name = re.search(r';n=([^;\]]+)', match.group(2))
		if name is None:
			continue
		if name.group(1) in beads:
			raise ValueError(f"Bead name {name.group(1)} is not unique in the CGSmiles string.")
		beads[name.group(1)] = match
	unknown = [name for name in bead_change if name not in beads]
	if unknown:
		raise ValueError(f"No bead named {', '.join(unknown)} in the CGSmiles string. Named beads: {', '.join(beads)}.")

	used = set(fragment_bodies) | {match.group(1) for match in BEAD_PATTERN.finditer(cg)}
	def unique_fragname(bead_type):
		if bead_type not in used and fragname_to_type(bead_type) == bead_type:
			return bead_type
		for letter in reversed(string.ascii_uppercase):
			if bead_type + letter not in used:
				return bead_type + letter
		raise ValueError(f"No unique fragment name left for bead type {bead_type}.")

	copies = {} # (old fragment, new type) -> new fragment, so beads sharing a fragment share the copy
	new_definitions = []
	replacements = [] # (start, end, new fragment) of the fragment names in the cg part
	for name, bead_type in bead_change.items():
		match = beads[name]
		fragname = match.group(1)
		if fragname_to_type(fragname) == bead_type:
			continue
		if (fragname, bead_type) not in copies:
			new_fragname = unique_fragname(bead_type)
			used.add(new_fragname)
			copies[(fragname, bead_type)] = new_fragname
			if fragname in fragment_bodies: # fragments without atoms (e.g. virtual sites) have no definition
				new_definitions.append(f'#{new_fragname}{fragment_bodies[fragname]}')
		replacements.append((match.start(1), match.end(1), copies[(fragname, bead_type)]))

	for start, end, new_fragname in sorted(replacements, reverse=True):
		cg = cg[:start] + new_fragname + cg[end:]
	return '{' + '}.{'.join(upper + [cg, ','.join(new_definitions + definitions)]) + '}'


def _bead_contents(cg_graph):
	'''{bead name: sorted elements of its atoms} of the named beads.'''
	return {data['n']: sorted(atom['element'] for _, atom in data['graph'].nodes(data=True))
			for _, data in cg_graph.nodes(data=True) if 'n' in data}


def check_bead_type_change(old_graphs, new_graphs, bead_change):
	'''
	Check that a bead type change (see change_bead_types) only changed bead types, by comparing the resolved
	graphs (cg_graph, aa_graph) of the old and the new CGSmiles string. Bead types are compared after removing
	the uniqueness suffix (fragname_to_type). Raises a ValueError describing the first difference.
	'''
	import networkx as nx
	(old_cg, old_aa), (new_cg, new_aa) = old_graphs, new_graphs
	if len(old_cg) != len(new_cg):
		raise ValueError(f"The new CGSmiles string has {len(new_cg)} beads instead of {len(old_cg)}.")
	old_beads, new_beads = _bead_contents(old_cg), _bead_contents(new_cg)
	if old_beads != new_beads:
		changed = sorted(set(old_beads) ^ set(new_beads) | {n for n in old_beads.keys() & new_beads.keys() if old_beads[n] != new_beads[n]})
		raise ValueError(f"The beads {', '.join(changed)} differ between the old and the new CGSmiles string.")
	match = nx.algorithms.isomorphism.categorical_node_match(['element', 'charge'], [None, 0])
	edge_match = nx.algorithms.isomorphism.categorical_edge_match('order', 1)
	if not nx.is_isomorphic(old_aa, new_aa, node_match=match, edge_match=edge_match):
		raise ValueError("The new CGSmiles string resolves to a different all-atom molecule.")
	new_types = {data['n']: fragname_to_type(data['fragname']) for _, data in new_cg.nodes(data=True) if 'n' in data}
	wrong = {name: new_types[name] for name, bead_type in bead_change.items() if new_types[name] != bead_type}
	if wrong:
		raise ValueError(f"Bead types were not changed as requested: {wrong}.")
