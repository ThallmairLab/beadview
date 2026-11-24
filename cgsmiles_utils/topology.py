'''
GROMACS topologies (ITP files) of coarse-grained molecules and their link to the CG graph of a CGMapping.
vermouth is only imported when an ITP is read or written.
'''
import copy
from pathlib import Path

import numpy as np

from .beads import BeadType


def _read_itp(lines, ff):
	'''
	Parse ITP lines into the force field ff with vermouth's ITP parser, keeping the comment of each interaction line
	(as fast_forward does), so that write() reproduces them.
	'''
	from vermouth.gmx.itp_read import ITPDirector
	from vermouth.parser_utils import split_comments

	class CommentKeepingParser(ITPDirector):
		current_comment = ''

		def parse(self, file_handle):
			def remember_comments(lines):
				for line in lines:
					comment = split_comments(line, self.COMMENT_CHAR)[1]
					if self.COMMENT_CHAR in comment: # only the first of several comments
						comment = split_comments(comment, self.COMMENT_CHAR)[0]
					self.current_comment = comment
					yield line
			return super().parse(remember_comments(file_handle))

		def _base_parser(self, tokens, context, section, atom_idxs):
			super()._base_parser(tokens, context, section, atom_idxs)
			if self.current_comment:
				context.interactions[section][-1].meta['comment'] = self.current_comment

	return list(CommentKeepingParser(ff).parse(iter(lines)))


class ITPConsistencyWarning(UserWarning):
	'''The loaded ITP and the CGSmiles string do not match, see Topology.check.'''


class Topology:
	'''
	A molecule type read from an ITP file.

	Attributes
	----------
	ff : vermouth.forcefield.ForceField
		The force field holding the parsed molecule types.
	molname : str
		The molecule type used.
	original : dict
		{block node: (atom type, mass)} as read, so changes are always reported relative to the file.
	'''
	def __init__(self, ff, molname):
		if molname not in ff.blocks:
			raise ValueError(f"Molname {molname} not found in the ITP file.")
		self.ff = ff
		self.molname = molname
		self.original = {node: (d['atype'], d.get('mass')) for node, d in self.block.nodes(data=True)}

	@classmethod
	def from_itp(cls, path, molname=None):
		'''
		Read an ITP file. molname selects the molecule type, by default the first one in the file.
		'''
		from vermouth.forcefield import ForceField
		with open(path) as f:
			lines = f.readlines()
		ff = ForceField("dummy")
		_read_itp(lines, ff)
		if not ff.blocks:
			raise ValueError(f"No molecule type found in {path}.")
		return cls(ff, molname if molname is not None else next(iter(ff.blocks)))

	@property
	def block(self):
		'''The vermouth block of the molecule type.'''
		return self.ff.blocks[self.molname]

	def cg_node_map(self, cg_graph):
		'''
		Map the nodes of the block to cg_graph nodes via their atom names.
		'''
		cg_by_name = {}
		for n, attrs in cg_graph.nodes(data=True):
			if attrs["atomname"] in cg_by_name:
				raise ValueError(f"Bead name {attrs['atomname']} is not unique in the CGSmiles string, "
								 "so the beads cannot be matched to the ITP by name.")
			cg_by_name[attrs["atomname"]] = n
		mapping = {}
		for node, d in self.block.nodes(data=True):
			if d["atomname"] not in cg_by_name:
				raise ValueError(f"Atom {d['atomname']} of the ITP has no bead with this name in the CGSmiles string.")
			mapping[node] = cg_by_name[d["atomname"]]
		return mapping

	def check(self, cg_graph):
		'''
		Differences between the molecule type and the CG graph of a CGSmiles string that would make them
		inconsistent: bead names that are not unique or present in only one of them and beads whose charge differs
		(q= in the string). Bonds, virtual sites and other interactions are not compared with the string.

		Returns
		-------
		list of str
			One message per problem, empty if they are consistent.
		'''
		problems = []
		itp_names, cg_names = {}, {}
		for names, nodes, where in ((itp_names, self.block.nodes(data=True), 'ITP'),
									(cg_names, cg_graph.nodes(data=True), 'CGSmiles string')):
			duplicates = []
			for node, data in nodes:
				if data['atomname'] in names:
					duplicates.append(data['atomname'])
				names[data['atomname']] = node
			if duplicates:
				problems.append(f"Bead names {', '.join(sorted(set(duplicates)))} are not unique in the {where}, "
								"so the beads cannot be matched to the ITP by name.")
		only_itp = [name for name in itp_names if name not in cg_names]
		only_cg = [name for name in cg_names if name not in itp_names]
		if only_itp:
			problems.append(f"Atoms {', '.join(only_itp)} of the ITP have no bead with this name in the CGSmiles string.")
		if only_cg:
			problems.append(f"Beads {', '.join(only_cg)} of the CGSmiles string are not in the ITP.")

		for name in (name for name in itp_names if name in cg_names):
			cg_charge = float(cg_graph.nodes[cg_names[name]].get('charge', 0.0))
			itp_charge = float(self.block.nodes[itp_names[name]].get('charge', 0.0))
			if not np.isclose(cg_charge, itp_charge, atol=1e-6):
				problems.append(f"{name} has charge {cg_charge:g} in the CGSmiles string but {itp_charge:g} in the ITP "
								"(the ITP charge is kept).")
		return problems

	def vs_constructions(self, cg_graph):
		'''
		The virtual sites of the ITP and the beads they are constructed from, as cg_graph nodes.
		The virtual site is the first atom of each [ virtual_sites* ] entry, the constructing atoms follow.
		virtual_sites1 entries are listed without constructing bead, since vermouth only keeps their first atom.

		Returns
		-------
		dict
			{virtual site: list of constructing beads}, in the order of the ITP.
		'''
		itp_to_cg = self.cg_node_map(cg_graph)
		constructions = {}
		for name, interactions in self.block.interactions.items():
			if not name.startswith('virtual_sites'):
				continue
			for interaction in interactions:
				site, *parents = (itp_to_cg[atom] for atom in interaction.atoms)
				known = constructions.setdefault(site, [])
				for parent in parents:
					if parent not in known:
						known.append(parent)
		return constructions

	def virtual_sites(self, cg_graph):
		'''
		The cg_graph nodes that are defined as virtual sites in the ITP.
		'''
		return set(self.vs_constructions(cg_graph))

	def interactions(self, cg_graph, interaction_type):
		'''
		Yield the cg_graph nodes of each interaction of the given type. Interactions between the same beads
		(in the same or reversed order, e.g. multiple dihedral terms) are only yielded once.
		'''
		itp_to_cg = self.cg_node_map(cg_graph)
		seen = set()
		for interaction in self.block.interactions.get(interaction_type, []):
			nodes = [itp_to_cg[atom] for atom in interaction.atoms]
			key = min(tuple(nodes), tuple(reversed(nodes)))
			if key in seen:
				continue
			seen.add(key)
			yield nodes

	def apply_cg_types(self, block, cg_graph):
		'''
		Set the atom types of block (the block of this topology or a copy of it) to the bead types of cg_graph.
		If the bead size changes and the atom has the default Martini mass of its original size, the mass is
		changed to the default of the new size. Changes are relative to the ITP as read.

		Returns
		-------
		dict
			{block node: comment describing the change}
		'''
		comments = {}
		itp_to_cg = self.cg_node_map(cg_graph)
		for node, d in block.nodes(data=True):
			old_type, old_mass = self.original[node]
			new_type = cg_graph.nodes[itp_to_cg[node]]["atype"]
			d['atype'] = new_type
			d['mass'] = old_mass
			if new_type == old_type:
				continue
			changes = [f"changed from {old_type} to {new_type}"]
			old_default, new_default = BeadType(old_type).default_mass, BeadType(new_type).default_mass
			if (old_default is not None and new_default is not None and old_default != new_default
					and old_mass is not None and np.isclose(old_mass, old_default)):
				d['mass'] = new_default
				changes.append(f"mass {old_mass:g} -> {d['mass']:g}")
			comments[node] = ', '.join(changes)
		return comments

	def write(self, path_out, cg_graph, outname=None, cgsmiles_string=None):
		'''
		Write the molecule type with the bead types of cg_graph. Changed atoms are commented.
		The topology itself is not modified.

		path_out is either the ITP file (ending in .itp; outname defaults to its stem) or a directory, which gets
		<outname>.itp (outname defaults to the molecule type). outname is the name of the written molecule type;
		if it differs from the molecule type and all atoms have the molecule type as residue name, the residue
		name is changed to outname as well.

		Returns
		-------
		Path
			The written file.
		'''
		path_out = Path(path_out)
		if path_out.suffix == '.itp':
			outname = outname or path_out.stem
		else:
			outname = outname or self.molname
			path_out = path_out / f'{outname}.itp'
		block = copy.deepcopy(self.block)
		comments = self.apply_cg_types(block, cg_graph)
		if outname != self.molname and all(d.get('resname') == self.molname for _, d in block.nodes(data=True)):
			for _, d in block.nodes(data=True):
				d['resname'] = outname
		path = itp_writer(outname, block, cgsmiles_string, path_out)
		# the atoms are numbered in the ITP in the order of the block, starting at 1
		itp_index = {node: i for i, node in enumerate(block.nodes, start=1)}
		add_comments_itp({itp_index[node]: comment for node, comment in comments.items()}, path)
		return path


def itp_writer(molname, block, cgsmiles_string=None, out_path='.'):
	'''
	Write a block as ITP file (adapted from fast_forward).

	Parameters
	----------
	molname : str
		Name of the molecule type.
	block : vermouth.molecule.Block
		The block to write.
	cgsmiles_string : str, optional
		CGSmiles string written into the header of the file.
	out_path : str or Path
		The ITP file (ending in .itp) or a directory, which gets <molname>.itp.

	Returns
	-------
	Path
		The path of the written file.
	'''
	from vermouth.gmx.itp import write_molecule_itp
	from vermouth.file_writer import DeferredFileWriter, deferred_open
	header = []
	if cgsmiles_string is not None:
		header = ['This file was generated using the following CGSmiles string:', cgsmiles_string, '\n']

	mol_out = block.to_molecule()
	mol_out.meta['molname'] = molname

	path = Path(out_path)
	if path.suffix != '.itp':
		path = path / f'{molname}.itp'
	with deferred_open(str(path), 'w') as fout:
		write_molecule_itp(mol_out, fout, moltype=molname, header=header)
	DeferredFileWriter().write()
	return path


def add_comments_itp(comments, itp_path):
	'''
	Append comments to lines of the [ atoms ] section of an ITP file.

	Parameters
	----------
	comments : dict
		{atom index as written in the ITP (starting at 1): comment}
	itp_path : str or Path
		The ITP file, modified in place.
	'''
	if not comments:
		return
	with open(itp_path) as f:
		lines = f.readlines()

	in_atoms = False
	for i, line in enumerate(lines):
		stripped = line.strip()
		if stripped.startswith('['):
			in_atoms = stripped.strip('[] \t') == 'atoms'
			continue
		if not in_atoms or not stripped[:1].isdigit(): # skip empty, comment and preprocessor lines
			continue
		index = int(stripped.split()[0])
		if index in comments:
			lines[i] = line.rstrip('\n') + f' ; {comments[index]}\n'

	with open(itp_path, 'w') as f:
		f.writelines(lines)
