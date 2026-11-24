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

	def virtual_sites(self, cg_graph):
		'''
		The cg_graph nodes that are defined as virtual sites in the ITP.
		'''
		itp_to_cg = self.cg_node_map(cg_graph)
		return {itp_to_cg[interaction.atoms[0]] # the virtual site is the first atom of each entry
				for name, interactions in self.block.interactions.items() if name.startswith('virtual_sites')
				for interaction in interactions}

	def interactions(self, cg_graph, interaction_type, hidden=()):
		'''
		Yield the cg_graph nodes of each interaction of the given type, skipping interactions that involve
		hidden nodes. Interactions between the same beads (in the same or reversed order, e.g. multiple
		dihedral terms) are only yielded once.
		'''
		itp_to_cg = self.cg_node_map(cg_graph)
		seen = set()
		for interaction in self.block.interactions.get(interaction_type, []):
			nodes = [itp_to_cg[atom] for atom in interaction.atoms]
			key = min(tuple(nodes), tuple(reversed(nodes)))
			if key in seen or any(node in hidden for node in nodes):
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
		Write the molecule type with the bead types of cg_graph to <path_out>/<outname>.itp.
		Changed atoms are commented. The topology itself is not modified.

		Returns
		-------
		Path
			The written file.
		'''
		block = copy.deepcopy(self.block)
		comments = self.apply_cg_types(block, cg_graph)
		path = itp_writer(outname or self.molname, block, cgsmiles_string, path_out)
		# the atoms are numbered in the ITP in the order of the block, starting at 1
		itp_index = {node: i for i, node in enumerate(block.nodes, start=1)}
		add_comments_itp({itp_index[node]: comment for node, comment in comments.items()}, path)
		return path


def itp_writer(molname, block, cgsmiles_string=None, out_path='.'):
	'''
	Write a block as ITP file <out_path>/<molname>.itp (adapted from fast_forward).

	Parameters
	----------
	molname : str
		Name of the molecule type, also used as file name.
	block : vermouth.molecule.Block
		The block to write.
	cgsmiles_string : str, optional
		CGSmiles string written into the header of the file.
	out_path : str or Path
		Directory to write the file to.

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

	path = Path(out_path) / f'{molname}.itp'
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
