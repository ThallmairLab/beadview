from pathlib import Path

from vermouth.gmx.itp import write_molecule_itp
from vermouth.file_writer import DeferredFileWriter, deferred_open

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
