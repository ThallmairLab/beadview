from vermouth.gmx.itp import write_molecule_itp
from vermouth.file_writer import DeferredFileWriter, deferred_open

def itp_writer(molname, block, cgsmiles_string=None, out_path='.'):
	'''
	adapted from fast_forward

	'''

	if cgsmiles_string is not None:
		header = ['This file was generated using the following CGSmiles string:',
				cgsmiles_string, '\n',
				]

	mol_out = block.to_molecule()

	mol_out.meta['molname'] = molname
	mol_out.nodes[0]['comment'] = 'comment'
	
	with deferred_open(f'{out_path}/{molname}.itp', 'w') as fout:
		write_molecule_itp(mol_out, fout, moltype=molname, header=header)
	DeferredFileWriter().write()

def add_comments_itp(comments, itp_path):
	with open(itp_path, 'r') as f:
		lines = f.readlines()

	for i, line in enumerate(lines):
		if line.startswith('[ atoms ]'):
			start=i
		if line.startswith('[ bonds ]'):
			end=i
			break

	new_lines = lines[:start+1]
	for i, line in enumerate(lines[start+1:end]):
		if line.startswith('['):
			break
		if line.strip() and line.strip()[0].isdigit():
			node_num = int(line.strip().split()[0])
			if node_num in comments:
				line = line.split('\n')[0] + f" ; {comments[node_num]}\n"
		new_lines.append(line)
	new_lines.extend(lines[end:])
	with open(itp_path, 'w') as f:
		f.writelines(new_lines)