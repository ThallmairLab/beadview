from fast_forward.itp_parser_sub import read_itp
import cgsmiles

from vermouth.gmx.itp import write_molecule_itp
from vermouth.file_writer import DeferredFileWriter, deferred_open
from vermouth.forcefield import ForceField

import string 
import sys
sys.path.append("/mnt/plutus/tduve/projects/CG_Photoswitches/CGsmiles/script")
from cgsmiles_coordinates import generate_CG_gro

def clean_cg_graph(cg_graph):
	for n, data in cg_graph.nodes(data=True):
		if data["fragname"][-1].isupper() and len(data["fragname"]) > 1: # remove trailing letters from fragname except for virtual sites
			data["fragname"] = data["fragname"][:-1]
		# add property atype = fragname
		data["atype"] = data["fragname"]
		data["atomname"] = data["n"]

def change_atype(cg_graph, ff, molname):
	comments = {}
	for node, d in ff.blocks[molname].nodes(data=True):
		cg_node = next((n for n, attrs in cg_graph.nodes(data=True) if attrs["atomname"] == d["atomname"]), None)
		if cg_node is None:
			raise ValueError("No matching node found in cg_graph for node {} in ff.blocks".format(node))
		if d['atype'] != cg_graph.nodes[cg_node]["atype"]:
			comments[node] = f"changed from {cg_graph.nodes[cg_node]['atype']} to {d['atype']}"
			d['atype'] = cg_graph.nodes[cg_node]["atype"]
	return comments
		
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
		
def add_comments(comments, itp_path):
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
	

def generate_itp_gro(path_itp, cgs_string, molname, outname=None, out_path_itp='.', out_path_gro='.'):
	cg_graph, _ = cgsmiles.MoleculeResolver.from_string(cgs_string).resolve()
	clean_cg_graph(cg_graph)

	with open(f'{path_itp}') as f:
		lines = f.readlines()

	ff = ForceField("dummy")
	read_itp(lines, ff)

	comments = change_atype(cg_graph, ff, molname)
	outname = outname if outname else molname

	itp_writer(outname, ff.blocks[molname], cgs_string, out_path_itp)

	DeferredFileWriter().write()

	add_comments(comments, f'{out_path_itp}/{outname}.itp')
	generate_CG_gro(cgs_string, outname, path=out_path_gro)

def change_cgs_string(cgs_string, bead_change):
	'''
	bead_change is a dictionary with the format {name: new_type}
	'''
	cg = cgs_string.split('}.{')[0][1:]
	aa = cgs_string.split('}.{')[1][:-1]

	# split the cg part into its components
	cg = [part.split('[') for part in cg.split(']')]
	cg = [item for sublist in cg for item in sublist if item]
	indices = [i for i, part in enumerate(cg) if part.startswith('#')]
	types = types_initial = [cg[i].split(';n=')[0][1:] for i in indices]
	names = [cg[i].split(';n=')[1] for i in indices]

	# check if names are unique
	if len(names) != len(set(names)):
		raise ValueError("Names are not unique")

	non_unique_count = 0
	letters = list(reversed(string.ascii_uppercase))
	# change types according to the bead_change
	new_aa_parts = []
	for name, type_ in bead_change.items():
		idx = names.index(name)
		if type_ != types[idx]: # change type
			new_type = f'{type_}{letters[non_unique_count]}' if type_ in types_initial else type_ # add letter if the type is already present in the cg part
			for part in aa.split(','):
				if part.startswith(f'#{types[idx]}'):
					new_aa_parts.append(part.replace(f'#{types[idx]}', f'#{new_type}')) # copy the part with the new type 
			types[idx] = new_type
			non_unique_count += 1 if type_ in types else 0
				
	aa = ','.join(new_aa_parts) + ',' + aa if new_aa_parts else aa # add the new parts to the aa part

	# assembling the cg string again
	for i, idx in enumerate(indices):
		cg[idx] = f'[#{types[i]};n={names[i]}]'
	cg_string = ''.join(cg)
	new_cgs_string = f'{{{cg_string}}}.{{{aa}}}'

	return new_cgs_string