import copy
from dataclasses import replace
from pathlib import Path

import cgsmiles
from rdkit.Chem import MolToSmiles

from . import cgsmiles_string, pka, structure
from .beads import fragname_to_type
from .drawing.options import DrawOptions
from .drawing.render import Renderer, show_cgsmiles_drawing, show_svg, svg_to_png
from .rdkit_conv import cgsmiles_to_rdkit
from .topology import Topology

# image formats save() rejects instead of writing <name>.<format>.svg
UNSUPPORTED_IMAGE_FORMATS = {'.pdf', '.eps', '.ps', '.jpg', '.jpeg', '.tif', '.tiff', '.gif', '.webp', '.bmp'}


class CGMapping:
	'''
	A coarse-grained mapping of a molecule, given as CGSmiles string with two resolutions.

	Attributes
	----------
	cgs_string : str
		The CGSmiles string.
	cg_graph, aa_graph : networkx.Graph
		The coarse-grained and the all-atom graph resolved from the string. Each cg_graph node has
		'atype' (bead type, the fragment name without uniqueness suffix) and 'atomname' (n= annotation,
		else the bead type).
	topology : Topology or None
		The ITP loaded with read_itp().
	'''
	def __init__(self, cgs_string: str):
		self.topology = None
		self._set_string(cgs_string)

	@staticmethod
	def _resolve(cgs_string):
		'''(cg_graph, aa_graph) of a CGSmiles string, as resolved by cgsmiles.'''
		return cgsmiles.MoleculeResolver.from_string(cgs_string).resolve()

	def _set_string(self, cgs_string, graphs=None):
		self.cgs_string = cgs_string
		self.cg_graph, self.aa_graph = graphs if graphs is not None else self._resolve(cgs_string)
		for _, data in self.cg_graph.nodes(data=True):
			data["fragname"] = fragname_to_type(data["fragname"])
			data["atype"] = data["fragname"]
			data["atomname"] = data.get("n", data["fragname"])

	def get_smiles(self, add_hydrogens=False):
		'''SMILES string of the all-atom molecule.'''
		mol, _ = cgsmiles_to_rdkit(self.aa_graph, add_hydrogens=add_hydrogens)
		return MolToSmiles(mol, isomericSmiles=True)

	# drawing
	@staticmethod
	def _options(kwargs):
		'''DrawOptions from opts=DrawOptions(...) and/or individual options as keyword arguments.'''
		return replace(kwargs.pop("opts", None) or DrawOptions(), **kwargs)

	def to_svg(self, **kwargs) -> str:
		'''
		Draw the molecule with its mapping as SVG string. Takes opts=DrawOptions(...) and/or the
		fields of DrawOptions as keyword arguments, which override the ones in opts.
		'''
		return Renderer(self, self._options(kwargs)).svg()

	def save(self, path, dpi=300, **kwargs):
		'''
		Save the drawing (see to_svg, which takes the drawing options) as SVG or PNG, chosen by the extension of path.
		Without .svg or .png extension, .svg is appended. dpi sets the resolution of PNGs (SVG pixels are 1/96 inch);
		PNGs need resvg-py (the [show] extra).
		Returns the path written.
		'''
		path = Path(path)
		suffix = path.suffix.lower()
		if suffix in UNSUPPORTED_IMAGE_FORMATS:
			raise ValueError(f"Cannot save as {suffix[1:]}, use a .svg or .png file.")
		if suffix not in ('.svg', '.png'):
			path, suffix = path.with_name(path.name + '.svg'), '.svg'
		svg_text = self.to_svg(**kwargs)
		if suffix == '.png':
			path.write_bytes(svg_to_png(svg_text, zoom=dpi / 96))
		else:
			path.write_text(svg_text)
		return path

	def show(self, ax=None, **kwargs):
		'''
		Show the drawing (see to_svg). With ax, it is drawn on that matplotlib axis (e.g. a subplot). Without,
		it is displayed as vector image in a notebook (Jupyter or marimo), else on a new matplotlib axis.
		Returns the matplotlib axis, or None when displayed in a notebook.
		'''
		return show_svg(self.to_svg(**kwargs), ax)

	def _repr_svg_(self):
		'''Display the mapping with the default drawing options in Jupyter and marimo.'''
		return self.to_svg()

	def show_default(self, ax=None, **kwargs):
		'''Draw the all-atom molecule with cgsmiles' own drawing function (only show_hydrogens and scale_factor are used).'''
		return show_cgsmiles_drawing(self.aa_graph, self._options(kwargs), ax)

	# structures
	def aa_structure(self, out_file="molecule_aa.pdb", resname=None, random_seed=42):
		'''
		Generate and save an all-atom structure file (format from the extension, default pdb).
		resname defaults to the first 5 characters of the file name. random_seed makes the conformer
		reproducible; None gives a different one on every call.
		'''
		return structure.aa_structure(self, out_file, resname, random_seed)

	def cg_structure(self, out_file="molecule_cg.pdb", resname=None, write_aa=True, random_seed=42):
		'''
		Generate and save a coarse-grained structure file (format from the extension, default pdb).
		Bead positions are the centers of the atoms they map to; virtual sites are placed
		at the weighted center of the atoms of the beads they are connected to (weights from wVS, else w).
		If write_aa, the matching all-atom structure is written next to it (``_cg`` in the file name
		replaced by ``_aa``, or ``_aa`` appended). random_seed makes the conformer reproducible;
		None gives a different one on every call.
		'''
		return structure.cg_structure(self, out_file, resname, write_aa, random_seed)

	# bead types and topology
	def change_bead_types(self, bead_change):
		'''
		Change the types of beads, selected by their name (n= annotation), see
		cgsmiles_string.change_bead_types.

		Parameters
		----------
		bead_change : dict
			{bead name: new bead type}

		Returns
		-------
		str
			The new CGSmiles string, also stored in self.cgs_string.

		The new string is resolved with cgsmiles and compared with the old one: the all-atom molecule, the beads
		and their atoms must be unchanged and the beads must have the requested types, else a ValueError is
		raised and the mapping is not changed.
		'''
		new_string = cgsmiles_string.change_bead_types(self.cgs_string, bead_change)
		new_graphs = self._resolve(new_string)
		cgsmiles_string.check_bead_type_change((self.cg_graph, self.aa_graph), new_graphs, bead_change)
		self._set_string(new_string, new_graphs)
		return self.cgs_string

	def with_bead_types(self, bead_change):
		'''
		A new CGMapping with the types of some beads changed (see change_bead_types); this mapping is not changed.
		The new mapping shares the loaded ITP, which is never modified, so it is read only once for a scan of
		bead types.
		'''
		variant = copy.copy(self)
		variant.change_bead_types(bead_change) # replaces the string and graphs of the copy only
		return variant

	def read_itp(self, path_in, molname=None):
		'''Load an ITP file; molname selects the molecule type, by default the first one.'''
		self.topology = Topology.from_itp(path_in, molname)
		return self.topology

	def _require_topology(self):
		if self.topology is None:
			raise RuntimeError("No ITP loaded, call read_itp() first.")
		return self.topology

	def change_itp(self):
		'''
		The changes write_itp() makes to the loaded ITP to match the bead types of the CGSmiles string
		(see Topology.apply_cg_types). The loaded ITP is not modified.

		Returns
		-------
		dict
			{ITP block node: comment describing the change}
		'''
		topology = self._require_topology()
		return topology.apply_cg_types(copy.deepcopy(topology.block), self.cg_graph)

	def write_itp(self, path_out, outname=None):
		'''
		Write the loaded ITP with the bead types of the CGSmiles string to <path_out>/<outname>.itp
		(outname defaults to the molecule type). Changed atoms are commented. The loaded ITP itself is not modified.
		'''
		return self._require_topology().write(path_out, self.cg_graph, outname, self.cgs_string)

	# pKa prediction
	def predict_pka(self, show_protonation_reactions=False, show_protonation_map=True, print_pka_values=False):
		'''
		Predict the pKa values of the molecule with pkasolver (optional dependency).

		Parameters
		----------
		show_protonation_reactions : bool
			Display the network of protonation reactions.
		show_protonation_map : bool
			Display the pKa map.
		print_pka_values : bool
			Print the pKa values.

		Returns
		-------
		list
			The protonation states with their predicted pKa values.
		'''
		return pka.predict_pka(self.aa_graph, show_protonation_reactions, show_protonation_map, print_pka_values)
