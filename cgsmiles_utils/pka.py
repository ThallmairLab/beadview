'''
pKa prediction with pkasolver (optional dependency, only imported when used).
'''
import logging
import os
import warnings

from rdkit.Chem import SanitizeMol, rdDepictor

from .rdkit_conv import cgsmiles_to_rdkit


class suppress_output:
	'''Redirect stdout and stderr, including output of C extensions, to devnull.'''
	def __enter__(self):
		self._null_fd = os.open(os.devnull, os.O_RDWR)
		self._save_stdout = os.dup(1)
		self._save_stderr = os.dup(2)
		os.dup2(self._null_fd, 1)
		os.dup2(self._null_fd, 2)

	def __exit__(self, exc_type, exc_value, traceback):
		os.dup2(self._save_stdout, 1)
		os.dup2(self._save_stderr, 2)
		os.close(self._null_fd)
		os.close(self._save_stdout)
		os.close(self._save_stderr)


def predict_pka(aa_graph, show_protonation_reactions=False, show_protonation_map=True, print_pka_values=False):
	'''
	Predict the pKa values of a molecule with pkasolver. See CGMapping.predict_pka.
	'''
	try:
		from pkasolver.query import draw_pka_map, draw_pka_reactions, calculate_microstate_pka_values
	except ImportError:
		raise ImportError("pKa prediction needs pkasolver, install it with 'pip install pkasolver'.") from None

	mol, _ = cgsmiles_to_rdkit(aa_graph, add_hydrogens=False)
	SanitizeMol(mol)
	rdDepictor.Compute2DCoords(mol)

	# silence pkasolver (warnings, logging and output of the underlying libraries) only during the prediction
	previous_logging_level = logging.root.manager.disable
	try:
		with warnings.catch_warnings(), suppress_output():
			warnings.simplefilter("ignore", category=FutureWarning)
			logging.disable(logging.CRITICAL)
			pka_results = calculate_microstate_pka_values(mol)
	finally:
		logging.disable(previous_logging_level)

	if show_protonation_reactions or show_protonation_map:
		from IPython.display import display
		from rdkit.Chem.Draw import IPythonConsole
		previous_size = IPythonConsole.molSize
		IPythonConsole.molSize = 400, 400
		try:
			if show_protonation_reactions:
				print('Drawing pKa reaction network...')
				display(draw_pka_reactions(pka_results))
			if show_protonation_map:
				print('Drawing pKa map...')
				display(draw_pka_map(pka_results))
		finally:
			IPythonConsole.molSize = previous_size

	if print_pka_values:
		for state in pka_results:
			print(f'pKa = {state.pka}')
	return pka_results
