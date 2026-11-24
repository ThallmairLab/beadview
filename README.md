# cgsmiles_utils

Utilities for coarse-grained (Martini) mappings written as [CGSmiles](https://github.com/gruenewald-lab/CGsmiles) strings:

- **drawing**: the all-atom molecule with its beads as SVG, with bead labels and the bonds, angles and dihedrals of an ITP
- **structures**: all-atom and coarse-grained structure files (PDB, GRO, …) generated from the string
- **ITP editing**: write an ITP with the bead types of the string, with comments on every changed atom
- **bead type changes** directly in the CGSmiles string
- **pKa prediction** with pkasolver (optional)

## Installation

From a clone of this repository:

```bash
pip install ".[all]"        # or -e for an editable install
```

The core package (drawing to SVG, bead type changes) only needs cgsmiles, pysmiles, RDKit, numpy, networkx and contourpy.
The other features need these extras:

| extra       | enables                                   | installs                                   |
|-------------|-------------------------------------------|--------------------------------------------|
| `show`      | `show()` outside notebooks and on matplotlib axes, `save()` as PNG, `show_default()` | matplotlib, pillow, resvg-py, scipy |
| `structure` | `aa_structure()`, `cg_structure()`        | MDAnalysis                                 |
| `itp`       | `read_itp()`, `change_itp()`, `write_itp()` | vermouth                                 |
| `all`       | all of the above                          |                                            |
| `dev`       | `all` and the test suite                  | pytest, ipython                            |

All of these are plain pip wheels without system libraries.
`predict_pka()` needs [pkasolver](https://github.com/mayrf/pkasolver), which is not on PyPI; install it from its repository.

The package was renamed from `CGSmiles_utils` to `cgsmiles_utils`; update old imports to `cgsmiles_utils`.

## Quick start

```python
import matplotlib.pyplot as plt
from cgsmiles_utils import CGMapping, DrawOptions

mapping = CGMapping("{[#SC3;n=A][#SN4;n=B][#SP1;n=C]}.{#SC3=CC[$],#SN4=[$]CC[$],#SP1=[$]CO}")

mapping                                           # last line of a Jupyter/marimo cell: shows the drawing
mapping.show(show_bead_labels=True)               # vector image in Jupyter/marimo, else matplotlib
mapping.save("pentanol.svg", style="outline")     # SVG file
mapping.save("pentanol.png", dpi=600)             # PNG file, format from the extension (needs [show])
svg = mapping.to_svg(scale_factor=2)              # SVG string

opts = DrawOptions(show_hydrogens=True, show_bead_labels=True)
mapping.save("pentanol_h.svg", opts=opts, rotate_by=90)   # keyword arguments override opts

fig, axs = plt.subplots(1, 2)                     # several drawings in one figure (needs [show])
mapping.show(ax=axs[0])
mapping.show(ax=axs[1], style="outline")

mapping.aa_structure("pentanol_aa.pdb")           # needs [structure]
mapping.cg_structure("pentanol_cg.gro")           # also writes pentanol_aa.gro

mapping.change_bead_types({"A": "SN4"})           # new bead type for the bead named A
mapping.get_smiles()                              # 'CCCCCO'
```

### ITP files (needs `[itp]`)

```python
mapping = CGMapping(cgs_string)
mapping.read_itp("molecule.itp")                  # first molecule type, or molname="..."
mapping.show(show_bonds=True, show_angles=True, show_dihedrals=True)
mapping.show(show_vs_construction=True)           # which beads construct each virtual site
mapping.show(show_angles=True, highlight_beads=["R1"])  # only the angles that include R1

mapping.change_bead_types({"R2": "TN1"})
mapping.change_itp()                              # preview: {block node: 'changed from TC5 to TN1'}
mapping.write_itp("itps/MOL.itp")                 # molecule type (and residue) named MOL
mapping.cg_structure("MOL.gro", resname="MOL")    # beads in the atom order of the ITP
```

ITP atoms are matched to beads by name (`n=`). `read_itp` warns (`ITPConsistencyWarning`) about bead names that are
not unique or missing on one side and bead charges (`q=`) that differ from the ITP. Bonds, virtual sites and other
interactions are not compared with the string.

`write_itp` writes the bead types of the CGSmiles string, marks every changed atom with a comment and updates the
mass when the bead size changes (T/S/R = 36/54/72) and the mass was the default one. Charges are not changed.
The CGSmiles string is written into the header of the ITP; the loaded ITP itself is never modified.

`change_bead_types` edits the CGSmiles string and checks the result with cgsmiles: the new string must resolve to
the same atoms and beads, with the requested bead types.

#### Scanning bead types

`with_bead_types` returns a changed copy of a mapping that shares its ITP, so the ITP is read only once:

```python
base = CGMapping(cgs_string)
base.read_itp(itp_path, molname)
for name, bead_change in zip(["V1", "V2"], [{"N6": "TN2a"}, {"N6": "TN3a", "R3": "TC5e"}]):
    variant = base.with_bead_types(bead_change)
    variant.write_itp(f"itps/{name}.itp")                          # molecule type and residue name = name
    variant.cg_structure(f"solute/{name}.gro", resname=name)        # bead order from the ITP
```

## CGSmiles annotations used

| annotation | on    | meaning |
|------------|-------|---------|
| `n=`       | bead  | bead name, used for labels, structure files and matching ITP atoms |
| `index=`   | bead  | bead order in `cg_structure` without ITP; all beads need one (default: the numbers at the end of the bead names if all have one, else the order in the string) |
| `w=`       | atom  | mapping weight of the atom in the bead position (default 1) |
| `wVS=`     | atom  | weight of the atom in the position of a neighboring virtual site; a hydrogen without one uses the value of its heavy atom (default: `w`) |

A bead connected to the rest of the molecule only by bonds of order 0 is a virtual site. It is
placed at the weighted center of the atoms of its neighboring beads.

Bead types follow Martini 3: an optional size letter (`T`, `S`, `R` = regular), the chemical class (`C`, `N`, `P`,
`Q`, `X`, `D`; it sets the default drawing color) and its level, e.g. `TC5`, `SN4a`, `P1`. `U` marks a dummy bead.
Several fragments of the same type are told apart by a trailing uppercase letter (`#TC5A`), which is stripped from
the bead type.

## Drawing options

All drawing methods take `opts=DrawOptions(...)` and/or the fields of `DrawOptions` as keyword arguments.
Lengths are in pixels at `scale_factor=1`.

| option | default | description |
|--------|---------|-------------|
| `canvas_size` | `'auto'` | `'auto'` fits the image to the drawing; or a fixed `(width, height)`, e.g. to give several molecules the same scale |
| `padding` | `10` | space around the drawing with `canvas_size='auto'` |
| `scale_factor` | `1` | scales the molecule and all lengths |
| `rotate_by` | `0` | rotation in degrees |
| `flip` | `False` | mirror horizontally |
| `fixed_coords` | `None` | fixed 2D positions of atoms, `{aa_graph node: (x, y)}` in RDKit units (bond length 1.5), or a function `aa_graph -> dict` such as `parallel_tails()`; RDKit lays out the other atoms around them. Applied before `rotate_by` and `flip` |
| `remove_background` | `True` | transparent background |
| `show_hydrogens` | `False` | draw hydrogens |
| `color_atoms` | `False` | color heteroatoms by element |
| `show_atom_indices` | `False` | atom indices |
| `bond_width` | `1.5` | width of the atomistic bonds |
| `show_mapping` | `True` | draw the beads |
| `style` | `'beads'` | `'beads'`: circles; `'outline'`: rounded outlines along the bonds of each bead |
| `outline_width` | `12.0` | width of the outlines (`style='outline'`) |
| `border_width`, `border_color`, `border_alpha` | `1.5`, `'#000000'`, `0.5` | bead borders |
| `fill_alpha` | `0.5` | opacity of the bead filling |
| `bead_colors` | `None` | fill colors by bead type or chemical class, e.g. `{'TN1a': '#2ca02c', 'C': 'lightgray'}`; the type wins over the class |
| `bead_radii` | `None` | circle radii by bead type or size, e.g. `{'T': 12, 'SP1': 22}`; the type wins over the size |
| `include_hydrogens_in_bead_position` | `None` | `None` follows `show_hydrogens` |
| `indicate_vs` | `True` | dashed outline for beads that are virtual sites in the loaded ITP |
| `show_bead_labels` | `False` | bead type and name next to each bead |
| `bead_type_label_color`, `bead_name_label_color` | `'#666666'`, `'#000000'` | label colors |
| `bead_label_position` | `'optimize'` | `'optimize'` places labels with minimal overlap; or a function `(bead_position, bead_radius, bead_name) -> position` |
| `show_vs_construction` | `False` | dashed lines from each virtual site of the loaded ITP to the beads it is constructed from; the virtual site's border takes the same color, fully opaque |
| `show_bonds`, `show_angles`, `show_dihedrals` | `False` | interactions of the loaded ITP (dihedrals: proper and improper) |
| `highlight_beads` | `None` | bead names; only the interactions and constructions that include one of them are drawn |
| `vs_construction_colors`, `angle_colors`, `dihedral_colors` | `None` | one color per virtual site, angle or dihedral: a list (cycled), a single color for all, or `None` for the default palette of each type |
| `vs_construction_width` | `2.0` | width of the construction lines |
| `bond_color`, `constraint_color` | `'#000000'`, `'#020070FF'` | colors of the bond and constraint lines |
| `cg_line_width` | `2.0` | width of the interaction lines |
| `cg_line_offset` | `8.0` | distance of angle/dihedral lines from the bonds they follow |
| `line_spacing` | `4.0` | distance between overlapping angle/dihedral lines |
| `check_orientation` | `False` | report and highlight the most vertical bond |

### Layout: straight, parallel tails

By default RDKit chooses the 2D layout. `parallel_tails()` draws the tails of a molecule (unbranched carbon
chains of at least `min_length=6` carbons ending in a CH3, plus the carbonyl or branching carbon they hang on) as
straight zigzag chains pointing down, side by side `spacing=4.0` apart (RDKit units, bond length 1.5), in the order
of the atoms in the CGSmiles string. Cis double bonds keep their geometry as a sideways step in the chain.

```python
from cgsmiles_utils import parallel_tails

lipid.save("popc.svg", fixed_coords=parallel_tails())
lipid.save("popc_h.svg", fixed_coords=parallel_tails(spacing=6), show_hydrogens=True, rotate_by=180)
```

## Development

```bash
pip install -e ".[dev]"
pytest
```

The tests run on GitHub Actions for Python 3.11–3.13.
