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
| `show`      | `show()`, `show_default()`                | matplotlib, pillow, cairosvg, scipy        |
| `structure` | `aa_structure()`, `cg_structure()`        | MDAnalysis                                 |
| `itp`       | `read_itp()`, `change_itp()`, `write_itp()` | vermouth                                 |
| `all`       | all of the above                          |                                            |
| `dev`       | `all` and the test suite                  | pytest, ipython                            |

cairosvg needs the system cairo library (`libcairo2` on Debian/Ubuntu, `cairo` with conda or Homebrew).
`predict_pka()` needs [pkasolver](https://github.com/mayrf/pkasolver), which is not on PyPI; install it from its repository.

The package was renamed from `CGSmiles_utils` to `cgsmiles_utils`; update old imports to `cgsmiles_utils`.

## Quick start

```python
from cgsmiles_utils import CGMapping, DrawOptions

mapping = CGMapping("{[#SC3;n=A][#SN4;n=B][#SP1;n=C]}.{#SC3=CC[$],#SN4=[$]CC[$],#SP1=[$]CO}")

mapping.show(show_bead_labels=True)               # in a notebook (needs [show])
mapping.save("pentanol.svg", style="outline")     # SVG file
svg = mapping.to_svg(scale_factor=2)              # SVG string

opts = DrawOptions(show_hydrogens=True, show_bead_labels=True)
mapping.save("pentanol_h.svg", opts=opts, rotate_by=90)   # keyword arguments override opts

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

mapping.change_bead_types({"R2": "TN1"})
mapping.change_itp()                              # {block node: 'changed from TC5 to TN1'}
mapping.write_itp("out_dir", "MOL")               # out_dir/MOL.itp
```

`write_itp` writes the bead types of the CGSmiles string, marks every changed atom with a comment and updates the
mass when the bead size changes (T/S/R = 36/54/72) and the mass was the default one.
The CGSmiles string is written into the header of the ITP. ITP atoms are matched to beads by name (`n=`), so
bead names have to be unique.

## CGSmiles annotations used

| annotation | on    | meaning |
|------------|-------|---------|
| `n=`       | bead  | bead name, used for labels, structure files and matching ITP atoms |
| `index=`   | bead  | bead order in `cg_structure` (default: the number in the bead name, then the order in the string) |
| `w=`       | atom  | mapping weight of the atom in the bead position (default 1) |
| `wVS=`     | atom  | weight of the atom in the position of a neighboring virtual site; a hydrogen without one uses the value of its heavy atom (default: `w`) |

A bead connected to the rest of the molecule only by bonds of order 0 is a virtual site. It is
placed at the weighted center of the atoms of its neighboring beads.

Bead types follow Martini 3: an optional size letter (`T`, `S`, `R` = regular), the chemical class (`C`, `N`, `P`,
`Q`, `X`, `D`; it sets the drawing color) and its level, e.g. `TC5`, `SN4a`, `P1`. `U` marks a dummy bead.
Several fragments of the same type are told apart by a trailing uppercase letter (`#TC5A`), which is stripped from
the bead type.

## Drawing options

All drawing methods take `opts=DrawOptions(...)` and/or the fields of `DrawOptions` as keyword arguments.
Lengths are in pixels at `scale_factor=1`.

| option | default | description |
|--------|---------|-------------|
| `canvas_size` | `(300, 300)` | size of the image |
| `scale_factor` | `1` | scales the molecule and all lengths |
| `rotate_by` | `0` | rotation in degrees |
| `flip` | `False` | mirror horizontally |
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
| `include_hydrogens_in_bead_position` | `None` | `None` follows `show_hydrogens` |
| `show_vs` | `True` | draw virtual sites |
| `indicate_vs` | `True` | dashed outline for beads that are virtual sites in the loaded ITP |
| `show_bead_labels` | `False` | bead type and name next to each bead |
| `bead_type_label_color`, `bead_name_label_color` | `'#666666'`, `'#000000'` | label colors |
| `bead_label_position` | `'optimize'` | `'optimize'` places labels with minimal overlap; or a function `(bead_position, bead_radius, bead_name) -> position` |
| `show_bonds`, `show_angles`, `show_dihedrals` | `False` | interactions of the loaded ITP |
| `bond_color`, `constraint_color`, `angle_color`, `dihedral_color` | | colors of the interaction lines |
| `cg_line_width` | `2.0` | width of the interaction lines |
| `cg_line_offset` | `8.0` | distance of angle/dihedral lines from the bonds they follow |
| `line_spacing` | `4.0` | distance between overlapping angle/dihedral lines |
| `check_orientation` | `False` | report and highlight the most vertical bond |

## Development

```bash
pip install -e ".[dev]"
pytest
```

The tests run on GitHub Actions for Python 3.11–3.13.
