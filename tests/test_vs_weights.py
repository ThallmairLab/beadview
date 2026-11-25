from beadview.beads import detect_virtual_nodes, vs_atom_weights
from beadview.mapping import CGMapping


def _weights(cgs):
    """{virtual site fragname: {element: sorted weights}} for every virtual site of `cgs`."""
    cg_graph, aa_graph = CGMapping._resolve(cgs)
    virtual = detect_virtual_nodes(cg_graph)
    out = {}
    for node in virtual:
        weights = vs_atom_weights(cg_graph, node, virtual)
        by_element = {}
        for atom, weight in weights.items():
            by_element.setdefault(aa_graph.nodes[atom]["element"], []).append(weight)
        out[cg_graph.nodes[node]["fragname"]] = {e: sorted(w) for e, w in by_element.items()}
    return out


def test_wvs_per_virtual_site():
    """wVS_<name> sets the weight for that virtual site only; wVS stays the default for the others
    (as in fast_forward)."""
    weights = _weights(r"{[#A][#C](.[#U]).[#V]}.{#A=C[$],#C=[$][N;wVS=0;wVS_V=1][C;wVS=1;wVS_V=0]}")
    assert weights["U"] == {"N": [0.0], "C": [1.0], "H": [0.0, 1.0, 1.0, 1.0]}
    assert weights["V"] == {"N": [1.0], "C": [0.0], "H": [0.0, 0.0, 0.0, 1.0]}


def test_a_hydrogens_own_wvs_wins_over_its_heavy_atoms():
    weights = _weights(r"{[#A][#C].[#V]}.{#A=C[$],#C=[$]N[C;wVS_V=0.5][H;wVS_V=0]}")
    assert weights["V"] == {"N": [1.0], "C": [0.5], "H": [0.0, 0.5, 0.5, 1.0]}
