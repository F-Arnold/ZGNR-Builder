# ZGNR-Builder
A tool to build edge-patterned zigzag graphene nanoribbons (ZGNR) with a given geometry. 

## Nomenclature of systems

The script can create three types of systems: 
1) ZGNRs without edge patterning, defined by their width `N`.
2) Cove-edged ZGNRs (ZGNR-Cs), where single carbon atoms are removed periodically from the edges of a ZGNR. They are defined by the width `N` of the underlying ZGNR, the distance between two cove edges on the same edge `a`, and the offset between the cove edges on the top and bottom edge of the ribbon `b`. For a more in-depth description of the structural parameters, please check out Fig. 1 in [this publication](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.129.216401) ([arXiv version](https://doi.org/10.48550/arXiv.2205.15811)).
3) Gulf-edged ZGNRs (ZGNR-Gs), where instead of a single carbon atom, multiple adjacent carbon atoms are removed on the edges. This introduces an additional parameter, `M`, with the number of removed carbon atoms per gulf edge given by 2`M`-1. A figure explaining this structure type will be added once the related manuscript is published on arXiv.

## Usage

The script is written in Python with a command-line interface. Below is the list of parameters that can be set this way. This can also be accessed using the `--help` option when running the script.

| Command | Parameters | Explanation |
|---------|------------|--------------|
| `-N` | `[width]` | Width of N-ZGNR backbone |
| `-C` | `[a] [b]` | Parameters for N-ZGNR-C(a,b) |
| `-G` | `[a] [b] [M]` | Parameters for N-ZGNR-G_M(a,b) |
| `-R` | `[units]` | Number of primitive unit cells in supercell of ZGNR-C/G, also applicable when creating heterojunctions (default: R=1) |
| `-finite` | None | Set structure type to finite molecule instead or periodic (default: periodic) |
| `-oop` | `[num_oop]` | Activate random oop displacement, give number of structures `num_oop` to create (default: False) |
| `-make1d` | None | Deactivate VEC2 and VEC3 lines for .xyz file (default: False -> 3D system) | 
| `-saturate` | None | Toggle saturation of dangling bonds with H (default: False -> unsaturated) |
| `-noprint` | None | Add this command to suppress console outputs except for error messages (default: False) |
