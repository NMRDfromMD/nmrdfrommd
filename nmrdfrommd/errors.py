# nmrdfrommd/errors.py

NON_ORTHOGONAL_BOX = (
    "Non-orthogonal simulation box detected.\n\n"
    "NMRDfromMD currently only supports orthorhombic cells.\n"
    "You can convert your trajectory using:\n"
    "  - lipyphilic.triclinic_to_orthorhombic\n"
)

INVALID_TYPE_ANALYSIS = (
    "Invalid type_analysis='{value}'. "
    "Must be one of: inter_molecular, intra_molecular, full."
)