from typing import Dict, Tuple, TypedDict, Union, List
import numpy as np
from openbabel import openbabel, pybel
from rdkit import Chem
#from scipy.sparse.csgraph import floyd_warshall
import os

# Add Biopython import
try:
    from Bio import PDB
    from Bio.PDB.Polypeptide import is_aa
    BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False

ATOMIC_NAME = {
    1: "H",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
    10: "Ne",
    15: "P",
    16: "S",
    17: "Cl",
    35: "Br"
}

ATOMIC_COLOR = {
    "H": "silver",
    "C": "black",
    "N": "blue",
    "O": "red",
    "F": "green",
    "P": "orange",
    "S": "yellow",
    "Cl": "limegreen",
    "Br": "salmon"
}

# covalent radii
ATOMIC_SIZE = {
    "H": 0.31,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "F": 0.57,
    "Ne": 0.58,
    "P": 1.07,
    "S": 1.05,
    "Cl": 1.02,
    "Br": 1.20
}

ATOMIC_PROPERTIES = {
    "H": {"en": 2.20, "vdw": 1.10, "cov": 0.31},   # Hydrogen (H)
    "C": {"en": 2.55, "vdw": 1.70, "cov": 0.76},   # Carbon (C)
    "N": {"en": 3.04, "vdw": 1.55, "cov": 0.71},   # Nitrogen (N)
    "O": {"en": 3.44, "vdw": 1.52, "cov": 0.66},   # Oxygen (O)
    "F": {"en": 3.98, "vdw": 1.47, "cov": 0.64},   # Fluorine (F)
    "P": {"en": 2.19, "vdw": 1.80, "cov": 1.06},  # Phosphorus (P)
    "S": {"en": 2.58, "vdw": 1.80, "cov": 1.02},  # Sulfur (S)
    "Cl": {"en": 3.16, "vdw": 1.75, "cov": 0.99},  # Chlorine (Cl)
    "Br": {"en": 2.96, "vdw": 1.85, "cov": 1.14},  # Bromine (Br)
}


def process_rdkit_mol(mol: Chem.rdchem.Mol, heavy_atoms_only: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """process a rdkit mol object. Assume it is a single molecule.
    """
    #if heavy_atoms_only:
    #    mol = Chem.RemoveHs(mol)
        
    n = len(mol.GetAtoms())
    X = np.empty([n, 3], dtype=float)
    T = np.empty([n], dtype=str)
    B = np.zeros([n, n], dtype=float)
    for i, atom in enumerate(mol.GetAtoms()):
        X[i, :] = np.array(mol.GetConformer().GetAtomPosition(i))
        T[i] = atom.GetSymbol()

    for bond in mol.GetBonds():
        B[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()] = 1
        B[bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()] = 1

    if heavy_atoms_only:
        # Create mask for non-hydrogen atoms
        heavy_mask = T != 'H'
            # Apply mask to all arrays
        X = X[heavy_mask]
        T = T[heavy_mask]
        B = B[heavy_mask, :][:, heavy_mask]
    return X, T, B


def process_molecule(mol: pybel.Molecule, heavy_atoms_only: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Process a molecule object read from an xyz file.

    Parameters
    ----------
    mol : openbabel.pybel.Molecule
        An OpenBabel molecule object loaded by pybel.
    heavy_atoms_only : bool, optional
        If True, only heavy atoms (non-hydrogen) will be included in the output.
        Default is False.

    Returns
    -------
    coordinates : numpy.ndarray
        Array of shape (n_atoms, 3) containing the 3D coordinates of each atom.
    atom_types : numpy.ndarray
        Array of shape (n_atoms,) containing the atomic labels for each atom.
    bond_matrix : numpy.ndarray
        Array of shape (n_atoms, n_atoms) containing the bond information,
        where 1 indicates a bond and 0 indicates no bond.

    Raises
    ------
    ValueError
        If the molecule object is empty or invalid.
    """
    if not mol or len(mol.atoms) == 0:
        raise ValueError("Invalid or empty molecule object provided")

    n = len(mol.atoms)
    X = np.empty([n, 3], dtype=float)
    T = np.empty([n], dtype=object)
    B = np.zeros([n, n], dtype=float)
    
    for i in range(n):
        X[i, :] = mol.atoms[i].coords
        T[i] = ATOMIC_NAME[mol.atoms[i].atomicnum]
    
    obmol = mol.OBMol
    for bond in openbabel.OBMolBondIter(obmol):
        atom1 = bond.GetBeginAtom()
        atom2 = bond.GetEndAtom()
        B[atom1.GetIdx()-1, atom2.GetIdx()-1] = 1
        B[atom2.GetIdx()-1, atom1.GetIdx()-1] = 1
        # TODO: bond order, atom connectivity
    
    if heavy_atoms_only:
        # Create mask for non-hydrogen atoms
        heavy_mask = T != 'H'
        # Apply mask to all arrays
        X = X[heavy_mask]
        T = T[heavy_mask]
        B = B[heavy_mask, :][:, heavy_mask]
        
    return X, T, B


def parse_sy2(file_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Parse a sy2 file to extract coordinates and atom types.

    Parameters
    ----------
    file_path : str
        Path to the sy2 file.

    Returns
    -------
    coordinates : numpy.ndarray
        Array of shape (n_atoms, 3) containing the 3D coordinates of each atom.
    atom_types : numpy.ndarray
        Array of shape (n_atoms,) containing the atom types as specified in the Mol2 file.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    atom_types = []
    coordinates = []

    with open(file_path, 'r') as file:
        lines = file.readlines()
        atom_section = False

        for line in lines:
            line = line.strip()
            # Detect the start of the ATOM section
            if line == "@<TRIPOS>ATOM":
                atom_section = True
                continue
            # Detect the end of the ATOM section
            if line == "@<TRIPOS>BOND":
                break
            # Process lines in the ATOM section
            if atom_section:
                parts = line.split()
                if len(parts) >= 8:
                    atom_types.append(parts[5])  # SYBYL atom type is in the 6th column
                    coordinates.append([float(parts[2]), float(parts[3]), float(parts[4])])  # x, y, z coordinates

    return np.array(coordinates), np.array(atom_types)


def parse_mna(file_path: str) -> np.ndarray:
    """Parses an .mna file to extract atom connectivity.

    Parameters
    ----------
    file_path : str
        Path to the mna file.

    Returns
    -------
    numpy.ndarray
        Array of shape (n_atoms,) containing the atom connectivity.
    """
    atom_connectivity = []

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line and not line.startswith("#"):  # Ignore empty lines and comments
                atom_connectivity.append(line)

    return np.array(atom_connectivity, dtype=str)


def write_xyz_with_custom_labels(
        output_file: str, 
        coordinates: np.ndarray, 
        labels: np.ndarray,
        comment: str = "Modified xyz file with custom names. Used for ArbAlign."):
    """Write xyz file with coordinates and custom names.
    Intended for preparing input for ArbAlign.

    Parameters
    ----------
    output_file : str
        Path to the output xyz file.
    coordinates : np.ndarray
        Array of shape (n_atoms, 3) containing the 3D coordinates of each atom.
    connectivity : np.ndarray
        Array of shape (n_atoms,) containing the SYBYL type or atom connectivity.
    """
    with open(output_file, 'w') as f:
        f.write(f"{coordinates.shape[0]}\n")
        f.write(f"{comment}\n")
        for coord, atom_name in zip(coordinates, labels):
            f.write(f"{atom_name:4s} {coord[0]:12.6f} {coord[1]:12.6f} {coord[2]:12.6f}\n")


def parse_pdb_file(file_path: str) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """
    Parse a PDB file and extract coordinates, element names, and adjacency matrix.
    
    Args:
        file_path (str): Path to the PDB file
        
    Returns:
        Tuple[np.ndarray, List[str], np.ndarray]: 
            - X: coordinates array (N, 3) where N is number of atoms
            - T: list of element names (N,)
            - B: adjacency matrix (N, N) where B[i,j] = 1 if atoms i and j are bonded
    """
    
    # Initialize data structures
    atoms = []  # List to store atom information
    bonds = []  # List to store bond information
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Parse ATOM/HETATM records
            if line.startswith(('ATOM', 'HETATM')):
                atom_info = parse_atom_line(line)
                if atom_info:
                    atoms.append(atom_info)
            
            # Parse CONECT records
            elif line.startswith('CONECT'):
                bond_info = parse_conect_line(line)
                if bond_info:
                    bonds.append(bond_info)
    
    # Convert to numpy arrays
    if not atoms:
        raise ValueError("No atoms found in PDB file")
    
    # Extract coordinates and element names
    X = np.array([atom['coords'] for atom in atoms])
    T = np.array([atom['element'] for atom in atoms])
    
    # Create adjacency matrix
    num_atoms = len(atoms)
    B = np.zeros((num_atoms, num_atoms), dtype=int)
    
    # Fill adjacency matrix based on CONECT records
    for bond_info in bonds:
        central_atom = bond_info[0]         
        for connected_atom in bond_info[1:]:
            B[central_atom-1, connected_atom-1] = 1
            B[connected_atom-1, central_atom-1] = 1  # Symmetric
    
    heavy_mask = [True if T[i] not in ['H', 'D'] else False for i in range(len(T))]
    X = X[heavy_mask]
    T = T[heavy_mask]
    B = B[heavy_mask, :][:, heavy_mask]

    return X, T, B

def parse_atom_line(line: str) -> dict:
    """
    Parse an ATOM or HETATM line from PDB format.
    
    Args:
        line (str): ATOM or HETATM line
        
    Returns:
        dict: Dictionary containing atom information
    """
    # Extract atom serial number (1-indexed)
    atom_serial = int(line[6:11].strip())
        
    # Extract coordinates (columns 30-54)
    x = float(line[30:38].strip())
    y = float(line[38:46].strip())
    z = float(line[46:54].strip())
        
    # Extract element name (columns 76-78)
    element = line[76:78].strip()
        
    return {
        'serial': atom_serial,
        'coords': [x, y, z],
        'element': element
    }

def parse_conect_line(line: str) -> List[int]:
    """
    Parse a CONECT line from PDB format.
    
    Args:
        line (str): CONECT line
        
    Returns:
        List[int]: List of atoms [central_atom, connected_atom1, connected_atom2, ...]
    """
    # CONECT format: CONECT, central_atom, connected_atom1, connected_atom2, ...
    # Example: CONECT    1    2   52
    parts = line.split()
        
    if len(parts) < 3:
        return None
        
    # Convert all parts to integers, skipping the first part ("CONECT")
    atoms = [int(atom) for atom in parts[1:] if atom.isdigit()]
        
    return atoms
