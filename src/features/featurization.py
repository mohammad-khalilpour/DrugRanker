from typing import List, Tuple, Union

import numpy as np
import torch

from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.Draw import SimilarityMaps

# Atom feature sizes
MAX_ATOMIC_NUM = 100
ATOM_FEATURES = {
	'atomic_num': list(range(MAX_ATOMIC_NUM)),
	'degree': [0, 1, 2, 3, 4, 5],
	'formal_charge': [-1, -2, 1, 2, 0],
	'chiral_tag': [0, 1, 2, 3],
	'num_Hs': [0, 1, 2, 3, 4],
	'hybridization': [
		Chem.rdchem.HybridizationType.SP,
		Chem.rdchem.HybridizationType.SP2,
		Chem.rdchem.HybridizationType.SP3,
		Chem.rdchem.HybridizationType.SP3D,
		Chem.rdchem.HybridizationType.SP3D2
	],
}

# Distance feature sizes
PATH_DISTANCE_BINS = list(range(10))
THREE_D_DISTANCE_MAX = 20
THREE_D_DISTANCE_STEP = 1
THREE_D_DISTANCE_BINS = list(range(0, THREE_D_DISTANCE_MAX + 1, THREE_D_DISTANCE_STEP))

# len(choices) + 1 to include room for uncommon values; + 2 at end for IsAromatic and mass
ATOM_FDIM = sum(len(choices) + 1 for choices in ATOM_FEATURES.values()) + 2
BOND_FDIM = 14


def get_atom_fdim() -> int:
	"""Gets the dimensionality of atom features."""
	return ATOM_FDIM


def get_bond_fdim(atom_messages: bool = False) -> int:
	"""
	Gets the dimensionality of bond features.

	:param atom_messages whether atom messages are being used. If atom messages, only contains bond features.
	Otherwise contains both atom and bond features.
	:return: The dimensionality of bond features.
	"""
	return BOND_FDIM + (not atom_messages) * get_atom_fdim()


def onek_encoding_unk(value: int, choices: List[int]) -> List[int]:
	"""
	Creates a one-hot encoding.

	:param value: The value for which the encoding should be one.
	:param choices: A list of possible values.
	:return: A one-hot encoding of the value in a list of length len(choices) + 1.
	If value is not in the list of choices, then the final element in the encoding is 1.
	"""
	encoding = [0] * (len(choices) + 1)
	index = choices.index(value) if value in choices else -1
	encoding[index] = 1

	return encoding


def atom_features(atom: Chem.rdchem.Atom, functional_groups: List[int] = None) -> List[Union[bool, int, float]]: # type: ignore
	"""
	Builds a feature vector for an atom.

	:param atom: An RDKit atom.
	:param functional_groups: A k-hot vector indicating the functional groups the atom belongs to.
	:return: A list containing the atom features.
	"""
	features = onek_encoding_unk(atom.GetAtomicNum() - 1, ATOM_FEATURES['atomic_num']) + \
			onek_encoding_unk(atom.GetTotalDegree(), ATOM_FEATURES['degree']) + \
			onek_encoding_unk(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge']) + \
			onek_encoding_unk(int(atom.GetChiralTag()), ATOM_FEATURES['chiral_tag']) + \
			onek_encoding_unk(int(atom.GetTotalNumHs()), ATOM_FEATURES['num_Hs']) + \
			onek_encoding_unk(int(atom.GetHybridization()), ATOM_FEATURES['hybridization']) + \
			[1 if atom.GetIsAromatic() else 0] + \
			[atom.GetMass() * 0.01]	# scaled to about the same range as other features
	if functional_groups is not None:
		features += functional_groups
	return features # type: ignore


def bond_features(bond: Chem.rdchem.Bond) -> List[Union[bool, int, float]]:
	"""
	Builds a feature vector for a bond.

	:param bond: A RDKit bond.
	:return: A list containing the bond features.
	"""
	if bond is None:
		fbond = [1] + [0] * (BOND_FDIM - 1)
	else:
		bt = bond.GetBondType()
		fbond = [
			0,	# bond is not None
			bt == Chem.rdchem.BondType.SINGLE,
			bt == Chem.rdchem.BondType.DOUBLE,
			bt == Chem.rdchem.BondType.TRIPLE,
			bt == Chem.rdchem.BondType.AROMATIC,
			(bond.GetIsConjugated() if bt is not None else 0),
			(bond.IsInRing() if bt is not None else 0)
		]
		fbond += onek_encoding_unk(int(bond.GetStereo()), list(range(6)))
	return fbond # type: ignore


class MolGraph:
	"""
	A MolGraph represents the graph structure and featurization of a single molecule.

	A MolGraph computes the following attributes:
	- n_atoms: The number of atoms in the molecule.
	- n_bonds: The number of bonds in the molecule.
	- f_atoms: A mapping from an atom index to a list atom features.
	- f_bonds: A mapping from a bond index to a list of bond features.
	- a2b: A mapping from an atom index to a list of incoming bond indices.
	- b2a: A mapping from a bond index to the index of the atom the bond originates from.
	- b2revb: A mapping from a bond index to the index of the reverse bond.
	"""

	def __init__(self, mol: Union[str, Chem.Mol]): # type: ignore
		"""
		Computes the graph structure and featurization of a molecule.

		:param mol: A SMILES string or an RDKit molecule.
		"""
		#self.smiles = None
		#self.cid = cid
		""" changed for speedup -- always input Chem.Mol
		# Convert SMILES to RDKit molecule if necessary
		if type(mol) == str or type(mol) == np.str_:
			self.smiles = mol
			mol = Chem.MolFromSmiles(mol)
		"""
		self.mol = mol
		self.n_atoms = 0  # number of atoms
		self.n_bonds = 0  # number of bonds
		self.f_atoms = []  # mapping from atom index to atom features
		self.f_bonds = []  # mapping from bond index to concat(in_atom, bond) features
		self.a2b = []  # mapping from atom index to incoming bond indices
		self.b2a = []  # mapping from bond index to the index of the atom the bond is coming from
		self.b2revb = []  # mapping from bond index to the index of the reverse bond

		# Get atom features
		self.f_atoms = [atom_features(atom) for atom in mol.GetAtoms()] # type: ignore
		self.n_atoms = len(self.f_atoms)

		# Initialize atom to bond mapping for each atom
		for _ in range(self.n_atoms):
			self.a2b.append([])

		# Get bond features
		for a1 in range(self.n_atoms):
			for a2 in range(a1 + 1, self.n_atoms):
				bond = mol.GetBondBetweenAtoms(a1, a2) # type: ignore

				if bond is None:
					continue

				f_bond = bond_features(bond)
				self.f_bonds.append(self.f_atoms[a1] + f_bond)
				self.f_bonds.append(self.f_atoms[a2] + f_bond)

				# Update index mappings
				b1 = self.n_bonds
				b2 = b1 + 1
				self.a2b[a2].append(b1)  # b1 = a1 --> a2
				self.b2a.append(a1)
				self.a2b[a1].append(b2)  # b2 = a2 --> a1
				self.b2a.append(a2)
				self.b2revb.append(b2)
				self.b2revb.append(b1)
				self.n_bonds += 2

	def set_atom_feature(self, fatom):
		self.f_atoms = fatom

	def set_bond_feature(self, fbond):
		self.f_bonds = fbond

	def get_chem_mol(self):
		return self.mol
	
	def draw(self, weights=None, fname="molecule", to_save=True):
		if weights is None: 
			weights = list(range(self.n_atoms))
		d2d = Draw.MolDraw2DCairo(400, 400)
		img = SimilarityMaps.GetSimilarityMapFromWeights(mol=self.mol, 
													weights=weights, 
													draw2d=d2d, 
													colorMap=None, 
													# contourLines=10
                                                    )
		if to_save:
			d2d.WriteDrawingText(f"/media/external_16TB_1/kian_khalilpour/DrugRanker/assets/molecules/{fname}.png")

	'''
	def get_components(self, atom_messages: bool = False) -> Tuple[torch.FloatTensor, torch.FloatTensor,
									torch.LongTensor, torch.LongTensor, torch.LongTensor]:
		"""
		Returns the components of the BatchMolGraph.

		:param atom_messages: Whether to use atom messages instead of bond messages. This changes the bond features
		to contain only bond features rather than a concatenation of atom and bond features.
		:return: A tuple containing PyTorch tensors with the atom features, bond features, and graph structure
		and two lists indicating the scope of the atoms and bonds (i.e. which molecules they belong to).
		"""
		if atom_messages:
			f_bonds = self.f_bonds[:, :get_bond_fdim(atom_messages=atom_messages)]
		else:
			f_bonds = self.f_bonds

		max_num_bonds = max(len(a) for a in self.a2b)
		a2b = [self.a2b[a] + [0]*(max_num_bonds - len(self.a2b[a])) for a in range(self.n_atoms)]
	#	print(torch.FloatTensor(self.f_atoms).shape, torch.FloatTensor(f_bonds).shape, self.a2b.shape, len(self.b2a), len(self.b2revb))
		return torch.FloatTensor(self.f_atoms), torch.FloatTensor(f_bonds), torch.LongTensor(a2b), torch.LongTensor(self.b2a), torch.LongTensor(self.b2revb)
	'''

class BatchMolGraph:
	"""
	A BatchMolGraph represents the graph structure and featurization of a batch of molecules.

	A BatchMolGraph contains the attributes of a MolGraph plus:
	- atom_fdim: The dimensionality of the atom features.
	- bond_fdim: The dimensionality of the bond features (technically the combined atom/bond features).
	- a_scope: A list of tuples indicating the start and end atom indices for each molecule.
	- b_scope: A list of tuples indicating the start and end bond indices for each molecule.
	- max_num_bonds: The maximum number of bonds neighboring an atom in this batch.
	- b2b: (Optional) A mapping from a bond index to incoming bond indices.
	- a2a: (Optional): A mapping from an atom index to neighboring atom indices.
	"""

	def __init__(self, mol_graphs: List[MolGraph]):
	#	self.smiles_batch = [mol_graph.smiles for mol_graph in mol_graphs]
#		self.ic_batch = [mol_graph.ic for mol_graph in mol_graphs]
#		self.cid_batch = [mol_graph.cid for mol_graph in mol_graphs]
		self.mol_graphs = mol_graphs
		self.atom_fdim = get_atom_fdim()
		self.bond_fdim = get_bond_fdim()

		# Start n_atoms and n_bonds at 1 b/c zero padding
		self.n_atoms = 1  # number of atoms (start at 1 b/c need index 0 as padding)
		self.n_bonds = 1  # number of bonds (start at 1 b/c need index 0 as padding)
		self.a_scope = []  # list of tuples indicating (start_atom_index, num_atoms) for each molecule
		self.b_scope = []  # list of tuples indicating (start_bond_index, num_bonds) for each molecule

		# All start with zero padding so that indexing with zero padding returns zeros
		f_atoms = [[0] * self.atom_fdim]  # atom features
		f_bonds = [[0] * self.bond_fdim]  # combined atom/bond features
		a2b = [[]]	# mapping from atom index to incoming bond indices
		b2a = [0]  # mapping from bond index to the index of the atom the bond is coming from
		b2revb = [0]  # mapping from bond index to the index of the reverse bond
		for mol_graph in mol_graphs:
			f_atoms.extend(mol_graph.f_atoms) # type: ignore
			f_bonds.extend(mol_graph.f_bonds)

			for a in range(mol_graph.n_atoms):
				a2b.append([b + self.n_bonds for b in mol_graph.a2b[a]])

			for b in range(mol_graph.n_bonds):
				b2a.append(self.n_atoms + mol_graph.b2a[b])
				b2revb.append(self.n_bonds + mol_graph.b2revb[b])

			self.a_scope.append((self.n_atoms, mol_graph.n_atoms))
			self.b_scope.append((self.n_bonds, mol_graph.n_bonds))
			self.n_atoms += mol_graph.n_atoms
			self.n_bonds += mol_graph.n_bonds

		self.max_num_bonds = max(1, max(len(in_bonds) for in_bonds in a2b))  # max with 1 to fix a crash in rare case of all single-heavy-atom mols

		self.f_atoms = torch.FloatTensor(f_atoms)
		self.f_bonds = torch.FloatTensor(f_bonds)
		self.a2b = torch.LongTensor([a2b[a] + [0] * (self.max_num_bonds - len(a2b[a])) for a in range(self.n_atoms)])
		self.b2a = torch.LongTensor(b2a)
		self.b2revb = torch.LongTensor(b2revb)
		self.b2b = None  # try to avoid computing b2b b/c O(n_atoms^3)
		self.a2a = None  # only needed if using atom messages

	def get_components(self, atom_messages: bool = False) -> Tuple[torch.FloatTensor, torch.FloatTensor,
									torch.LongTensor, torch.LongTensor, torch.LongTensor,
									List[Tuple[int, int]], List[Tuple[int, int]]]:
		"""
		Returns the components of the BatchMolGraph.

		:param atom_messages: Whether to use atom messages instead of bond messages. This changes the bond features
		to contain only bond features rather than a concatenation of atom and bond features.
		:return: A tuple containing PyTorch tensors with the atom features, bond features, and graph structure
		and two lists indicating the scope of the atoms and bonds (i.e. which molecules they belong to).
		"""
		if atom_messages:
			f_bonds = self.f_bonds[:, :get_bond_fdim(atom_messages=atom_messages)]
		else:
			f_bonds = self.f_bonds

		return self.f_atoms, f_bonds, self.a2b, self.b2a, self.b2revb, self.a_scope, self.b_scope # type: ignore

	def get_b2b(self) -> torch.LongTensor:
		"""
		Computes (if necessary) and returns a mapping from each bond index to all the incoming bond indices.

		:return: A PyTorch tensor containing the mapping from each bond index to all the incoming bond indices.
		"""

		if self.b2b is None:
			b2b = self.a2b[self.b2a]  # num_bonds x max_num_bonds
			# b2b includes reverse edge for each bond so need to mask out
			revmask = (b2b != self.b2revb.unsqueeze(1).repeat(1, b2b.size(1))).long()  # num_bonds x max_num_bonds
			self.b2b = b2b * revmask

		return self.b2b # type: ignore

	def get_a2a(self) -> torch.LongTensor:
		"""
		Computes (if necessary) and returns a mapping from each atom index to all neighboring atom indices.

		:return: A PyTorch tensor containing the mapping from each bond index to all the incodming bond indices.
		"""
		if self.a2a is None:
			# b = a1 --> a2
			# a2b maps a2 to all incoming bonds b
			# b2a maps each bond b to the atom it comes from a1
			# thus b2a[a2b] maps atom a2 to neighboring atoms a1
			self.a2a = self.b2a[self.a2b]  # num_atoms x max_num_bonds

		return self.a2a # type: ignore
		
	def get_chem_mols(self):
		return self.mol_graphs

def mol2graph(mols: Union[List[str], List[Chem.Mol]]) -> BatchMolGraph: # type: ignore
	"""
	Converts a list of SMILES strings or RDKit molecules to a BatchMolGraph containing the batch of molecular graphs.

	:param mols: A list of SMILES strings or a list of RDKit molecules.
	:return: A BatchMolGraph containing the combined molecular graph for the molecules
	"""
	return BatchMolGraph([MolGraph(mol) for mol in mols])
