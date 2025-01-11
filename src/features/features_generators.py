from typing import Callable, List, Union

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
# from map4 import MAP4Calculator
from rdkit.Avalon.pyAvalonTools import GetAvalonFP
from rdkit.Chem.AtomPairs.Pairs import GetAtomPairFingerprintAsBitVect
from rdkit.Chem.rdMolDescriptors import GetHashedAtomPairFingerprintAsBitVect
from rdkit.Chem.Pharm2D import Generate, Gobbi_Pharm2D
from rdkit.Chem.Descriptors import CalcMolDescriptors
# from descriptastorus.descriptors import rdDescriptors, rdNormalizedDescriptors

Molecule = Union[str, Chem.Mol]
FeaturesGenerator = Callable[[Molecule], np.ndarray]


FEATURES_GENERATOR_REGISTRY = {}


def register_features_generator(features_generator_name: str) -> Callable[[FeaturesGenerator], FeaturesGenerator]:
	"""
	Registers a features generator.

	:param features_generator_name: The name to call the FeaturesGenerator.
	:return: A decorator which will add a FeaturesGenerator to the registry using the specified name.
	"""
	def decorator(features_generator: FeaturesGenerator) -> FeaturesGenerator:
		FEATURES_GENERATOR_REGISTRY[features_generator_name] = features_generator
		return features_generator

	return decorator


def get_features_generator(features_generator_name: str) -> FeaturesGenerator:
	"""
	Gets a registered FeaturesGenerator by name.

	:param features_generator_name: The name of the FeaturesGenerator.
	:return: The desired FeaturesGenerator.
	"""
	if features_generator_name not in FEATURES_GENERATOR_REGISTRY:
		raise ValueError(f'Features generator "{features_generator_name}" could not be found. '
				f'If this generator relies on rdkit features, you may need to install descriptastorus.')

	return FEATURES_GENERATOR_REGISTRY[features_generator_name]


def get_available_features_generators() -> List[str]:
	"""Returns the names of available features generators."""
	return list(FEATURES_GENERATOR_REGISTRY.keys())


MORGAN_RADIUS = 3
MORGAN_NUM_BITS = 2048


@register_features_generator('morgan')
def morgan_binary_features_generator(mol: Molecule,
					radius: int = MORGAN_RADIUS,
					num_bits: int = MORGAN_NUM_BITS) -> np.ndarray:
	"""
	Generates a binary Morgan fingerprint for a molecule.

	:param mol: A molecule (i.e. either a SMILES string or an RDKit molecule).
	:param radius: Morgan fingerprint radius.
	:param num_bits: Number of bits in Morgan fingerprint.
	:return: A 1-D numpy array containing the binary Morgan fingerprint.
	"""
	mol = Chem.MolFromSmiles(mol) if type(mol) == str else mol
	features_vec = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=num_bits)
	features = np.zeros((1,))
	DataStructs.ConvertToNumpyArray(features_vec, features)

	return features


@register_features_generator('morgan_count')
def morgan_counts_features_generator(mol: Molecule,
					radius: int = MORGAN_RADIUS,
					num_bits: int = MORGAN_NUM_BITS) -> np.ndarray:
	"""
	Generates a counts-based Morgan fingerprint for a molecule.

	:param mol: A molecule (i.e. either a SMILES string or an RDKit molecule).
	:param radius: Morgan fingerprint radius.
	:param num_bits: Number of bits in Morgan fingerprint.
	:return: A 1D numpy array containing the counts-based Morgan fingerprint.
	"""
	mol = Chem.MolFromSmiles(mol) if type(mol) == str else mol
	features_vec = AllChem.GetHashedMorganFingerprint(mol, radius, nBits=num_bits)
	features = np.zeros((1,))
	DataStructs.ConvertToNumpyArray(features_vec, features)

	return features

@register_features_generator('rdkit_2d_desc')
def rdkit_2d_desc_features_generator(mol: Molecule) -> np.ndarray:
	"""
	Generates RDKit 2D features for a molecule.

	:param mol: A molecule (i.e. either a SMILES string or an RDKit molecule).
	:return: A 1D numpy array containing the RDKit 2D features.
	"""
	mol = Chem.MolFromSmiles(mol) if isinstance(mol, str) else mol
	features_vec = CalcMolDescriptors(mol)
	features_vec = np.array(list(features_vec.values()))

	# normalized output
	features = (features_vec-min(features_vec))/(max(features_vec)-min(features_vec))
	features = features.astype("float32")
	features = np.nan_to_num(features)

	return features

try:
	from descriptastorus.descriptors import rdDescriptors, rdNormalizedDescriptors

	@register_features_generator('rdkit_2d')
	def rdkit_2d_features_generator(mol: Molecule) -> np.ndarray:
		"""
		Generates RDKit 2D features for a molecule.

		:param mol: A molecule (i.e. either a SMILES string or an RDKit molecule).
		:return: A 1D numpy array containing the RDKit 2D features.
		"""
		smiles = Chem.MolToSmiles(mol, isomericSmiles=True) if type(mol) != str else mol
		generator = rdDescriptors.RDKit2D()
		features = generator.process(smiles)[1:]

		return features

	@register_features_generator('rdkit_2d_normalized')
	def rdkit_2d_normalized_features_generator(mol: Molecule) -> np.ndarray:
		"""
		Generates RDKit 2D normalized features for a molecule.

		:param mol: A molecule (i.e. either a SMILES string or an RDKit molecule).
		:return: A 1D numpy array containing the RDKit 2D normalized features.
		"""
		smiles = Chem.MolToSmiles(mol, isomericSmiles=True) if type(mol) != str else mol
		generator = rdNormalizedDescriptors.RDKit2DNormalized()
		features = generator.process(smiles)[1:]

		return features
except ImportError as e:
	print(e)
	pass

@register_features_generator('morgan_tanimoto_bioassay')
def custom_features_generator(mol: Molecule, list_mols: List[Molecule]) -> np.ndarray:
	feature_mol = morgan_binary_features_generator(mol)
	feature_list_mols = [morgan_binary_features_generator(m) for m in list_mols]

	feature_mol = DataStructs.cDataStructs.CreateFromBitString(''.join(map(str, feature_mol)))
	feature_list_mols = [DataStructs.cDataStructs.CreateFromBitString(''.join(map(str, i))) for i in feature_list_mols]

	return DataStructs.BulkTanimotoSimilarity(feature_mol, feature_list_mols)

# @register_features_generator('map4')
# def map4_features_generator(mol: Molecule) -> np.ndarray:
# 	mol = Chem.MolFromSmiles(mol) if isinstance(mol, str) else mol
# 	map4_calculator = MAP4Calculator()
# 	return np.array(map4_calculator.calculate(mol))

@register_features_generator('avalon')
def avalon_features_generator(mol: Molecule, num_bits: int = 1024) -> np.ndarray:
	mol = Chem.MolFromSmiles(mol) if isinstance(mol, str) else mol
	features_vec = GetAvalonFP(mol, nBits=num_bits)
	features = np.zeros((1,))
	DataStructs.ConvertToNumpyArray(features_vec, features)
	return features

@register_features_generator('atom_pair')
def atom_pair_features_generator(mol: Molecule) -> np.ndarray:
	mol = Chem.MolFromSmiles(mol) if isinstance(mol, str) else mol
	features_vec = GetHashedAtomPairFingerprintAsBitVect(mol, 1024)
	features = np.zeros((1,))
	DataStructs.ConvertToNumpyArray(features_vec, features)
	return features

@register_features_generator('2d_pharmacophore')
def pharmacophore_2d_features_generator(mol: Molecule) -> np.ndarray:
	mol = Chem.MolFromSmiles(mol) if isinstance(mol, str) else mol

	factory = Gobbi_Pharm2D.factory
	features_vec = Generate.Gen2DFingerprint(mol, factory)

	features = np.zeros((len(features_vec),), dtype=np.int8)
	for idx in features_vec.GetOnBits():
		features[idx] = 1

	return features

@register_features_generator('layered_rdkit')
def layered_rdkit_features_generator(mol: Molecule) -> np.ndarray:
	mol = Chem.MolFromSmiles(mol) if isinstance(mol, str) else mol
	features_vec = AllChem.LayeredFingerprint(mol, fpSize=1024)
	features = np.zeros((1,))
	DataStructs.ConvertToNumpyArray(features_vec, features)
	return features

@register_features_generator('rdkit2d_morgan')
def rdkit2d_morgan_features_generator(mol: Molecule) -> np.ndarray:
	morgan_features = morgan_binary_features_generator(mol)
	rdkit_2d_features = rdkit_2d_desc_features_generator(mol)

	features = np.concatenate((morgan_features, rdkit_2d_features))
	return features

@register_features_generator('rdkit2d_morganc')
def rdkit2d_morgan_features_generator(mol: Molecule) -> np.ndarray:
	morgan_features = morgan_counts_features_generator(mol)
	rdkit_2d_features = rdkit_2d_desc_features_generator(mol)

	features = np.concatenate((morgan_features, rdkit_2d_features))
	return features

@register_features_generator('rdkit2d_atompair')
def rdkit2d_atompair_features_generator(mol: Molecule) -> np.ndarray:
	atompair_features = atom_pair_features_generator(mol)
	rdkit_2d_features = rdkit_2d_desc_features_generator(mol)

	features = np.concatenate((atompair_features, rdkit_2d_features))
	return features

"""
Custom features generator template.

Note: The name you use to register the features generator is the name
you will specify on the command line when using the --features_generator <name> flag.
Ex. python train.py ... --features_generator custom ...

@register_features_generator('custom')
def custom_features_generator(mol: Molecule) -> np.ndarray:
	# If you want to use the SMILES string
	smiles = Chem.MolToSmiles(mol, isomericSmiles=True) if type(mol) != str else mol

	# If you want to use the RDKit molecule
	mol = Chem.MolFromSmiles(mol) if type(mol) == str else mol

	# Replace this with code which generates features from the molecule
	features = np.array([0, 0, 1])

	return features
"""
