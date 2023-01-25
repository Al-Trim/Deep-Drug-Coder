

import numpy as np
import pickle

from ddc_pub import ddc_v3 as ddc
#import molvecgen
import rdkit, h5py

from GASA import gasa

from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem, QED, rdFMCS
from rdkit import rdBase
import pandas as pd

sub_mol = Chem.MolFromSmiles("O=C(OC)C1=CC(C2=O)CCCC2C1=O")
target_names = ["logp", "tpsa", "sub_similarity", "qed", "hba", "hbd", "gasas"]
#filenames = ['CHEMBL25_TRAIN_MOLS.h5', 'CHEMBL25_FILTERED_1.h5', 'CHEMBL25_FILTERED_2.h5', 'GENERATED_MOLS_1.h5', 'GENERATED_MOLS_2.h5']
filenames = ['GENERATED_MOLS_2.h5']

for filename in filenames:
    filename = 'datasets/' + filename
    
    with h5py.File(filename, "r") as f:
        binmols = np.asarray(f['mols'])

    mols = [Chem.Mol(binmol) for binmol in binmols]

    molwts = [Descriptors.ExactMolWt(mol) for mol in mols]

    with h5py.File(filename[:-3] + "_descrs.h5", "w") as f:
        data = np.asarray(molwts)
        f.create_dataset("molwt", data=data)

