# 相似度换成Dice

import numpy as np
import pickle

from ddc_pub import ddc_v3 as ddc
#import molvecgen
import rdkit, h5py

from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem, QED, rdFMCS, RDKFingerprint
from rdkit import rdBase
import pandas as pd

def get_sim(mol, sub_mol) -> float: 
    try:
        res = rdFMCS.FindMCS([mol, sub_mol], timeout=1, bondCompare=rdFMCS.BondCompare.CompareAny, ringMatchesRingOnly=True, atomCompare=rdFMCS.AtomCompare.CompareAny)
        if res.smartsString == "" or res.canceled:
            return 0
        mcs_mol = Chem.MolFromSmarts(res.smartsString)
        Chem.SanitizeMol(mcs_mol)

        mcs_mol_fp = RDKFingerprint(mcs_mol)
        sub_mol_fp = RDKFingerprint(sub_mol)
        sim = DataStructs.FingerprintSimilarity(sub_mol_fp, mcs_mol_fp, metric=DataStructs.DiceSimilarity)

        return sim
    except Exception as e:
        #print(e)
        return 0

sub_mol = Chem.MolFromSmiles("O=C(OC)C1=CC(C2=O)CCCC2C1=O")
target_names = ["logp", "tpsa", "sub_similarity", "qed", "hba", "hbd", "gasas"]
filenames = ['CHEMBL25_TRAIN_MOLS.h5', 'CHEMBL25_FILTERED_1.h5', 'CHEMBL25_FILTERED_2.h5', 'GENERATED_MOLS_1.h5', 'GENERATED_MOLS_2.h5']
#filenames = ['GENERATED_MOLS_2.h5']

for filename in filenames:
    filename = 'datasets/' + filename
    
    with h5py.File(filename, "r") as f:
        binmols = np.asarray(f['mols'])

    mols = [Chem.Mol(binmol) for binmol in binmols]

    sims = [get_sim(mol,sub_mol) for mol in mols]

    with h5py.File(filename[:-3] + "_descrs.h5", "a") as f:
        data = np.asarray(sims)
        del f['sub_similarity']
        f.create_dataset("sub_similarity", data=data)
    
    print("Calculated {}.".format(filename))

