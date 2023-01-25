#生成少一个侧链的分子，并筛出子结构相似度100%的分子

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

# 计算子结构相似度
def get_sim(mol, sub_mol) -> float: 
    try:
        res = rdFMCS.FindMCS([mol, sub_mol], timeout=1, bondCompare=rdFMCS.BondCompare.CompareAny, ringMatchesRingOnly=True, atomCompare=rdFMCS.AtomCompare.CompareAny)
        if res.smartsString == "" or res.canceled:
            return 0
        mcs_mol = Chem.MolFromSmarts(res.smartsString)
        Chem.SanitizeMol(mcs_mol)

        mcs_mol_fp = RDKFingerprint(mcs_mol)
        sub_mol_fp = RDKFingerprint(sub_mol)
        sim = DataStructs.FingerprintSimilarity(sub_mol_fp, mcs_mol_fp)

        return sim
    except Exception as e:
        #print(e)
        return 0

# 计算描述符
def get_descriptors(mol, sub_mol) -> list:
    descriptors = []
    if mol:
        try:
            logp = Descriptors.MolLogP(mol)
            tpsa = Descriptors.TPSA(mol)
            sim = get_sim(mol, sub_mol)
            hba = rdMolDescriptors.CalcNumHBA(mol)
            hbd = rdMolDescriptors.CalcNumHBD(mol)
            qed = QED.qed(mol)

            descriptors = [logp, tpsa, sim, qed, hba, hbd]
        except Exception as e:
            #print(e)
            return descriptors
    else:
        print("Invalid generation.")
    return descriptors

sub_mol = Chem.MolFromSmiles("O=C(OC)C1=CC(C2=O)CCCC2C1=O")
target_names = ["logp", "tpsa", "sub_similarity", "qed", "hba", "hbd", "gasas"]
#filenames = ['CHEMBL25_TRAIN_MOLS.h5', 'CHEMBL25_FILTERED_1.h5', 'CHEMBL25_FILTERED_2.h5', 'GENERATED_MOLS_1.h5', 'GENERATED_MOLS_2.h5']
filenames = ['GENERATED_MOLS_2.h5']

for filename in filenames:
    filename = 'datasets/' + filename
    
    with h5py.File(filename, "r") as f:
        binmols = np.asarray(f['mols'])

    mols = [Chem.Mol(binmol) for binmol in binmols]

    gasas = gasa.GASA([Chem.MolToSmiles(mol) for mol in mols])[1]

    descrs = [get_descriptors(mol, sub_mol) for mol in mols]
    descrs = list(np.asarray(descrs).T)
    descrs.append(gasas)
    descrs = np.asarray(descrs)

    with h5py.File(filename[:-3] + "_descrs.h5", "w") as f:
        for idx, name in enumerate(target_names):
            data = np.asarray(descrs[idx])
            f.create_dataset(name, data=data)
            print("saved {}, shape={}".format(name, data.shape))

