#Find mols whose sub_sim >0.3 (MCS fuzzy search enabled)
import rdkit, h5py
from rdkit import Chem
from rdkit.Chem import rdFMCS,AllChem
from rdkit import DataStructs

import numpy as np

def get_sim(idx, mol, sub_mol) -> float: 
    try:
        res = rdFMCS.FindMCS([mol, sub_mol], timeout=1, bondCompare=rdFMCS.BondCompare.CompareAny, ringMatchesRingOnly=True, atomCompare=rdFMCS.AtomCompare.CompareAny)
        if res.smartsString == "" or res.canceled:
            return 0
        mcs_mol = Chem.MolFromSmarts(res.smartsString)
        Chem.SanitizeMol(mcs_mol)

        mcs_mol_fp = AllChem.GetMorganFingerprintAsBitVect(mcs_mol, 2, nBits=2048)
        sub_mol_fp = AllChem.GetMorganFingerprintAsBitVect(sub_mol, 2, nBits=2048)
        sim = DataStructs.FingerprintSimilarity(sub_mol_fp, mcs_mol_fp)

    except Exception as e:
        print("...Exception Occurred at {:0}:{:1}:".format(idx, Chem.MolToSmiles(mol)))
        print(e)
        return 0
    
    return sim

#Set Substruct
sub_mol = Chem.MolFromSmiles("O=C(OC)C1=CC(C2=O)CCCC2C1=O")

#Load vanilla HDF5 file
#dataset_filename = r"C:\Users\Leave\OneDrive - hust.edu.cn\大创\pcko1-Deep-Drug-Coder-d6e7ef3\datasets\CHEMBL25_TRAIN_MOLS.h5"
dataset_filename = r"datasets/CHEMBL25_TRAIN_MOLS.h5"
with h5py.File(dataset_filename, "r") as f:
    binmols = np.asarray(f["mols"])

#Find mols
binmol_dataset = []
idx_dataset = []
sim_dataset = []
for idx, binmol in enumerate(binmols):
    mol = Chem.Mol(binmol)
    sim = get_sim(idx, mol, sub_mol)
    if(sim > 0.3):
        binmol_dataset.append(binmol)
        idx_dataset.append(idx)
        sim_dataset.append(sim)

#Save new file
new_filename = "datasets/CHEMBL25_FILTERED_2.h5"
with h5py.File(new_filename, "w") as f:
    f.create_dataset("mols", data=np.asarray(binmol_dataset))
    f.create_dataset("idxs", data=np.asarray(idx_dataset))
    f.create_dataset("sims", data=np.asarray(sim_dataset))