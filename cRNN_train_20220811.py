#换用新的指纹计算方法，重新得到生成含侧链的模型

import numpy as np
import random

import rdkit
from rdkit import Chem
from rdkit.Chem import rdFMCS, AllChem, RDKFingerprint
from rdkit import DataStructs

import h5py
import ast
import pickle

from ddc_pub import ddc_v3 as ddc

#Calculate the similarity between the aimed substructure and generated mol
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
        return 0

def get_descriptors(binmols_list):
    """Calculate molecular descriptors of SMILES in a list.
    The descriptors are logp, tpsa, mw, qed, hba, hbd and probability of being active towards DRD2.

    Returns:
        A np.ndarray of descriptors.
    """
    from tqdm import tqdm_notebook as tqdm
    import rdkit
    from rdkit import Chem, DataStructs
    from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem, QED

    descriptors = []
    active_mols = []

    for idx, binmol in enumerate(binmols_list):
        mol = Chem.Mol(binmol)
        if mol:
            global sub_mol
            try:
                logp = Descriptors.MolLogP(mol)
                tpsa = Descriptors.TPSA(mol)
                #molwt = Descriptors.ExactMolWt(mol)
                sim = get_sim(mol, sub_mol)
                hba = rdMolDescriptors.CalcNumHBA(mol)
                hbd = rdMolDescriptors.CalcNumHBD(mol)
                qed = QED.qed(mol)

                '''fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                ecfp4 = np.zeros((2048,))
                DataStructs.ConvertToNumpyArray(fp, ecfp4)
                active = qsar_model.predict_proba([ecfp4])[0][1]'''
                descriptors.append([logp, tpsa, sim, qed, hba, hbd])

            except Exception as e:
                print("Exception Occurred at {:0}:{:1}".format(idx, Chem.MolToSmiles(mol)))
                print(e)
        else:
            print("Invalid generation.")

    return np.asarray(descriptors)

# Set substruct
sub_mol = Chem.MolFromSmiles("O=C(OC)C1=CC(C2=O)CCCC2C1=O")

# Load dataset
print("Loading dataset.")
#dataset_filename = r"C:\Users\Leave\OneDrive - hust.edu.cn\大创\pcko1-Deep-Drug-Coder-d6e7ef3\datasets\CHEMBL25_TRAIN_MOLS.h5"

#随机从CHEMBL25里挑 * 10000
dataset_filename = r"datasets/CHEMBL25_TRAIN_MOLS.h5"
rnd_idx = random.randint(0, 1340000)
with h5py.File(dataset_filename, "r") as f:
    binmols_1 = np.asarray(f["mols"][rnd_idx:rnd_idx+10000])

#筛选出的子结构相似度0.3~1的分子 * 12000
dataset_filename = r"datasets/CHEMBL25_FILTERED_2.h5"
with h5py.File(dataset_filename, "r") as f:
    binmols_2 = np.asarray(f["mols"])

#种子分子 * 1
binmols_3 = np.asarray([Chem.MolFromSmiles("CC1(C)C(CCC2(C)C1CCC(C34C)(C)C2CC(C)(C4=O)C(C)=C(C(OC)=O)C3=O)=O").ToBinary()])

#用0808模型生成的子结构相似度100%的分子 * 1257
dataset_filename = r"datasets/GENERATED_MOLS_1.h5"
with h5py.File(dataset_filename, "r") as f:
    binmols_4 = np.asarray(f["mols"])

binmols = np.concatenate((binmols_1, binmols_3, binmols_2, binmols_3, binmols_4, binmols_3))
#binmols = np.concatenate(())

# Calculate the descriptors for the molecules in the dataset
# This process takes a lot of time and it's good if the descriptors are
# pre-calculated and stored in a file to load every time
print("Calculating descriptors.")
descr = get_descriptors(binmols)
print("Calculated descriptors.")

# All apriori known characters of the SMILES in the dataset
charset = "Brc1(-23[nH])45C=NOso#FlS67+89%0"
# Apriori known max length of the SMILES in the dataset
maxlen = 128
# Name of the dataset
name = "ChEMBL25_TRAIN_FILTERED_4"

dataset_info = {"charset": charset, "maxlen": maxlen, "name": name}

# Initialize a model
model = ddc.DDC(x              = descr,        # input
                y              = binmols,      # output
                dataset_info   = dataset_info, # dataset information
                scaling        = True,         # scale the descriptors
                noise_std      = 0.1,          # std of the noise layer
                lstm_dim       = 512,          # breadth of LSTM layers
                dec_layers     = 3,            # number of decoding layers
                batch_size     = 128)          # batch size for training

model.fit(epochs              = 300,         # number of epochs
          lr                  = 1e-3,        # initial learning rate for Adam, recommended
          model_name          = "model_0811", # base name to append the checkpoints with
          checkpoint_dir      = "checkpoints_0811/",          # save checkpoints in the notebook's directory
          mini_epochs         = 10,          # number of sub-epochs within an epoch to trigger lr decay
          save_period         = 50,          # checkpoint frequency (in mini_epochs)
          lr_decay            = True,        # whether to use exponential lr decay or not
          sch_epoch_to_start  = 500,         # mini-epoch to start lr decay (bypassed if lr_decay=False)
          sch_lr_init         = 1e-3,        # initial lr, should be equal to lr (bypassed if lr_decay=False)
          sch_lr_final        = 1e-6,        # final lr before finishing training (bypassed if lr_decay=False)
          patience            = 25)          # patience for Keras' ReduceLROnPlateau (bypassed if lr_decay=True)

# Save the final model
model.save("models/model_0811")