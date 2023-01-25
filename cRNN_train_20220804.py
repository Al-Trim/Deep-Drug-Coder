'''%load_ext autoreload
%autoreload 2
# Occupy a GPU for the model to be loaded 
%env CUDA_DEVICE_ORDER=PCI_BUS_ID
# GPU ID, if occupied change to an available GPU ID listed under !nvidia-smi
%env CUDA_VISIBLE_DEVICES=0'''

import numpy as np

import rdkit
from rdkit import Chem
from rdkit.Chem import rdFMCS,AllChem
from rdkit import DataStructs

import h5py
import ast
import pickle

from ddc_pub import ddc_v3 as ddc

#Calculate the similarity between the aimed substructure and generated mol
def get_sim(mol, sub_mol) -> float: 
    try:
        res = rdFMCS.FindMCS([mol, sub_mol], timeout=1)
        if res.smartsString == "" or res.canceled:
            return 0
        mcs_mol = Chem.MolFromSmarts(res.smartsString)
        Chem.SanitizeMol(mcs_mol)

        mcs_mol_fp = AllChem.GetMorganFingerprintAsBitVect(mcs_mol, 2, nBits=2048)
        sub_mol_fp = AllChem.GetMorganFingerprintAsBitVect(sub_mol, 2, nBits=2048)
        sim = DataStructs.FingerprintSimilarity(sub_mol_fp, mcs_mol_fp)

        return sim
    except Exception as e:
        print("...Exception Occurred at {:1}:".format(Chem.MolToSmiles(mol)))
        print(e)
        return 0

def get_descriptors(binmols_list, qsar_model=None):
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

                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                ecfp4 = np.zeros((2048,))
                DataStructs.ConvertToNumpyArray(fp, ecfp4)
                active = qsar_model.predict_proba([ecfp4])[0][1]
                descriptors.append([logp, tpsa, sim, qed, hba, hbd, active])

            except Exception as e:
                print("Exception Occurred at {:0}:{:1}".format(idx, Chem.MolToSmiles(mol)))
                print(e)
        else:
            print("Invalid generation.")

    return np.asarray(descriptors)

# Set substruct
sub_mol = Chem.MolFromSmiles("C1C(C2(=O))C(=O)C=CC2CC1")

# Load QSAR model
print("Loading QSAR model.")
qsar_model_name = "models/qsar_model.pickle"
with open(qsar_model_name, "rb") as file:
    qsar_model = pickle.load(file)["classifier_sv"]

# Load dataset
print("Loading dataset.")
#dataset_filename = r"C:\Users\Leave\OneDrive - hust.edu.cn\大创\pcko1-Deep-Drug-Coder-d6e7ef3\datasets\CHEMBL25_TRAIN_MOLS.h5"
dataset_filename = r"datasets/CHEMBL25_TRAIN_MOLS.h5"
with h5py.File(dataset_filename, "r") as f:
    binmols = f["mols"][0:30000]

# Calculate the descriptors for the molecules in the dataset
# This process takes a lot of time and it's good if the descriptors are
# pre-calculated and stored in a file to load every time
print("Calculating descriptors.")
descr = get_descriptors(binmols, qsar_model=qsar_model)
print("Calculated descriptors.")

# All apriori known characters of the SMILES in the dataset
charset = "Brc1(-23[nH])45C=NOso#FlS67+89%0"
# Apriori known max length of the SMILES in the dataset
maxlen = 128
# Name of the dataset
name = "ChEMBL25_TRAIN"

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
          model_name          = "model_0804", # base name to append the checkpoints with
          checkpoint_dir      = "checkpoints_0804/",          # save checkpoints in the notebook's directory
          mini_epochs         = 10,          # number of sub-epochs within an epoch to trigger lr decay
          save_period         = 50,          # checkpoint frequency (in mini_epochs)
          lr_decay            = True,        # whether to use exponential lr decay or not
          sch_epoch_to_start  = 500,         # mini-epoch to start lr decay (bypassed if lr_decay=False)
          sch_lr_init         = 1e-3,        # initial lr, should be equal to lr (bypassed if lr_decay=False)
          sch_lr_final        = 1e-6,        # final lr before finishing training (bypassed if lr_decay=False)
          patience            = 25)          # patience for Keras' ReduceLROnPlateau (bypassed if lr_decay=True)

# Save the final model
model.save("models/model_0804")