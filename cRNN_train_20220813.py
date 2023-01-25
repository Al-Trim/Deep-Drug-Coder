#加入生成的分子，重新训练

import numpy as np
import random

import rdkit
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFMCS, AllChem, RDKFingerprint, Descriptors, rdMolDescriptors, AllChem, QED
from rdkit import DataStructs

import h5py
import pandas as pd
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
                pass
                #print("Exception Occurred at {:0}:{:1}".format(idx, Chem.MolToSmiles(mol)))
                #print(e)
        else:
            print("Invalid generation.")

    return np.asarray(descriptors)

class mol_generator:
    def __init__(self, seed_smile: str = "", sub_smile:str = "", model: str = "", qsar_model: str = ""):
        self.mols = []
        self.target = []
        self.target_names = ["logp", "tpsa", "sub_similarity", "qed", "hba", "hbd"]
        self.sani_mols = []
        self.sani_properties = []
        self.data = None
        self.set_seed(seed_smile)
        self.set_sub(sub_smile)
        self.set_model(model)
        self.set_qsar_model(qsar_model)

    def set_seed(self, seed_smile):
        if(seed_smile == ""):
            return
        #self.seed_smile = seed_smile
        self.seed_mol = Chem.MolFromSmiles(seed_smile)

        print("Seed Molecular:")
        print(Chem.MolToSmiles(self.seed_mol))

    def set_sub(self, sub_smile):
        if(sub_smile == ""):
            return
        self.sub_mol = Chem.MolFromSmiles(sub_smile)

        print("Substruct:")
        print(Chem.MolToSmiles(self.sub_mol))

    def set_model(self, model):
        if(model == ""):
            return
        self.model = ddc.DDC(model_name=model)

    def set_qsar_model(self, qsar_model):
        if(qsar_model == ""):
            return
        self.qsar_model = pickle.load(open(qsar_model, "rb"))["classifier_sv"]

    #Calculate the similarity between the aimed substructure and generated mol
    def get_sim(self, mol, sub_mol) -> float: 
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

    def get_descriptors(self, mol):
        logp = Descriptors.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        sim = self.get_sim(mol, self.sub_mol)
        hba = rdMolDescriptors.CalcNumHBA(mol)
        hbd = rdMolDescriptors.CalcNumHBD(mol)
        qed = QED.qed(mol)
        
        return [logp, tpsa, sim, qed, hba, hbd]

    def sanitize(self, mol):  # 检查分子
        try:
            Chem.SanitizeMol(mol)
            return mol
        except Exception as e:
            return None
            #print(e)

    # 采样指定性质的分子
    def sample(self, sample_times: int = 4, conditions: list = [None]*6):
        # 确定目标
        #assert len(conditions) >= 7
        self.target = self.get_descriptors(self.seed_mol)
        for i in range(len(conditions)):
            if(conditions[i] != None):
                self.target[i] = conditions[i]
        self.target = np.array(self.target)
        print("target:{:}".format(self.target))
        # 采样
        smiles_out = []
        self.model.batch_input_length = 256  # 太大会减慢速度
        for i in range(sample_times):
            smiles, _ = self.model.predict_batch(latent=self.target.reshape(1, -1), temp=1.0)
            print("#{:}:{:}".format(i,smiles))
            smiles_out.append(smiles)
        smiles_out = np.concatenate(smiles_out)
        self.mols = [Chem.MolFromSmiles(smi) for smi in smiles_out]
        # 检查分子
        self.sani_mols.clear()
        self.sani_properties.clear()
        for mol in self.mols:
            sani_mol = self.sanitize(mol)
            if sani_mol != None:
                self.sani_mols.append(sani_mol)
                self.sani_properties.append(self.get_descriptors(sani_mol))
        # 打印结果
        print("生成分子数:{:},有效性:{:}".format(
            len(self.mols), len(self.sani_mols)/len(self.mols)))

    # 根据id显示生成的分子
    def showmol(self, i):  
        print(Chem.MolToSmiles(self.sani_mols[i]))

    #筛选sub_similarity==1的分子
    def filter_data(self, filename: str=""):
        #筛选结果
        print("Saving results.")
        self.binmols = np.asarray([[i.ToBinary() for i in self.sani_mols]])
        self.binmols_data = pd.DataFrame(self.binmols.T, columns=["binmol"], copy=True)
        self.properties_data = pd.DataFrame(self.sani_properties, columns=self.target_names, copy=True)
        self.filtered_data = self.binmols_data.loc[[i==1 for i in self.properties_data["sub_similarity"]]]
        print(self.filtered_data.head(5))
        #保存文件
        with h5py.File(filename, "w") as f:
            f.create_dataset("mols", data=np.asarray(self.filtered_data["binmol"]))

#Start Here
rdBase.DisableLog("rdApp.*")

for i in range(3):
    print("Running round {}".format(i))
#-----------------------------------generate
    generator = mol_generator("CC1(C)C(CCC2(C)C1CCC(C34C)(C)C2CC(C)(C4=O)C(C)=C(C(OC)=O)C3=O)=O", "O=C(OC)C1=CC(C2=O)CCCC2C1=O", "models/model_0812")
    generator.sample(10000)
    generator.filter_data("datasets/model_0813_{}.h5".format(i))

#-----------------------------------train
    # Set substruct
    sub_mol = Chem.MolFromSmiles("O=C(OC)C1=CC(C2=O)CCCC2C1=O")

    # Load dataset
    print("Loading dataset.")
    #dataset_filename = r"C:\Users\Leave\OneDrive - hust.edu.cn\大创\pcko1-Deep-Drug-Coder-d6e7ef3\datasets\CHEMBL25_TRAIN_MOLS.h5"

    #随机从CHEMBL25里挑 * 5000
    dataset_filename = r"datasets/CHEMBL25_TRAIN_MOLS.h5"
    rnd_idx = random.randint(0, 1340000)
    with h5py.File(dataset_filename, "r") as f:
        binmols_1 = np.asarray(f["mols"][rnd_idx:rnd_idx+5000])

    #从CHEMBL25筛选出的子结构相似度0.3~1的分子 * 12000
    dataset_filename = r"datasets/CHEMBL25_FILTERED_2.h5"
    with h5py.File(dataset_filename, "r") as f:
        binmols_2 = np.asarray(f["mols"])

    #种子分子 * 1
    binmols_3 = np.asarray([Chem.MolFromSmiles("CC1(C)C(CCC2(C)C1CCC(C34C)(C)C2CC(C)(C4=O)C(C)=C(C(OC)=O)C3=O)=O").ToBinary()])

    #用0808模型生成的子结构相似度100%（但是缺少双键）的分子 * 1257
    dataset_filename = r"datasets/GENERATED_MOLS_1.h5"
    with h5py.File(dataset_filename, "r") as f:
        binmols_4 = np.asarray(f["mols"])

    #用0812模型生成的子结构相似度100%的分子 * ?
    dataset_filename = r"datasets/model_0813_{}.h5".format(i)
    with h5py.File(dataset_filename, "r") as f:
        binmols_5 = np.asarray(f["mols"])

    binmols = np.concatenate((binmols_1, binmols_3, binmols_2, binmols_3, binmols_4, binmols_3, binmols_5))

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
    name = "ChEMBL25_TRAIN_0813"

    dataset_info = {"charset": charset, "maxlen": maxlen, "name": name}

    # Initialize a model
    if i==0:
        model = ddc.DDC(x              = descr,        # input
                        y              = binmols,      # output
                        model_name     = "models/model_0812")          # existed model name
    else:
        model = ddc.DDC(x=descr, y=binmols, model_name="models/model_0812_retrain_{}".format(i-1))

    model.fit(epochs              = 300,         # number of epochs
            lr                  = 1e-3,        # initial learning rate for Adam, recommended
            model_name          = "model_0812", # base name to append the checkpoints with
            checkpoint_dir      = "checkpoints_0812_retrain/",          # save checkpoints in the notebook's directory
            mini_epochs         = 10,          # number of sub-epochs within an epoch to trigger lr decay
            save_period         = 50,          # checkpoint frequency (in mini_epochs)
            lr_decay            = True,        # whether to use exponential lr decay or not
            sch_epoch_to_start  = 500,         # mini-epoch to start lr decay (bypassed if lr_decay=False)
            sch_lr_init         = 1e-3,        # initial lr, should be equal to lr (bypassed if lr_decay=False)
            sch_lr_final        = 1e-6,        # final lr before finishing training (bypassed if lr_decay=False)
            patience            = 25)          # patience for Keras' ReduceLROnPlateau (bypassed if lr_decay=True)

    # Save the final model
    model.save("models/model_0812_retrain_{}".format(i))