#尝试增加描述符

import numpy as np
import random

import rdkit
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFMCS, AllChem, RDKFingerprint, Descriptors, rdMolDescriptors, AllChem, QED
from rdkit import DataStructs

from rdkit.Chem import RDConfig
import os
import sys
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
# now you can import sascore!
import sascorer
mol = Chem.MolFromSmiles('NC(=O)c1ccccc1')
s = sascorer.calculateScore(mol)

import h5py
import pandas as pd
import ast
import pickle

from ddc_pub import ddc_v3 as ddc

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
            sas = sascorer.calculateScore(mol)

            descriptors = [logp, tpsa, sim, qed, hba, hbd, sas]
        except Exception as e:
            #print(e)
            return descriptors
    else:
        print("Invalid generation.")
    return descriptors

# 生成器
class mol_generator:
    def __init__(self, seed_smile:str = "", sub_smile:str = "", model = None, target_names:list = [], qsar_model:str = ""):
        self.mols = []
        self.target = []
        self.target_names = target_names
        self.sani_mols = []
        self.sani_properties = []
        self.data = None
        self.set_seed(seed_smile)
        self.set_sub(sub_smile)
        self.set_model(model)
        self.set_qsar_model(qsar_model)

    def set_seed(self, seed_smile):
        if seed_smile == "":
            return
        #self.seed_smile = seed_smile
        self.seed_mol = Chem.MolFromSmiles(seed_smile)

        print("Seed Molecular:")
        print(Chem.MolToSmiles(self.seed_mol))

    def set_sub(self, sub_smile):
        if sub_smile == "":
            return
        self.sub_mol = Chem.MolFromSmiles(sub_smile)

        print("Substruct:")
        print(Chem.MolToSmiles(self.sub_mol))

    def set_model(self, model):
        if model == "":
            return

        #根据model参数的类型，从文件载入模型或直接接收trainer的模型
        if type(model)==str:
            self.model = ddc.DDC(model_name=model)
        else:
            self.model = model

    def set_qsar_model(self, qsar_model):
        if(qsar_model == ""):
            return
        self.qsar_model = pickle.load(open(qsar_model, "rb"))["classifier_sv"]

    # 检查分子
    def sanitize(self, mol):  
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
        self.target = get_descriptors(self.seed_mol, self.sub_mol)
        for i in range(len(conditions)):
            if(conditions[i] != None):
                self.target[i] = conditions[i]
        self.target = np.array(self.target)
        print("Sampling with target:{:}.".format(self.target))
        # 采样
        smiles_out = []
        self.model.batch_input_length = 256  # 太大会减慢速度
        for i in range(sample_times):
            smiles, _ = self.model.predict_batch(latent=self.target.reshape(1, -1), temp=1.0)
            #print("#{:}:{:}".format(i,smiles))
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
                self.sani_properties.append(get_descriptors(sani_mol, self.sub_mol))
        # 打印结果
        print("生成分子数:{:},有效性:{:}".format(
            len(self.mols), len(self.sani_mols)/len(self.mols)))

    # 根据id显示生成的分子
    def showmol(self, i):  
        print(Chem.MolToSmiles(self.sani_mols[i]))

    # 筛选sub_similarity==1的分子
    def filter_data(self, filename: str=""):
        #筛选结果
        #print("Saving results.")
        self.binmols = np.asarray([[i.ToBinary() for i in self.sani_mols]])
        self.binmols_data = pd.DataFrame(self.binmols.T, columns=["binmol"], copy=True)
        self.properties_data = pd.DataFrame(self.sani_properties, columns=self.target_names, copy=True)
        self.filtered_data = self.binmols_data.loc[[i==1 for i in self.properties_data["sub_similarity"]]]
        print("Filtered {} mols.".format(len(self.filtered_data["binmol"])))
        #保存文件
        with h5py.File(filename, "w") as f:
            f.create_dataset("mols", data=np.asarray(self.filtered_data["binmol"]))
    
    # 导出数据
    def dump(self, mols_filename:str = "", properties_filename:str=""):
        with open(mols_filename, "wb") as f:
            pickle.dump(self.sani_mols, f)
        with open(properties_filename, "wb") as f:
            pickle.dump(self.sani_properties, f)

# 训练器
class model_trainer:
    def __init__(self, is_retrain:bool = False, sub_smile:str = "", model = None, binmols = None):
        self.binmols = []
        self.mols = []
        self.descr = []
        #设置子结构
        self.set_sub(sub_smile)
        #设置描述符和分子
        self.set_binmols(binmols)
        #建立模型
        self.set_model(is_retrain, model)

    def set_sub(self, sub_smile):
        if sub_smile == "":
            return
        self.sub_mol = Chem.MolFromSmiles(sub_smile)
    
    def set_binmols(self, binmols):
        if binmols is None:
            return
        print("Trying to load {} binmols.".format(len(binmols)))
        
        self.binmols = np.asarray(binmols)
        self.mols = np.asarray([Chem.Mol(binmol) for binmol in self.binmols])
        self.descr = np.asarray([get_descriptors(mol, self.sub_mol) for mol in self.mols])

        print("Binmols loaded.")
    
    def set_model(self, is_retrain:bool = False, model = None):
        self.model_name = "new_model"
        if is_retrain:
            if type(model) == str:
                self.model_name = model
                self.model = ddc.DDC(x=self.descr, y=self.binmols, model_name=self.model_name)
            else:
                self.model = model
            print("Loading existing model {}.".format(self.model_name))
        else:
            if type(model) == str:
                self.model_name = model
            self.model = None
            print("Created empty model {}.".format(self.model_name))
    
    # 如果是新模型，需要初始化模型
    def init_model(self, dataset_info=None, scaling=True, noise_std=0.1, lstm_dim=512, dec_layers=3, batch_size = 128):
        if dataset_info is None:
            print("Dataset info not set.")
            return

        self.model = ddc.DDC(x              = self.descr,  # input
                            y              = self.binmols, # output
                            dataset_info   = dataset_info, # dataset information
                            scaling        = scaling,      # scale the descriptors
                            noise_std      = noise_std,    # std of the noise layer
                            lstm_dim       = lstm_dim,     # breadth of LSTM layers
                            dec_layers     = dec_layers,   # number of decoding layers
                            batch_size     = batch_size)   # batch size for training
        print("New model initialized.")

    # 装载训练集
    def fit(self, epochs=300, lr=1e-3, mini_epochs=10, save_period=50, lr_delay=True, sch_epoch_to_start=500, lr_init=1e-3, lr_final=1e-6, patience=25):
        if self.model is None:
            print("Model to fit uninitialized.")
            return
        print("Training model.")

        self.model.fit(epochs              = epochs,                               # number of epochs
                        lr                  = lr,                                  # initial learning rate for Adam, recommended
                        model_name          = self.model_name,                     # base name to append the checkpoints with
                        checkpoint_dir      = "checkpoints_"+self.model_name+"/",  # save checkpoints in the notebook's directory
                        mini_epochs         = mini_epochs,                         # number of sub-epochs within an epoch to trigger lr decay
                        save_period         = save_period,                         # checkpoint frequency (in mini_epochs)
                        lr_decay            = lr_delay,                            # whether to use exponential lr decay or not
                        sch_epoch_to_start  = sch_epoch_to_start,                  # mini-epoch to start lr decay (bypassed if lr_decay=False)
                        sch_lr_init         = lr_init,                             # initial lr, should be equal to lr (bypassed if lr_decay=False)
                        sch_lr_final        = lr_final,                            # final lr before finishing training (bypassed if lr_decay=False)
                        patience            = patience)                            # patience for Keras' ReduceLROnPlateau (bypassed if lr_decay=True)

        print("Training completed.")
    
    # 保存模型
    def save_model(self, filename:str = ""):
        self.model.save(filename)
        print("Model saved.")

    # 导出训练器
    def dump(self, filename:str = ""):
        with open(filename, "wb") as f:
            pickle.dump(self, f)

# 公用变量
target_names = ["logp", "tpsa", "sub_similarity", "qed", "hba", "hbd", "sa_score"]      
sub_smiles = "O=C(OC)C1=CC(C2=O)CCCC2C1=O"
seed_smiles = "CC1(C)C(CCC2(C)C1CCC(C34C)(C)C2CC(C)(C4=O)C(C)=C(C(OC)=O)C3=O)=O"

# 训练变量
charset = "Brc1(-23[nH])45C=NOso#FlS67+89%0"
maxlen = 128
name = "ChEMBL25_TRAIN_0813"
dataset_info = {"charset": charset, "maxlen": maxlen, "name": name}

# 训练集
print("Loading dataset.")
#随机挑选5000个幸运分子
dataset_filename = r"datasets/CHEMBL25_TRAIN_MOLS.h5"
rnd_idx = random.randint(0, 1335000)
with h5py.File(dataset_filename, "r") as f:
    binmols_1 = np.asarray(f["mols"][rnd_idx:rnd_idx+5000])
#筛选的相似度0.3-1的分子
dataset_filename = r"datasets/CHEMBL25_FILTERED_2.h5"
with h5py.File(dataset_filename, "r") as f:
    binmols_2 = np.asarray(f["mols"])
#种子分子
binmols_3 = np.asarray([Chem.MolFromSmiles(seed_smiles).ToBinary()]*3)
binmols_list = [binmols_3, binmols_2, binmols_1]
#合并训练集
binmols = np.concatenate(binmols_list)
#binmols = binmols_1

# 训练
trainer_1 = model_trainer(is_retrain=True, sub_smile=sub_smiles, model="model_0822", binmols=binmols)
trainer_1.fit()
trainer_1.save_model("models/model_0823")

#生成
generator_1 = mol_generator(seed_smile=seed_smiles, sub_smile=sub_smiles, model=trainer_1.model, target_names=target_names)
generator_1.sample(100, [None]*6+[10.0])
generator_1.dump("datasets/generator_0823_m.pickle", "datasets/generator_0823_p.pickle")
