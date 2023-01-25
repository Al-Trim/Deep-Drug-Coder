# 换用GASA
# 尝试使用GC回收内存
# 采用新的相似度计算策略
# 阶梯式更改采样条件
# 更改筛选条件为sim>0.8 && 不和种子完全相同
# 加入分子量描述符
# 连续训练4次
# 更改sys.path判断条件
# 针对0921尝试去掉CHEMBL25训练
# 换用Dice
# 更改gasas筛选条件
# 更改dump、load方式
import gc, os, sys
import numpy as np
import random

import rdkit
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFMCS, AllChem, RDKFingerprint, Descriptors, rdMolDescriptors, AllChem, QED
from rdkit import DataStructs

default_path = os.getcwd()
if default_path[-4:] != "GASA":
    sys.path.append(os.path.join(default_path, "GASA"))
import gasa

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
        sim = DataStructs.FingerprintSimilarity(sub_mol_fp, mcs_mol_fp, metric=DataStructs.DiceSimilarity)

        return sim
    except Exception as e:
        #print(e)
        return 0

# 计算合成可及性
def get_gasas(mols: list=[]) -> list:
    if default_path[-4:] != "GASA":
        os.chdir(os.path.join(default_path, "GASA"))
    smiles_list = [Chem.MolToSmiles(mol) for mol in mols]
    gasas = gasa.GASA(smiles_list)[1]
    os.chdir(default_path)
    return gasas

# 计算除了gasas以外的描述符
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
            molwt = Descriptors.ExactMolWt(mol)

            descriptors = [logp, tpsa, sim, qed, hba, hbd, molwt]
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
        self.seed_fp = RDKFingerprint(seed_mol)

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
            if Chem.MolToSmiles(mol) == "":
                return None
            else:
                return mol
        except Exception as e:
            return None
            #print(e)

        # 计算目标
    def get_target(self, conditions: list=[]):
        target = get_descriptors(self.seed_mol, self.sub_mol)
        target.append(get_gasas([self.seed_mol])[0])
        #print(conditions, target)
        for idx,condition in enumerate(conditions):
            if idx >= len(target):
                break
            if condition != None:
                target[idx] = condition
        return np.asarray(target)

    # 采样指定性质的分子
    def sample(self, sample_times: int = 4, conditions: list = []):
        # 确定目标
        self.target = self.get_target(conditions)
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
        print("Checking mols.")
        self.sani_properties.clear()
        self.sani_fps = []
        sani_mols = []
        for idx,mol in enumerate(self.mols):
            sani_mol = self.sanitize(mol)
            if sani_mol != None:
                #去重
                mol_fp = RDKFingerprint(sani_mol)
                is_dupli = False
                for i in self.sani_fps:
                    sani_sim = DataStructs.FingerprintSimilarity(mol_fp, i)
                    if sani_sim == 1.0:
                        is_dupli = True
                        break
                if not is_dupli:
                    self.sani_fps.append(mol_fp)
                    sani_properties = get_descriptors(sani_mol, self.sub_mol)
                    sani_properties.insert(0, smiles_out[idx])
                    sani_properties.insert(0, sani_mol)
                    sani_mols.append(sani_mol)
                    self.sani_properties.append(sani_properties)
        
        # 计算GASA score
        print("Calculating GASA score for generated mols.")
        sani_gasas = get_gasas(sani_mols)
        for idx,properties in enumerate(self.sani_properties):
            properties.append(sani_gasas[idx])

        # 打印结果
        self.data = pd.DataFrame(self.sani_properties, columns=['mol','smiles']+self.target_names)
        print("Generated mols:{:}, sanitized mols:{:}, validity:{:}".format(len(self.mols), len(self.sani_properties), len(self.sani_properties)/len(self.mols)))

    
    # 根据id显示生成的分子
    def showmol(self, i):  
        print(Chem.MolToSmiles(self.sani_properties[i][0]))

    # 筛选分子
    def filter_data(self):
        #筛选结果
        #print("Saving results.")
        binmols = np.asarray([i[0].ToBinary() for i in self.sani_properties])
        combined_properties = []
        for idx,binmol in enumerate(binmols):
            combined_properties.append([binmol] + self.sani_properties[idx][2:])

        combined_data = pd.DataFrame(combined_properties, columns=['binmols']+self.target_names)
        #print(combined_data.keys())
        combined_data = combined_data.sort_values(['gasas'])
        #print(combined_data.sort_values(['sub_similarity']).tail(10)['sub_similarity'])

        filter_list = [i >= 0.8 for i in combined_data['sub_similarity']]
        '''for idx,sim in enumerate(combined_data['sub_similarity']):
            filter_list.append(float(sim) >= 0.8 and float(combined_data['gasas'][idx]) > 0.25)'''
        filtered_properties_data = combined_data.loc[filter_list]

        print("Filtered {} mols.".format(len(filtered_properties_data)))

        return (filtered_properties_data)
    
    # 导出数据
    def dump(self, filename:str = "", data = None):
        with open(filename, "wb") as f:
            pickle.dump(data, f)

    # 加载数据
    def load_data_old(self, mols_filename:str = "", properties_filename:str = ""):
        with open(mols_filename, "rb") as f:
            sani_mols = pickle.load(f)
        with open(properties_filename, "rb") as f:
            sani_properties = pickle.load(f)

        assert len(sani_mols) == len(sani_properties)
        for i in range(len(sani_mols)):
            sani_properties[i].insert(0, '')
            sani_properties[i].insert(0, sani_mols[i])

        self.sani_properties = sani_properties
        self.data = pd.DataFrame(self.sani_properties, columns=['mol','smiles']+self.target_names)
        print("Loaded {} mols with properties.".format(len(self.data))) 
    
    def load_data(self, properties_filename:str = ""):
        with open(properties_filename, "rb") as f:
            self.sani_properties = pickle.load(f)
        self.data = pd.DataFrame(self.sani_properties, columns=['mol','smiles']+self.target_names) 
        print("Loaded {} mols with properties.".format(len(self.data))) 

# 训练器
class model_trainer:
    def __init__(self, is_retrain:bool = False, sub_smile:str = "", model = None, binmols = None, descrs:list = []):
        self.binmols = []
        self.mols = []
        #设置子结构
        self.set_sub(sub_smile)
        #设置描述符和分子
        self.set_binmols(binmols)
        #设置描述符
        self.set_descrs(descrs)
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

        print("Binmols loaded.")
    
    def set_descrs(self, descrs:list=[]):
        if len(descrs) != 0:
            self.descr = np.asarray(descrs)
        else:
            print("Calculating descriptors.")
            self.descr = [get_descriptors(mol, self.sub_mol) for mol in self.mols]
            gasas = get_gasas(self.mols)
            for idx,descr in enumerate(self.descr):
                descr.append(gasas[idx])
            self.descr = np.asarray(self.descr)

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
        print("Saving model.")
        self.model.save(filename)

    # 导出训练器
    '''
    def dump(self, filename:str = ""):
        with open(filename, "wb") as f:
            pickle.dump(self, f)
    '''

# 从已知h5文件中读取分子及其描述符
def get_mols_and_descr(dataset_filename):

    with h5py.File(dataset_filename, "r") as f:
        binmols = np.asarray(f['mols'])

    with h5py.File(dataset_filename[:-3]+"_descrs.h5", "r") as f:
        names = target_names
        descrs = np.concatenate([[np.asarray(f[name]) for name in names]], axis=1)
        descrs = descrs.T

    return (binmols, descrs)

# 公用变量
target_names = ["logp", "tpsa", "sub_similarity", "qed", "hba", "hbd", "molwt", "gasas"]      
sub_smiles = "O=C(OC)C1=CC(C2=O)CCCC2C1=O"
sub_mol = Chem.MolFromSmiles(sub_smiles)
seed_smiles = "CC1(C)C(CCC2(C)C1CCC(C34C)(C)C2CC(C)(C4=O)C(C)=C(C(OC)=O)C3=O)=O"
seed_mol = Chem.MolFromSmiles(seed_smiles)

# 训练变量
charset = "Brc1(-23[nH])45C=NOso#FlS67+89%0"
maxlen = 128
name = "ChEMBL25_TRAIN_0813"
dataset_info = {"charset": charset, "maxlen": maxlen, "name": name}
'''
# 训练集
print("Loading dataset.")

# 筛选的相似度0.3-1的分子
binmols_2, descrs_2 = get_mols_and_descr("datasets/CHEMBL25_FILTERED_2.h5")
# 种子分子
binmols_3 = np.asarray([seed_mol.ToBinary()] * 3)
descrs_3 = np.asarray([get_descriptors(seed_mol, sub_mol) + get_gasas([seed_mol])] * 3)
# 合并
binmols_list =  [binmols_2]
descrs_list = [descrs_2]
binmols = np.concatenate(binmols_list)
descrs = np.vstack(descrs_list)

print("Running init round (init model)")
trainer = model_trainer(sub_smile=sub_smiles, model="model_1004_init", binmols=binmols, descrs=descrs)
trainer.init_model(dataset_info)
trainer.fit(lr=pow(10,-4.5), lr_init=pow(10,-4.5))
trainer.save_model("models/model_1004_init")
'''

for i in range(4):
    print("Running round {}.".format(i))
    gc.collect()

    #更新训练集
    filename = "datasets/generator_1003_{}.pickle".format(i)
    with open(filename, "rb") as f:
        sani_properties = pickle.load(f)
    
    binmols_list = [sani_property[0].ToBinary() for sani_property in sani_properties]
    descrs_list = [sani_property[2:] for sani_property in sani_properties]

    binmols = np.asarray(binmols_list[:20000])
    descrs = np.asarray(descrs_list[:20000])

    #重新训练
    if i == 0:
        model_name = "models/model_1004_init"
    else:
        model_name = "models/model_1004_retrain_{}".format(i-1)
    trainer = model_trainer(is_retrain=True, sub_smile=sub_smiles, model=model_name, binmols=binmols, descrs=descrs)
    trainer.fit(lr=pow(10,-4.5), lr_init=pow(10,-4.5))
    trainer.save_model("models/model_1004_retrain_{}".format(i))

