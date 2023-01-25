#生成含完整子结构的分子，并筛出子结构相似度100%的分子

import numpy as np
import pickle

from ddc_pub import ddc_v3 as ddc
#import molvecgen
import rdkit, h5py

from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdMolDescriptors, RDKFingerprint, QED, rdFMCS
from rdkit import rdBase
import pandas as pd

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

rdBase.DisableLog("rdApp.*")

#创建生成器
generator_1 = mol_generator(sub_smile="O=C(OC)C1=CC(C2=O)CCCC2C1=O") 

#设置种子
#generator_1.set_seed("CN(C)CCCN1C2=CC=CC=C2SC3=C1C=C(C=C3)Cl") 
#generator_1.set_seed("CC1(C)C(CC[C@@]2(C)[C@@]1([H])CC[C@@]([C@@]34C)(C)[C@]2([H])C[C@@](C)(C4=O)C(C)=C(C(OC)=O)C3=O)=O")
generator_1.set_seed("CC1(C)C(CCC2(C)C1CCC(C34C)(C)C2CC(C)(C4=O)C(C)=C(C(OC)=O)C3=O)=O")
#generator_1.set_sub("O=C(OC)C1=CC(C2=O)CCCC2C1=O")

#设置生成模型
generator_1.set_model("models/model_0811") 

#设置QSAR模型
#generator_1.set_qsar_model("models/qsar_model.pickle") 

#采样/生成
generator_1.sample(1000)

#筛选sub_similarity==1的分子
generator_1.filter_data("datasets/GENERATED_MOLS_2.h5")

quit()
import pickle
with open("gnr_mol_1.pickle","wb") as f:
    pickle.dump(generator_1.sani_mols, f)
with open("gnr_mol_1_p.pickle","wb") as f:
    pickle.dump(generator_1.sani_properties, f)
