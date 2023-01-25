import pandas as pd
import rdkit
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFMCS, AllChem, RDKFingerprint, Descriptors, rdMolDescriptors, AllChem, QED

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

# 加载0910
smiles_data = pd.read_csv("datasets/model_0910_final_smiles.csv")
sani_mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_data['smiles']]
print("Loading completed.")

#筛选
seed_mol = Chem.MolFromSmiles("CC1(C)C(CCC2(C)C1CCC(C34C)(C)C2CC(C)(C4=O)C(C)=C(C(OC)=O)C3=O)=O")
filter_list = [get_sim(mol, seed_mol) >= 0.7 for mol in sani_mols]
print("Calculating completed.")
filtered_smiles_data = smiles_data.loc[filter_list]
filtered_smiles_data.to_csv("datasets/model_0910_final_smiles_sim_0.7.csv")