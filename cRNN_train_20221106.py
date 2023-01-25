# 换用GASA
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
# 更换ddc_gt（2）
# 尝试重复1003
# 针对1027修改初始学习率
import os, sys
import numpy as np
import random

import rdkit
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFMCS, RDKFingerprint, Descriptors, rdMolDescriptors, AllChem, QED
from rdkit import DataStructs

default_path = os.getcwd()
if default_path[-4:] != "GASA":
    sys.path.append(os.path.join(default_path, "GASA"))
import gasa

import h5py
import pandas as pd
import ast
import pickle

from ddc_pub import ddc_gt

def get_sim(mol, sub_mol) -> float: 
    '''
    计算子结构相似度
    '''
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

def get_gasas(mols: list=[]) -> list:
    '''
    计算GASA打分
    '''
    if default_path[-4:] != "GASA":
        os.chdir(os.path.join(default_path, "GASA"))
    smiles_list = [Chem.MolToSmiles(mol) for mol in mols]
    gasas = gasa.GASA(smiles_list)[1]
    os.chdir(default_path)
    return gasas

def get_descriptors(mol_list, sub_mol):
    '''
    计算描述符
    顺序："logp", "tpsa", "sub_similarity", "qed", "hba", "hbd", "molwt", "gasas"
    '''
    print("Calculating descriptors for {} mols.".format(len(mol_list)))

    logp_list = [Descriptors.MolLogP(mol) for mol in mol_list]
    tpsa_list = [Descriptors.TPSA(mol) for mol in mol_list]
    sim_list = [get_sim(mol, sub_mol) for mol in mol_list]
    hba_list = [rdMolDescriptors.CalcNumHBA(mol) for mol in mol_list]
    hbd_list = [rdMolDescriptors.CalcNumHBD(mol) for mol in mol_list]
    qed_list = [QED.qed(mol) for mol in mol_list]
    molwt_list = [Descriptors.ExactMolWt(mol) for mol in mol_list]
    gasas_list = get_gasas(mol_list)

    descriptors = [logp_list, tpsa_list, sim_list, qed_list, hba_list, hbd_list, molwt_list, gasas_list]
    return np.asarray(descriptors).T

def filter_data(sani_properties, target_names):
    '''
    筛选sub_similarity>=0.8且按gasa打分排序
    '''
    #print("Saving results.")
    binmols = np.asarray([i[0].ToBinary() for i in sani_properties])
    combined_properties = []
    for idx,binmol in enumerate(binmols):
        combined_properties.append([binmol] + sani_properties[idx][2:])

    combined_data = pd.DataFrame(combined_properties, columns=['binmols']+target_names)
    #print(combined_data.keys())
    combined_data = combined_data.sort_values(['gasas'])
    #print(combined_data.sort_values(['sub_similarity']).tail(10)['sub_similarity'])

    filter_list = [i >= 0.8 for i in combined_data['sub_similarity']]
    '''for idx,sim in enumerate(combined_data['sub_similarity']):
        filter_list.append(float(sim) >= 0.8 and float(combined_data['gasas'][idx]) > 0.25)'''
    filtered_properties_data = combined_data.loc[filter_list]

    filtered_count = len(filtered_properties_data)
    print("Filtered {} mols.".format(filtered_count))

    return (filtered_properties_data, filtered_count)

def get_mols_and_descr(dataset_filename):
    '''
    从已知h5文件中读取分子及其描述符
    '''
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
conditions = [4.47190000e+00, 7.75100000e+01, 1.00000000e+00, 3.49746014e-01, 5.00000000e+00, 0.00000000e+00, 4.28256274e+02, 7.00000000e-01]

charset = "Brc1(-23[nH])45C=NOso#FlS67+89%0"
maxlen = 128
name = "ChEMBL25_TRAIN_1106"
dataset_info = {"charset": charset, "maxlen": maxlen, "name": name}

# 训练集
print("Loading dataset.")

# 筛选的相似度0.3-1的分子
binmols_2, descrs_2 = get_mols_and_descr("datasets/CHEMBL25_FILTERED_2.h5")
# 种子分子
binmols_3 = np.asarray([seed_mol.ToBinary()] * 3)
descrs_3 = np.asarray([conditions] * 3)
# 合并
binmols_list =  [binmols_3, binmols_2]
descrs_list = [descrs_3, descrs_2]
binmols = np.concatenate(binmols_list)
descrs = np.vstack(descrs_list)

gt = ddc_gt.ddc_gt(target_names=target_names, conditions=conditions)

report_names = ['Round', 'Total mols', 'Valid mols without duplicate mols', 'Duplicate mols', 'Validity', 'Filtered mols', 'Filtered rate']
report_list = []

'''
# 首训
for i in range(2):
    print("Running normal round {}.".format(i))

    #重新训练
    gt.init_model(x=descrs, y=binmols, dataset_info=dataset_info)
    gt.fit(model_name='model_1106_{}'.format(i), lr_init=pow(10,-4.5))
    gt.save_model("models/model_1106_{}".format(i))

    #生成分子
    target_gasas = 0.25 + i*0.2
    gt.conditions = [4.47190000e+00, 7.75100000e+01, 1.00000000e+00, 3.49746014e-01, 5.00000000e+00, 0.00000000e+00, 4.28256274e+02, target_gasas]
    if i == 0:
        sample_times=500
    else:
        sample_times=1000
    mols, total_count, val_count, dup_count, val = gt.generate(sample_times=sample_times, batch_input_length=256)
    properties = get_descriptors(mols, sub_mol)
    gt.set_properties(properties)
    gt.dump("datasets/generator_1106_{}.pickle".format(i), gt.sani_properties)
    gt.plot("results/model_1106_{}.png".format(i), title="model_1106_{}".format(i))

    #筛选训练集
    filtered_data, filtered_count = filter_data(gt.sani_properties, target_names)
    report_list.append(['Normal_{}'.format(i), total_count, val_count, dup_count, val, filtered_count, filtered_count/val_count])

    #计算新训练集
    filtered_binmols = filtered_data['binmols']
    filtered_descrs = np.asarray([filtered_data[name] for name in target_names]).T
    if i == 1:
        break

    #更新训练集
    binmols_list.append(filtered_binmols[:20000])
    descrs_list.append(filtered_descrs[:20000])
    binmols = np.concatenate(binmols_list)
    descrs = np.concatenate(descrs_list)

# retrain前的预训练
binmols = np.asarray(binmols_2)
descrs = np.asarray(descrs_2)

gt.init_model(x=descrs, y=binmols, dataset_info=dataset_info)
gt.fit(model_name='model_1106_retrain_0', lr=pow(10,-4.5), lr_init=pow(10,-4.5))
gt.save_model("models/model_1106_retrain_0")
'''

# 载入1103的normal train结果
gt.load_data("datasets/generator_1106_{}.pickle".format(1))
#筛选训练集
filtered_data, filtered_count = filter_data(gt.sani_properties, target_names)
#计算新训练集
filtered_binmols = filtered_data['binmols']
filtered_descrs = np.asarray([filtered_data[name] for name in target_names]).T

# 计算可以retrain的次数
trainset_count = len(filtered_binmols)
retrain_times = trainset_count // 20000

# retrain
for i in range(retrain_times - 1):
    print("Running retrain round {}.".format(i))

    #更新训练集
    binmols = np.asarray(filtered_binmols[20000*i : 20000*(i+1)])
    descrs = np.asarray(filtered_descrs[20000*i : 20000*(i+1)])

    #重训
    gt.retrain("models/model_1106_retrain_{}".format(i), x=descrs, y=binmols, epochs=100, lr=pow(10,-4.5), lr_init=pow(10,-4.5))
    gt.save_model("models/model_1106_retrain_{}".format(i+1))

    #生成分子
    gt.conditions = conditions
    mols, total_count, val_count, dup_count, val = gt.generate(sample_times=10, batch_input_length=128)
    properties = get_descriptors(mols, sub_mol)
    gt.set_properties(properties)
    gt.plot("results/model_1106_retrain_{}.png".format(i), title="model_1106_retrain_{}".format(i))
    
    #筛选训练集
    filtered_data, filtered_count = filter_data(gt.sani_properties, target_names)
    if val_count == 0:   #防止zero divide
        val_count = 0.1
    report_list.append(['Retrain_{}'.format(i), total_count, val_count, dup_count, val, filtered_count, filtered_count/val_count])

#导出报告
report_data = pd.DataFrame(data=report_list, columns=report_names)
report_data.to_csv('results/report_1106.csv')
